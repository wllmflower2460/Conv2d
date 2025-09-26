"""
Disk caching system for quantized features and intermediate results.
Implements content-addressable storage with compression and deduplication.
"""

import os
import hashlib
import pickle
import gzip
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass, asdict
import numpy as np
import torch
import threading
from contextlib import contextmanager
import shutil


@dataclass
class CacheConfig:
    """Configuration for feature caching."""
    window_size: int
    stride: int
    quantization_levels: Tuple[int, ...]
    preprocessing_params: Dict[str, Any]
    model_architecture: str
    
    def to_hash(self) -> str:
        """Generate hash for cache key."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


@dataclass
class CacheEntry:
    """Cache entry metadata."""
    key: str
    size_bytes: int
    created_time: float
    last_accessed: float
    access_count: int
    compression_ratio: float
    data_shape: Tuple[int, ...]
    data_dtype: str


class FeatureCacheManager:
    """
    High-performance disk cache for quantized features and intermediate results.
    
    Features:
    - Content-addressable storage with deduplication
    - LZ4/gzip compression for space efficiency
    - LRU eviction policy with size limits
    - Thread-safe operations
    - Configurable cache locations and limits
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path] = "./cache/features",
        max_cache_size_gb: float = 50.0,
        compression_level: int = 6,
        enable_compression: bool = True,
        max_entries: int = 10000
    ):
        """
        Initialize feature cache manager.
        
        Args:
            cache_dir: Directory to store cached features
            max_cache_size_gb: Maximum cache size in GB
            compression_level: Compression level (1-9)
            enable_compression: Enable compression
            max_entries: Maximum number of cached entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size = int(max_cache_size_gb * 1024**3)
        self.compression_level = compression_level
        self.enable_compression = enable_compression
        self.max_entries = max_entries
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self.entries: Dict[str, CacheEntry] = {}
        self.current_size = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing metadata
        self._load_metadata()
        
        # Cleanup on startup
        self._cleanup_orphaned_files()
    
    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                for key, entry_data in data.get('entries', {}).items():
                    entry = CacheEntry(**entry_data)
                    self.entries[key] = entry
                    self.current_size += entry.size_bytes
                
            except Exception as e:
                print(f"Warning: Could not load cache metadata: {e}")
                self.entries = {}
                self.current_size = 0
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            metadata = {
                'entries': {k: asdict(v) for k, v in self.entries.items()},
                'total_size': self.current_size,
                'last_updated': time.time()
            }
            
            # Atomic write
            temp_file = self.metadata_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            temp_file.replace(self.metadata_file)
            
        except Exception as e:
            print(f"Warning: Could not save cache metadata: {e}")
    
    def _cleanup_orphaned_files(self):
        """Remove orphaned cache files not in metadata."""
        existing_files = set(self.cache_dir.glob("*.cache"))
        expected_files = {
            self.cache_dir / f"{key}.cache" 
            for key in self.entries.keys()
        }
        
        orphaned = existing_files - expected_files
        for orphan in orphaned:
            try:
                orphan.unlink()
                print(f"Removed orphaned cache file: {orphan.name}")
            except Exception as e:
                print(f"Warning: Could not remove orphaned file {orphan}: {e}")
    
    def _generate_cache_key(
        self,
        config: CacheConfig,
        data_hash: str
    ) -> str:
        """Generate cache key from config and data hash."""
        config_hash = config.to_hash()
        combined = f"{config_hash}_{data_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _compute_data_hash(
        self,
        data: Union[np.ndarray, torch.Tensor, List]
    ) -> str:
        """Compute hash of input data."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        elif isinstance(data, list):
            data = np.array(data)
        
        # Use content hash for deduplication
        return hashlib.sha256(data.tobytes()).hexdigest()
    
    def _compress_data(self, data: bytes) -> Tuple[bytes, float]:
        """Compress data and return compression ratio."""
        if not self.enable_compression:
            return data, 1.0
        
        original_size = len(data)
        compressed = gzip.compress(data, compresslevel=self.compression_level)
        compressed_size = len(compressed)
        
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        return compressed, compression_ratio
    
    def _decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress data."""
        if not self.enable_compression:
            return compressed_data
        
        return gzip.decompress(compressed_data)
    
    def _evict_lru_entries(self, required_space: int):
        """Evict least recently used entries to free space."""
        if not self.entries:
            return
        
        # Sort by last accessed time (LRU first)
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda x: x[1].last_accessed
        )
        
        freed_space = 0
        entries_to_remove = []
        
        for key, entry in sorted_entries:
            if freed_space >= required_space:
                break
            
            entries_to_remove.append(key)
            freed_space += entry.size_bytes
        
        # Remove entries
        for key in entries_to_remove:
            self._remove_entry(key)
        
        print(f"Evicted {len(entries_to_remove)} entries, freed {freed_space / 1024**2:.1f} MB")
    
    def _remove_entry(self, key: str):
        """Remove cache entry and file."""
        if key not in self.entries:
            return
        
        entry = self.entries[key]
        cache_file = self.cache_dir / f"{key}.cache"
        
        # Remove file
        if cache_file.exists():
            cache_file.unlink()
        
        # Update metadata
        self.current_size -= entry.size_bytes
        del self.entries[key]
    
    def _ensure_space(self, required_space: int):
        """Ensure sufficient cache space is available."""
        # Check total size limit
        if self.current_size + required_space > self.max_cache_size:
            space_to_free = (self.current_size + required_space) - self.max_cache_size
            self._evict_lru_entries(space_to_free)
        
        # Check entry count limit
        if len(self.entries) >= self.max_entries:
            entries_to_remove = len(self.entries) - self.max_entries + 1
            sorted_entries = sorted(
                self.entries.items(),
                key=lambda x: x[1].last_accessed
            )
            
            for key, _ in sorted_entries[:entries_to_remove]:
                self._remove_entry(key)
    
    def store(
        self,
        data: Union[np.ndarray, torch.Tensor, List],
        config: CacheConfig
    ) -> str:
        """
        Store data in cache with given configuration.
        
        Args:
            data: Data to cache
            config: Cache configuration
            
        Returns:
            Cache key for retrieval
        """
        with self._lock:
            # Compute hashes
            data_hash = self._compute_data_hash(data)
            cache_key = self._generate_cache_key(config, data_hash)
            
            # Check if already cached
            if cache_key in self.entries:
                # Update access time and count
                self.entries[cache_key].last_accessed = time.time()
                self.entries[cache_key].access_count += 1
                self._save_metadata()
                return cache_key
            
            # Serialize data
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            elif isinstance(data, list):
                data = np.array(data)
            
            serialized = pickle.dumps(data)
            compressed, compression_ratio = self._compress_data(serialized)
            
            # Ensure space is available
            self._ensure_space(len(compressed))
            
            # Write to disk
            cache_file = self.cache_dir / f"{cache_key}.cache"
            with open(cache_file, 'wb') as f:
                f.write(compressed)
            
            # Update metadata
            entry = CacheEntry(
                key=cache_key,
                size_bytes=len(compressed),
                created_time=time.time(),
                last_accessed=time.time(),
                access_count=1,
                compression_ratio=compression_ratio,
                data_shape=data.shape,
                data_dtype=str(data.dtype)
            )
            
            self.entries[cache_key] = entry
            self.current_size += entry.size_bytes
            
            self._save_metadata()
            
            return cache_key
    
    def retrieve(
        self,
        cache_key: str,
        return_torch: bool = False
    ) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """
        Retrieve data from cache.
        
        Args:
            cache_key: Cache key from store()
            return_torch: Return as PyTorch tensor
            
        Returns:
            Cached data or None if not found
        """
        with self._lock:
            if cache_key not in self.entries:
                return None
            
            entry = self.entries[cache_key]
            cache_file = self.cache_dir / f"{cache_key}.cache"
            
            if not cache_file.exists():
                # File missing, remove from metadata
                del self.entries[cache_key]
                self.current_size -= entry.size_bytes
                self._save_metadata()
                return None
            
            try:
                # Read and decompress
                with open(cache_file, 'rb') as f:
                    compressed = f.read()
                
                decompressed = self._decompress_data(compressed)
                data = pickle.loads(decompressed)
                
                # Update access statistics
                entry.last_accessed = time.time()
                entry.access_count += 1
                
                # Convert to torch tensor if requested
                if return_torch:
                    data = torch.from_numpy(data)
                
                self._save_metadata()
                return data
                
            except Exception as e:
                print(f"Error retrieving cache entry {cache_key}: {e}")
                self._remove_entry(cache_key)
                return None
    
    def get_or_compute(
        self,
        config: CacheConfig,
        compute_fn,
        compute_args: tuple = (),
        compute_kwargs: dict = None,
        return_torch: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Get cached result or compute and cache it.
        
        Args:
            config: Cache configuration
            compute_fn: Function to compute result if not cached
            compute_args: Arguments for compute_fn
            compute_kwargs: Keyword arguments for compute_fn
            return_torch: Return as PyTorch tensor
            
        Returns:
            Computed or cached result
        """
        if compute_kwargs is None:
            compute_kwargs = {}
        
        # Create a temporary hash based on function and args
        fn_name = getattr(compute_fn, '__name__', str(compute_fn))
        args_str = f"{fn_name}_{compute_args}_{compute_kwargs}"
        temp_hash = hashlib.md5(args_str.encode()).hexdigest()
        
        cache_key = self._generate_cache_key(config, temp_hash)
        
        # Try to retrieve from cache
        cached_result = self.retrieve(cache_key, return_torch=return_torch)
        if cached_result is not None:
            return cached_result
        
        # Compute result
        result = compute_fn(*compute_args, **compute_kwargs)
        
        # Store in cache (use actual result hash)
        actual_cache_key = self.store(result, config)
        
        # Return result
        if return_torch and isinstance(result, np.ndarray):
            return torch.from_numpy(result)
        elif not return_torch and isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy()
        
        return result
    
    @contextmanager
    def batch_operation(self):
        """Context manager for batch cache operations."""
        with self._lock:
            # Defer metadata saves during batch operation
            old_save = self._save_metadata
            save_needed = False
            
            def deferred_save():
                nonlocal save_needed
                save_needed = True
            
            self._save_metadata = deferred_save
            
            try:
                yield self
            finally:
                # Restore original save function
                self._save_metadata = old_save
                
                # Save metadata if needed
                if save_needed:
                    self._save_metadata()
    
    def invalidate_pattern(self, pattern: str):
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match against cache keys
        """
        with self._lock:
            keys_to_remove = []
            
            for key in self.entries:
                if pattern in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            if keys_to_remove:
                self._save_metadata()
                print(f"Invalidated {len(keys_to_remove)} cache entries matching '{pattern}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            if not self.entries:
                return {
                    'total_entries': 0,
                    'total_size_mb': 0,
                    'cache_hit_rate': 0,
                    'avg_compression_ratio': 0
                }
            
            total_accesses = sum(e.access_count for e in self.entries.values())
            avg_compression = np.mean([e.compression_ratio for e in self.entries.values()])
            
            return {
                'total_entries': len(self.entries),
                'total_size_mb': self.current_size / 1024**2,
                'max_size_mb': self.max_cache_size / 1024**2,
                'utilization': self.current_size / self.max_cache_size,
                'total_accesses': total_accesses,
                'avg_compression_ratio': avg_compression,
                'oldest_entry_age_hours': (time.time() - min(e.created_time for e in self.entries.values())) / 3600,
                'cache_directory': str(self.cache_dir)
            }
    
    def cleanup(self, max_age_hours: float = 168):  # 1 week default
        """
        Clean up old cache entries.
        
        Args:
            max_age_hours: Maximum age in hours for cache entries
        """
        with self._lock:
            cutoff_time = time.time() - (max_age_hours * 3600)
            keys_to_remove = []
            
            for key, entry in self.entries.items():
                if entry.created_time < cutoff_time:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            if keys_to_remove:
                self._save_metadata()
                print(f"Cleaned up {len(keys_to_remove)} old cache entries")
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            # Remove all files
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            
            # Clear metadata
            self.entries.clear()
            self.current_size = 0
            self._save_metadata()
            
            print("Cache cleared")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self._save_metadata()
        except:
            pass


# Global cache manager instance
_global_cache_manager: Optional[FeatureCacheManager] = None


def get_global_cache_manager() -> FeatureCacheManager:
    """Get or create global cache manager instance."""
    global _global_cache_manager
    
    if _global_cache_manager is None:
        cache_dir = os.environ.get('CONV2D_CACHE_DIR', './cache/features')
        max_size_gb = float(os.environ.get('CONV2D_CACHE_SIZE_GB', '50'))
        
        _global_cache_manager = FeatureCacheManager(
            cache_dir=cache_dir,
            max_cache_size_gb=max_size_gb
        )
    
    return _global_cache_manager


def cached_quantization(
    model_fn,
    data: Union[np.ndarray, torch.Tensor],
    config: CacheConfig
) -> Union[np.ndarray, torch.Tensor]:
    """
    Cached FSQ quantization with automatic cache management.
    
    Args:
        model_fn: Quantization model function
        data: Input data
        config: Cache configuration
        
    Returns:
        Quantized features
    """
    cache_manager = get_global_cache_manager()
    
    return cache_manager.get_or_compute(
        config=config,
        compute_fn=model_fn,
        compute_args=(data,),
        return_torch=isinstance(data, torch.Tensor)
    )