"""
Pinned memory manager for efficient CPU→GPU transfers.
Optimizes data movement in the Conv2d pipeline.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple, List
import threading
import queue
import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    pinned_allocated: int = 0
    pinned_cached: int = 0
    gpu_allocated: int = 0
    gpu_cached: int = 0
    cpu_to_gpu_transfers: int = 0
    transfer_time_ms: float = 0.0


class PinnedMemoryManager:
    """
    Efficient pinned memory manager for CPU→GPU transfers.
    
    Features:
    - Pinned memory allocation and caching
    - Asynchronous non-blocking transfers
    - Memory pool management
    - Transfer statistics tracking
    """
    
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        max_pinned_memory: int = 2 * 1024**3,  # 2GB default
        enable_async: bool = True
    ):
        """
        Initialize pinned memory manager.
        
        Args:
            device: Target device for transfers
            max_pinned_memory: Maximum pinned memory in bytes
            enable_async: Enable asynchronous transfers
        """
        self.device = torch.device(device)
        self.max_pinned_memory = max_pinned_memory
        self.enable_async = enable_async and torch.cuda.is_available()
        
        # Memory pools
        self._pinned_cache: Dict[Tuple[torch.Size, torch.dtype], List[torch.Tensor]] = {}
        self._gpu_cache: Dict[Tuple[torch.Size, torch.dtype], List[torch.Tensor]] = {}
        
        # Statistics
        self._stats = MemoryStats()
        self._lock = threading.Lock()
        
        # Async transfer queue
        if self.enable_async:
            self._transfer_queue = queue.Queue()
            self._transfer_thread = threading.Thread(
                target=self._transfer_worker,
                daemon=True
            )
            self._transfer_thread.start()
            self._stream = torch.cuda.Stream()
        else:
            self._stream = None
    
    def allocate_pinned(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Allocate or reuse pinned memory tensor.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            
        Returns:
            Pinned memory tensor
        """
        key = (torch.Size(shape), dtype)
        
        with self._lock:
            # Try to reuse from cache
            if key in self._pinned_cache and self._pinned_cache[key]:
                tensor = self._pinned_cache[key].pop()
                return tensor
            
            # Check memory limit
            required_bytes = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
            if self._stats.pinned_allocated + required_bytes > self.max_pinned_memory:
                self._cleanup_pinned_cache()
            
            # Allocate new pinned tensor
            tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
            self._stats.pinned_allocated += required_bytes
            
            return tensor
    
    def free_pinned(self, tensor: torch.Tensor):
        """
        Return pinned tensor to cache for reuse.
        
        Args:
            tensor: Pinned tensor to cache
        """
        if not tensor.is_pinned():
            return
        
        key = (tensor.shape, tensor.dtype)
        
        with self._lock:
            if key not in self._pinned_cache:
                self._pinned_cache[key] = []
            
            # Cache for reuse (limit cache size per shape/dtype)
            if len(self._pinned_cache[key]) < 5:
                self._pinned_cache[key].append(tensor)
                self._stats.pinned_cached += tensor.numel() * tensor.element_size()
            else:
                # Actually free the tensor
                self._stats.pinned_allocated -= tensor.numel() * tensor.element_size()
                del tensor
    
    def to_gpu_async(
        self,
        cpu_tensor: Union[torch.Tensor, np.ndarray],
        non_blocking: bool = True
    ) -> torch.Tensor:
        """
        Transfer tensor to GPU asynchronously.
        
        Args:
            cpu_tensor: CPU tensor or numpy array
            non_blocking: Use non-blocking transfer
            
        Returns:
            GPU tensor
        """
        start_time = time.perf_counter()
        
        # Convert numpy to tensor if needed
        if isinstance(cpu_tensor, np.ndarray):
            cpu_tensor = torch.from_numpy(cpu_tensor)
        
        # Ensure tensor is pinned for fast transfer
        if not cpu_tensor.is_pinned():
            pinned = self.allocate_pinned(cpu_tensor.shape, cpu_tensor.dtype)
            pinned.copy_(cpu_tensor)
            cpu_tensor = pinned
        
        # Transfer to GPU
        if self.enable_async and non_blocking:
            with torch.cuda.stream(self._stream):
                gpu_tensor = cpu_tensor.to(
                    self.device,
                    non_blocking=True
                )
        else:
            gpu_tensor = cpu_tensor.to(self.device)
        
        # Update statistics
        transfer_time = (time.perf_counter() - start_time) * 1000
        with self._lock:
            self._stats.cpu_to_gpu_transfers += 1
            self._stats.transfer_time_ms += transfer_time
        
        return gpu_tensor
    
    def batch_to_gpu(
        self,
        tensors: List[Union[torch.Tensor, np.ndarray]],
        non_blocking: bool = True
    ) -> List[torch.Tensor]:
        """
        Transfer batch of tensors to GPU efficiently.
        
        Args:
            tensors: List of CPU tensors/arrays
            non_blocking: Use non-blocking transfers
            
        Returns:
            List of GPU tensors
        """
        gpu_tensors = []
        
        if self.enable_async and non_blocking:
            with torch.cuda.stream(self._stream):
                for cpu_tensor in tensors:
                    gpu_tensor = self.to_gpu_async(cpu_tensor, non_blocking=True)
                    gpu_tensors.append(gpu_tensor)
                
                # Synchronize stream to ensure all transfers complete
                self._stream.synchronize()
        else:
            for cpu_tensor in tensors:
                gpu_tensor = self.to_gpu_async(cpu_tensor, non_blocking=False)
                gpu_tensors.append(gpu_tensor)
        
        return gpu_tensors
    
    def prefetch_to_gpu(
        self,
        cpu_tensor: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Prefetch tensor to GPU in background.
        
        Args:
            cpu_tensor: CPU tensor to prefetch
            
        Returns:
            Future GPU tensor (may not be ready immediately)
        """
        if not self.enable_async:
            return self.to_gpu_async(cpu_tensor, non_blocking=False)
        
        # Add to transfer queue
        gpu_tensor = self.to_gpu_async(cpu_tensor, non_blocking=True)
        
        return gpu_tensor
    
    def _transfer_worker(self):
        """Background worker for async transfers."""
        while True:
            try:
                task = self._transfer_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                # Process transfer task
                task()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transfer worker error: {e}")
    
    def _cleanup_pinned_cache(self):
        """Clean up pinned memory cache when near limit."""
        with self._lock:
            # Remove half the cached tensors to free memory
            for key in list(self._pinned_cache.keys()):
                cached_tensors = self._pinned_cache[key]
                n_remove = len(cached_tensors) // 2
                
                for _ in range(n_remove):
                    if cached_tensors:
                        tensor = cached_tensors.pop()
                        self._stats.pinned_allocated -= tensor.numel() * tensor.element_size()
                        self._stats.pinned_cached -= tensor.numel() * tensor.element_size()
                        del tensor
    
    @contextmanager
    def cuda_stream_context(self):
        """Context manager for CUDA stream operations."""
        if self.enable_async and self._stream is not None:
            with torch.cuda.stream(self._stream):
                yield self._stream
        else:
            yield None
    
    def get_stats(self) -> MemoryStats:
        """Get memory usage statistics."""
        with self._lock:
            # Update GPU memory stats
            if torch.cuda.is_available():
                self._stats.gpu_allocated = torch.cuda.memory_allocated(self.device)
                self._stats.gpu_cached = torch.cuda.memory_reserved(self.device)
            
            return self._stats
    
    def clear_cache(self):
        """Clear all memory caches."""
        with self._lock:
            # Clear pinned cache
            for tensors in self._pinned_cache.values():
                for tensor in tensors:
                    del tensor
            self._pinned_cache.clear()
            
            # Clear GPU cache
            for tensors in self._gpu_cache.values():
                for tensor in tensors:
                    del tensor
            self._gpu_cache.clear()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Reset stats
            self._stats = MemoryStats()
    
    def shutdown(self):
        """Shutdown the memory manager."""
        if self.enable_async:
            # Signal worker thread to stop
            self._transfer_queue.put(None)
            self._transfer_thread.join(timeout=5.0)
        
        self.clear_cache()


class DataLoader:
    """
    High-performance data loader with pinned memory optimization.
    
    Optimized for Conv2d pipeline data loading patterns.
    """
    
    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        memory_manager: Optional[PinnedMemoryManager] = None,
        prefetch_factor: int = 2,
        num_workers: int = 0
    ):
        """
        Initialize optimized data loader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            memory_manager: Memory manager instance
            prefetch_factor: Number of batches to prefetch
            num_workers: Number of worker processes
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.memory_manager = memory_manager or PinnedMemoryManager()
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        
        self._prefetch_queue = queue.Queue(maxsize=prefetch_factor)
        self._prefetch_thread = None
        
        if prefetch_factor > 0:
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_worker,
                daemon=True
            )
            self._prefetch_thread.start()
    
    def __iter__(self):
        """Iterate over batches with optimized memory transfers."""
        for i in range(0, len(self.dataset), self.batch_size):
            end_idx = min(i + self.batch_size, len(self.dataset))
            
            # Get batch data
            batch_data = []
            batch_labels = []
            
            for j in range(i, end_idx):
                data, label = self.dataset[j]
                batch_data.append(data)
                batch_labels.append(label)
            
            # Convert to tensors with pinned memory
            batch_tensor = torch.stack(batch_data)
            label_tensor = torch.tensor(batch_labels)
            
            # Allocate pinned memory and copy
            pinned_data = self.memory_manager.allocate_pinned(
                batch_tensor.shape, batch_tensor.dtype
            )
            pinned_labels = self.memory_manager.allocate_pinned(
                label_tensor.shape, label_tensor.dtype
            )
            
            pinned_data.copy_(batch_tensor)
            pinned_labels.copy_(label_tensor)
            
            yield pinned_data, pinned_labels
            
            # Return to cache for reuse
            self.memory_manager.free_pinned(pinned_data)
            self.memory_manager.free_pinned(pinned_labels)
    
    def _prefetch_worker(self):
        """Background worker for data prefetching."""
        # Implementation for prefetching batches
        pass
    
    def __len__(self):
        """Number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def optimize_tensor_transfers():
    """
    Apply global optimizations for tensor transfers.
    
    Sets optimal CUDA settings for the Conv2d pipeline.
    """
    if torch.cuda.is_available():
        # Enable tensor core operations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Optimize memory allocation
        torch.cuda.empty_cache()
        
        # Set memory fraction if not already set
        if not hasattr(torch.cuda, '_memory_fraction_set'):
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.cuda._memory_fraction_set = True
        
        print(f"GPU optimizations enabled for {torch.cuda.get_device_name()}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, skipping GPU optimizations")


# Global memory manager instance
_global_memory_manager: Optional[PinnedMemoryManager] = None


def get_global_memory_manager() -> PinnedMemoryManager:
    """Get or create global memory manager instance."""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = PinnedMemoryManager()
    
    return _global_memory_manager


def free_global_memory_manager():
    """Free global memory manager."""
    global _global_memory_manager
    
    if _global_memory_manager is not None:
        _global_memory_manager.shutdown()
        _global_memory_manager = None