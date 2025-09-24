#!/usr/bin/env python3
"""
Central dataset manager for M1.6 Sprint.
Handles downloads, processing, and backup for all validation datasets.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import json
from datetime import datetime

class DatasetManager:
    """Manage all Conv2d validation datasets."""
    
    def __init__(self, 
                 ssd_path: str = "/mnt/ssd/Conv2d_Datasets",
                 backup_path: str = "/mnt/raid1/Conv2d_Datasets_Backup"):
        """
        Initialize dataset manager.
        
        Args:
            ssd_path: Primary storage on SSD
            backup_path: Backup location on RAID
        """
        self.ssd_path = Path(ssd_path)
        self.backup_path = Path(backup_path)
        
        # Dataset configurations
        self.datasets = {
            'tartanvo': {
                'type': 'semi_synthetic',
                'size_gb': 30,
                'environments': ['Downtown', 'OldTown', 'Hospital', 'Neighborhood'],
                'download_script': 'download_tartanvo.py'
            },
            'legkilo': {
                'type': 'real_quadruped',
                'size_gb': 21,
                'url': 'https://drive.google.com/drive/folders/1Egpj7FngTTPCeQDEzlbiK3iesPPZtqiM',
                'sequences': 7
            },
            'pamap2': {
                'type': 'har_adapted',
                'size_gb': 2.1,
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/pamap2/',
                'activities': 18
            },
            'horse_gaits': {
                'type': 'animal_locomotion',
                'size_gb': 15,
                'paper': 'https://www.nature.com/articles/s41598-020-73215-9',
                'gaits': 8
            },
            'dog_behavior': {
                'type': 'animal_locomotion',
                'size_gb': 5,
                'url': 'https://data.mendeley.com/datasets/vxhx934tbn/1',
                'behaviors': 7
            },
            'cear_mini_cheetah': {
                'type': 'real_quadruped',
                'size_gb': 20,
                'paper': 'arXiv:2404.04698',
                'environments': 31
            },
            'uzh_fpv': {
                'type': 'dynamic_validation',
                'size_gb': 10,
                'url': 'https://fpv.ifi.uzh.ch/',
                'max_speed_ms': 7.0
            },
            'blackbird': {
                'type': 'dynamic_validation', 
                'size_gb': 25,
                'url': 'http://blackbird-dataset.mit.edu/',
                'flights': 168
            }
        }
        
        self.status_file = self.ssd_path / "dataset_status.json"
        self.load_status()
    
    def load_status(self):
        """Load dataset download status."""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                self.status = json.load(f)
        else:
            self.status = {name: {'downloaded': False, 'processed': False} 
                          for name in self.datasets.keys()}
    
    def save_status(self):
        """Save dataset download status."""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def check_storage(self) -> Dict[str, float]:
        """Check available storage on SSD and RAID."""
        stats = {}
        
        for path, name in [(self.ssd_path, 'SSD'), (self.backup_path, 'RAID')]:
            if path.exists():
                statvfs = os.statvfs(path)
                total_gb = (statvfs.f_frsize * statvfs.f_blocks) / (1024**3)
                free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
                used_gb = total_gb - free_gb
                
                stats[name] = {
                    'total_gb': total_gb,
                    'used_gb': used_gb,
                    'free_gb': free_gb,
                    'usage_percent': (used_gb / total_gb) * 100
                }
        
        return stats
    
    def estimate_requirements(self, datasets: List[str]) -> float:
        """Estimate storage required for specified datasets."""
        total_gb = sum(self.datasets[name]['size_gb'] for name in datasets)
        return total_gb
    
    def download_week1(self):
        """Download Week 1 datasets (TartanVO, PAMAP2, LegKilo)."""
        print("\n" + "="*60)
        print("M1.6 Sprint - Week 1 Dataset Downloads")
        print("="*60)
        
        week1 = ['tartanvo', 'pamap2', 'legkilo']
        required_gb = self.estimate_requirements(week1)
        
        storage = self.check_storage()
        if 'SSD' in storage:
            print(f"\nStorage Status:")
            print(f"  SSD Free: {storage['SSD']['free_gb']:.1f} GB")
            print(f"  Required: {required_gb:.1f} GB")
            
            if storage['SSD']['free_gb'] < required_gb * 1.2:  # 20% buffer
                print("⚠️  Warning: Low storage space!")
                return
        
        # Download TartanVO
        if not self.status['tartanvo']['downloaded']:
            print("\n1. Downloading TartanVO...")
            self._download_tartanvo()
        
        # Download PAMAP2
        if not self.status['pamap2']['downloaded']:
            print("\n2. Downloading PAMAP2...")
            self._download_pamap2()
        
        # Download LegKilo
        if not self.status['legkilo']['downloaded']:
            print("\n3. Downloading LegKilo...")
            self._download_legkilo()
        
        print("\n✅ Week 1 downloads complete!")
        self.save_status()
    
    def _download_tartanvo(self):
        """Download TartanVO using dedicated script."""
        script_path = Path(__file__).parent / "download_tartanvo.py"
        
        for env in ['Downtown', 'OldTown']:
            cmd = [
                'python', str(script_path),
                '--env', env,
                '--base_path', str(self.ssd_path / 'semi_synthetic' / 'tartanvo'),
                '--process'
            ]
            
            print(f"  Downloading {env}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"    ✅ {env} downloaded")
            else:
                print(f"    ❌ {env} failed: {result.stderr}")
        
        self.status['tartanvo']['downloaded'] = True
    
    def _download_pamap2(self):
        """Download and process PAMAP2 dataset."""
        import urllib.request
        import zipfile
        
        pamap2_dir = self.ssd_path / 'har_adapted' / 'pamap2'
        pamap2_dir.mkdir(parents=True, exist_ok=True)
        
        # Download PAMAP2 - using correct URL
        url = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
        zip_path = pamap2_dir / "PAMAP2_Dataset.zip"
        
        if not zip_path.exists():
            print("  Downloading PAMAP2 dataset...")
            urllib.request.urlretrieve(url, zip_path)
            print("    ✅ Download complete")
        
        # Extract
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(pamap2_dir)
        
        self.status['pamap2']['downloaded'] = True
        print("    ✅ PAMAP2 ready")
    
    def _download_legkilo(self):
        """Download LegKilo dataset instructions (manual download required)."""
        legkilo_dir = self.ssd_path / 'real_quadruped' / 'legkilo'
        legkilo_dir.mkdir(parents=True, exist_ok=True)
        
        instructions_file = legkilo_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        
        instructions = """
LegKilo Unitree Go1 Dataset - Manual Download Required

The LegKilo dataset requires manual download from Google Drive:

1. Visit: https://drive.google.com/drive/folders/1Egpj7FngTTPCeQDEzlbiK3iesPPZtqiM
2. Download all 7 sequences to: {path}
3. Run: python scripts/process_legkilo.py

Sequences to download:
- corridor (445s, 3.0GB)
- park (532s, 3.5GB)  
- slope (423s, 2.8GB)
- grass (387s, 2.6GB)
- stairs (234s, 1.6GB)
- indoor (456s, 3.0GB)
- outdoor (512s, 3.4GB)

Total: ~21GB
""".format(path=legkilo_dir)
        
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        print(f"  ⚠️  LegKilo requires manual download")
        print(f"     Instructions saved to: {instructions_file}")
        
        # Mark as needing manual intervention
        self.status['legkilo']['downloaded'] = 'manual_required'
    
    def backup_to_raid(self, dataset_name: str):
        """Backup a dataset to RAID storage."""
        src = self.ssd_path / self.datasets[dataset_name]['type'] / dataset_name
        dst = self.backup_path / self.datasets[dataset_name]['type'] / dataset_name
        
        if not src.exists():
            print(f"Source not found: {src}")
            return
        
        print(f"Backing up {dataset_name} to RAID...")
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rsync for efficient backup
        cmd = ['rsync', '-av', '--progress', str(src) + '/', str(dst) + '/']
        subprocess.run(cmd)
        
        print(f"✅ Backup complete: {dst}")
    
    def generate_report(self):
        """Generate dataset status report."""
        print("\n" + "="*60)
        print("M1.6 Dataset Status Report")
        print("="*60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Storage status
        storage = self.check_storage()
        if 'SSD' in storage:
            s = storage['SSD']
            print(f"SSD Storage:")
            print(f"  Total: {s['total_gb']:.1f} GB")
            print(f"  Used:  {s['used_gb']:.1f} GB ({s['usage_percent']:.1f}%)")
            print(f"  Free:  {s['free_gb']:.1f} GB")
        
        # Dataset status
        print("\nDataset Status:")
        print("-"*60)
        print(f"{'Dataset':<20} {'Type':<15} {'Size':<8} {'Downloaded':<12} {'Processed'}")
        print("-"*60)
        
        for name, info in self.datasets.items():
            status = self.status.get(name, {})
            downloaded = "✅" if status.get('downloaded') == True else "❌"
            if status.get('downloaded') == 'manual_required':
                downloaded = "⚠️ Manual"
            processed = "✅" if status.get('processed') else "❌"
            
            print(f"{name:<20} {info['type']:<15} {info['size_gb']:>6.1f}GB {downloaded:<12} {processed}")
        
        print("-"*60)
        total_size = sum(d['size_gb'] for d in self.datasets.values())
        print(f"{'Total':<20} {'':<15} {total_size:>6.1f}GB")
        print()

def main():
    """Main entry point for dataset manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Conv2d M1.6 Dataset Manager")
    parser.add_argument("--action", choices=['week1', 'week2', 'week3', 'backup', 'report'],
                       default='report', help="Action to perform")
    parser.add_argument("--dataset", type=str, help="Specific dataset for backup")
    
    args = parser.parse_args()
    
    manager = DatasetManager()
    
    if args.action == 'week1':
        manager.download_week1()
    elif args.action == 'week2':
        print("Week 2 downloads not yet implemented")
    elif args.action == 'week3':
        print("Week 3 downloads not yet implemented")
    elif args.action == 'backup':
        if args.dataset:
            manager.backup_to_raid(args.dataset)
        else:
            print("Please specify --dataset for backup")
    elif args.action == 'report':
        manager.generate_report()

if __name__ == "__main__":
    main()