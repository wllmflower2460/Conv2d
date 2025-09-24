#!/usr/bin/env python3
"""
Download REAL datasets only for M1.6 Sprint.
No synthetic data - keeping real data pure.
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import zipfile
import tarfile
from typing import Dict, List

class RealDatasetDownloader:
    """Download only real-world datasets."""
    
    def __init__(self, base_path: str = "/mnt/ssd/Conv2d_Datasets"):
        self.base_path = Path(base_path)
        
    def download_pamap2(self):
        """
        Download PAMAP2 - Real human activity dataset.
        100Hz IMU from 9 subjects performing 18 activities.
        """
        print("\n" + "="*60)
        print("PAMAP2 - Real Human Activity Dataset")
        print("="*60)
        
        pamap2_dir = self.base_path / 'har_adapted' / 'pamap2'
        pamap2_dir.mkdir(parents=True, exist_ok=True)
        
        # Alternative download methods for PAMAP2
        urls = [
            # Try UCI direct link first
            "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
            # Alternative mirror
            "https://www.uni-mannheim.de/fileadmin/dws/research/Software_and_Datasets/PAMAP2_Dataset.zip",
        ]
        
        success = False
        for url in urls:
            try:
                print(f"Trying: {url}")
                zip_path = pamap2_dir / "PAMAP2_Dataset.zip"
                
                # Download with progress
                def download_hook(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(downloaded * 100 / total_size, 100)
                    sys.stdout.write(f"\r  Progress: {percent:.1f}%")
                    sys.stdout.flush()
                
                urllib.request.urlretrieve(url, zip_path, reporthook=download_hook)
                print("\n  ✅ Download complete")
                
                # Extract if successful
                if zip_path.stat().st_size > 0:
                    print("  Extracting...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(pamap2_dir)
                    print("  ✅ PAMAP2 ready!")
                    success = True
                    break
                    
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
        
        if not success:
            print("\n  Manual download required:")
            print("  1. Visit: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring")
            print(f"  2. Download to: {pamap2_dir}")
    
    def setup_legkilo(self):
        """
        Setup LegKilo Unitree Go1 - Real quadruped robot dataset.
        7 sequences from actual robot with IMU, joint encoders, LiDAR.
        """
        print("\n" + "="*60)
        print("LegKilo - Real Quadruped Robot Dataset")
        print("="*60)
        
        legkilo_dir = self.base_path / 'real_quadruped' / 'legkilo'
        legkilo_dir.mkdir(parents=True, exist_ok=True)
        
        # Create download script
        download_script = legkilo_dir / "download_legkilo.sh"
        
        script_content = """#!/bin/bash
# LegKilo Unitree Go1 Dataset Download Script
# Real quadruped robot data - 7 sequences, ~21GB total

echo "LegKilo Dataset Download"
echo "========================"
echo ""
echo "This dataset requires manual download from Google Drive."
echo ""
echo "Dataset URL: https://drive.google.com/drive/folders/1Egpj7FngTTPCeQDEzlbiK3iesPPZtqiM"
echo ""
echo "Please download these sequences:"
echo "  - corridor.bag (445s, 3.0GB) - Indoor corridor navigation"
echo "  - park.bag (532s, 3.5GB) - Outdoor park environment"
echo "  - slope.bag (423s, 2.8GB) - Inclined surfaces"
echo "  - grass.bag (387s, 2.6GB) - Grass terrain"
echo "  - stairs.bag (234s, 1.6GB) - Stair climbing"
echo "  - indoor.bag (456s, 3.0GB) - Indoor navigation"
echo "  - outdoor.bag (512s, 3.4GB) - Outdoor mixed terrain"
echo ""
echo "Save all files to: $(pwd)"
echo ""
echo "After downloading, run: python process_rosbags.py"
"""
        
        with open(download_script, 'w') as f:
            f.write(script_content)
        
        os.chmod(download_script, 0o755)
        
        print(f"  Download instructions saved to: {download_script}")
        print("  This is REAL robot data - requires manual Google Drive download")
        print(f"  Run: bash {download_script}")
    
    def download_animal_datasets(self):
        """
        Setup downloads for real animal locomotion datasets.
        """
        print("\n" + "="*60)
        print("Real Animal Locomotion Datasets")
        print("="*60)
        
        # Horse Gait Dataset
        horse_dir = self.base_path / 'animal_locomotion' / 'horse_gaits'
        horse_dir.mkdir(parents=True, exist_ok=True)
        
        horse_info = horse_dir / "DOWNLOAD_INFO.txt"
        with open(horse_info, 'w') as f:
            f.write("""Horse Gait Dataset - Real IMU from 120 horses
==============================================

Published in Scientific Reports (Nature)
Paper: https://www.nature.com/articles/s41598-020-73215-9

Dataset includes:
- 120 horses
- 7 IMU sensors per horse (200-500Hz)
- 8 different gaits
- 7,576 labeled strides
- Expert annotations synchronized with video

Download from paper's supplementary materials.
""")
        
        # Dog Behavior Dataset
        dog_dir = self.base_path / 'animal_locomotion' / 'dog_behavior'
        dog_dir.mkdir(parents=True, exist_ok=True)
        
        dog_info = dog_dir / "DOWNLOAD_INFO.txt"
        with open(dog_info, 'w') as f:
            f.write("""Dog Behavior Dataset - Real IMU from 45 dogs
=============================================

Mendeley Data Repository
URL: https://data.mendeley.com/datasets/vxhx934tbn/1

Dataset includes:
- 45 dogs
- ActiGraph GT9X (100Hz, 3-axis acc + 3-axis gyro)
- 7 behaviors (galloping, lying, sitting, sniffing, standing, trotting, walking)
- Collar and harness mounting positions
- Video-validated ground truth

Direct download available from Mendeley.
""")
        
        print(f"  ✅ Animal dataset info saved to:")
        print(f"     {horse_info}")
        print(f"     {dog_info}")
    
    def download_drone_datasets(self):
        """
        Setup drone datasets for dynamic validation.
        """
        print("\n" + "="*60)
        print("Drone Datasets - Dynamic Motion Validation")
        print("="*60)
        
        # UZH-FPV Dataset
        uzh_dir = self.base_path / 'dynamic_validation' / 'uzh_fpv'
        uzh_dir.mkdir(parents=True, exist_ok=True)
        
        uzh_script = uzh_dir / "download_uzh.sh"
        with open(uzh_script, 'w') as f:
            f.write("""#!/bin/bash
# UZH-FPV Drone Racing Dataset
# Aggressive trajectories up to 7.0 m/s

echo "Downloading UZH-FPV dataset..."
echo "Visit: https://fpv.ifi.uzh.ch/"
echo "Dataset includes:"
echo "  - 100Hz Snapdragon Flight IMU"
echo "  - Millimeter-accurate ground truth"
echo "  - Racing maneuvers"
echo ""
echo "Download sequences and extract to: $(pwd)"
""")
        os.chmod(uzh_script, 0o755)
        
        print(f"  ✅ Download scripts created")
        print(f"     UZH-FPV: {uzh_script}")
    
    def create_dataset_summary(self):
        """Create a summary of all real datasets."""
        summary_file = self.base_path / "REAL_DATASETS_M16.md"
        
        content = """# M1.6 Real-World Validation Datasets

## Week 1 - Immediate Access

### PAMAP2 (Human Activity)
- **Type**: Real human IMU data
- **Size**: 2.1GB
- **Sampling**: 100Hz
- **Sensors**: 3 Colibri IMUs (wrist, chest, ankle)
- **Activities**: 18 different activities
- **Adaptation**: Map to quadruped limb positions

### LegKilo (Unitree Go1)
- **Type**: Real quadruped robot
- **Size**: 21GB (7 sequences)
- **Sampling**: 50Hz IMU (hardware capable of 500Hz)
- **Sensors**: 9-axis IMU + joint encoders + LiDAR
- **Environments**: Corridors, parks, slopes, grass, stairs

## Week 2 - Animal Locomotion

### Horse Gaits
- **Type**: Real horse IMU data
- **Size**: 15GB
- **Sampling**: 200-500Hz
- **Sensors**: 7 IMUs per horse
- **Subjects**: 120 horses
- **Gaits**: 8 different (walk, trot, canter, tölt, etc.)

### Dog Behavior
- **Type**: Real dog IMU data
- **Size**: 5GB
- **Sampling**: 100Hz
- **Sensors**: ActiGraph GT9X
- **Subjects**: 45 dogs
- **Behaviors**: 7 classes

## Week 3 - Dynamic Validation

### UZH-FPV Drone Racing
- **Type**: Aggressive flight dynamics
- **Size**: 10GB
- **Sampling**: 100Hz IMU
- **Max Speed**: 7.0 m/s
- **Ground Truth**: Millimeter accuracy

### Blackbird Dataset
- **Type**: Large-scale drone flights
- **Size**: 25GB
- **Flights**: 168 sequences
- **Duration**: 10+ hours
- **Ground Truth**: 360Hz motion capture

## Expected Performance Drops

From synthetic baseline (88.98%):
- Semi-synthetic: -10% to -15%
- Real robot: -15% to -20%
- Cross-species: -20% to -30%
- Dynamic/extreme: -25% to -35%

Target: 70-85% on real data is EXCELLENT!
"""
        
        with open(summary_file, 'w') as f:
            f.write(content)
        
        print(f"\n✅ Dataset summary saved to: {summary_file}")
    
    def run_week1_downloads(self):
        """Execute Week 1 download plan."""
        print("\n" + "="*60)
        print("M1.6 Week 1 - Real Dataset Downloads")
        print("="*60)
        
        # 1. PAMAP2 - Can download automatically
        self.download_pamap2()
        
        # 2. LegKilo - Requires manual download
        self.setup_legkilo()
        
        # 3. Create summary
        self.create_dataset_summary()
        
        print("\n" + "="*60)
        print("Week 1 Setup Complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Check PAMAP2 download in har_adapted/pamap2/")
        print("2. Manually download LegKilo from Google Drive")
        print("3. Process datasets with preprocessing scripts")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download REAL datasets for M1.6")
    parser.add_argument("--base_path", type=str,
                       default="/mnt/ssd/Conv2d_Datasets",
                       help="Base path for real datasets")
    parser.add_argument("--week", type=int, default=1,
                       help="Week number (1, 2, or 3)")
    
    args = parser.parse_args()
    
    downloader = RealDatasetDownloader(args.base_path)
    
    if args.week == 1:
        downloader.run_week1_downloads()
    elif args.week == 2:
        downloader.download_animal_datasets()
        print("Week 2 animal datasets info created")
    elif args.week == 3:
        downloader.download_drone_datasets()
        print("Week 3 drone datasets info created")
    else:
        print(f"Invalid week: {args.week}")

if __name__ == "__main__":
    main()