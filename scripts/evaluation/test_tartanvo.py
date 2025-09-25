#!/usr/bin/env python3
"""
Test TartanVO download to verify it's working.
"""

import tartanair as ta
from pathlib import Path

# Initialize
base_path = Path("/mnt/ssd/Conv2d_Datasets/semi_synthetic/tartanvo")
base_path.mkdir(parents=True, exist_ok=True)

ta.init(str(base_path))

print("Checking available environments...")
# This should list what's available

# Try to download a small test
print("\nDownloading Downtown easy IMU data...")
try:
    ta.download(
        env="Downtown",
        difficulty=["easy"],
        modality=["imu"],
        unzip=True
    )
    print("✅ Download successful!")
    
    # Check what was downloaded
    import os
    for root, dirs, files in os.walk(base_path):
        level = root.replace(str(base_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{Path(root).name}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files)-5} more files")
            
except Exception as e:
    print(f"❌ Error: {e}")