#!/usr/bin/env python3
"""
Complete CVAT to Training Pipeline Automation
Runs the full 3-step pipeline:
1. CVAT individual points -> skeleton format
2. CVAT skeleton -> SLEAP dataset
3. SLEAP dataset -> TCN-VAE training format

Usage:
  python run_full_pipeline.py --cvat annotations.xml --images /path/to/images --output /path/to/training/data
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

class CVATTrainingPipeline:
    """Orchestrates the complete pipeline from CVAT to training"""

    def __init__(self):
        self.temp_files = []

    def run_pipeline(self, cvat_xml: str, images_dir: str, output_dir: str,
                    sequence_length: int = 30, stride: int = 10):
        """Run the complete 3-step pipeline"""

        print("🚀 Starting CVAT → Training Pipeline")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Input: {cvat_xml}")
        print(f"🖼️  Images: {images_dir}")
        print(f"📂 Output: {output_dir}")
        print()

        try:
            # Step 1: Individual points to skeleton
            skeleton_xml = self.step1_points_to_skeleton(cvat_xml)

            # Step 2: Skeleton to SLEAP
            sleap_file = self.step2_skeleton_to_sleap(skeleton_xml, images_dir)

            # Step 3: SLEAP to training format
            training_data = self.step3_sleap_to_training(sleap_file, output_dir,
                                                       sequence_length, stride)

            # Cleanup temporary files
            self.cleanup()

            print("🎉 Pipeline completed successfully!")
            print(f"📂 Training data available at: {output_dir}")

        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            self.cleanup()
            raise

    def step1_points_to_skeleton(self, cvat_xml: str) -> str:
        """Step 1: Convert individual points to skeleton format"""
        print("📍 Step 1: Converting individual points to skeleton format...")

        # Create temporary skeleton XML file
        input_path = Path(cvat_xml)
        skeleton_xml = input_path.parent / f"skeleton_{input_path.stem}.xml"
        self.temp_files.append(skeleton_xml)

        # Run the points-to-skeleton converter
        cmd = [
            sys.executable, "cvat_points_to_skeleton.py",
            "--input", str(cvat_xml),
            "--output", str(skeleton_xml)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Step 1 failed: {result.stderr}")

        print(f"✅ Step 1 complete: {skeleton_xml}")
        return str(skeleton_xml)

    def step2_skeleton_to_sleap(self, skeleton_xml: str, images_dir: str) -> str:
        """Step 2: Convert skeleton XML to SLEAP format"""
        print("🦴 Step 2: Converting skeleton to SLEAP format...")

        # Create SLEAP output file
        xml_path = Path(skeleton_xml)
        sleap_file = xml_path.parent / f"dataset_{xml_path.stem}.slp"
        self.temp_files.append(sleap_file)

        # Run the SLEAP converter
        cmd = [
            sys.executable, "cvat_to_sleap_converter.py",
            "--cvat", str(skeleton_xml),
            "--images", str(images_dir),
            "--output", str(sleap_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Step 2 failed: {result.stderr}")

        print(f"✅ Step 2 complete: {sleap_file}")
        return str(sleap_file)

    def step3_sleap_to_training(self, sleap_file: str, output_dir: str,
                               sequence_length: int, stride: int) -> str:
        """Step 3: Convert SLEAP to training format"""
        print("🎯 Step 3: Converting SLEAP to training format...")

        # Run the training integration
        cmd = [
            sys.executable, "sleap_to_training_integration.py",
            "--sleap", str(sleap_file),
            "--output", str(output_dir),
            "--sequence-length", str(sequence_length),
            "--stride", str(stride),
            "--create-loader"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Step 3 failed: {result.stderr}")

        print(f"✅ Step 3 complete: {output_dir}")
        return str(output_dir)

    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if Path(temp_file).exists():
                try:
                    os.remove(temp_file)
                    print(f"🗑️  Cleaned up: {temp_file}")
                except:
                    pass

    def validate_inputs(self, cvat_xml: str, images_dir: str):
        """Validate that input files exist"""
        if not os.path.exists(cvat_xml):
            raise FileNotFoundError(f"CVAT XML not found: {cvat_xml}")

        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        if not os.path.isdir(images_dir):
            raise NotADirectoryError(f"Images path is not a directory: {images_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run complete CVAT to training pipeline')
    parser.add_argument('--cvat', required=True, help='Path to CVAT XML annotations')
    parser.add_argument('--images', required=True, help='Directory containing images')
    parser.add_argument('--output', required=True, help='Output directory for training data')
    parser.add_argument('--sequence-length', type=int, default=30, help='Training sequence length')
    parser.add_argument('--stride', type=int, default=10, help='Sequence extraction stride')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary files')

    args = parser.parse_args()

    pipeline = CVATTrainingPipeline()

    try:
        # Validate inputs
        pipeline.validate_inputs(args.cvat, args.images)

        # Run pipeline
        pipeline.run_pipeline(
            args.cvat,
            args.images,
            args.output,
            args.sequence_length,
            args.stride
        )

        if not args.keep_temp:
            pipeline.cleanup()

        print(f"""
🎉 Complete Pipeline Success!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📂 Training data: {args.output}

Your data is now ready for TCN-VAE training!

Next steps:
1. cd {args.output}
2. python keypoint_dataloader.py  # Test the data loader
3. Integrate with your training scripts in ../training/
        """)

        return 0

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)