#!/usr/bin/env python3
"""
Complete CVAT to YOLOv8 Training Pipeline Automation
Runs the full YOLOv8 pipeline:
1. CVAT individual points -> skeleton format (shared with SLEAP pipeline)
2. CVAT skeleton -> YOLOv8 dataset (bounding boxes)
3. YOLOv8 training & integration -> TCN-VAE training format

Usage:
  python run_yolo8_pipeline.py --cvat annotations.xml --images /path/to/images --output /path/to/training/data
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

class CVATYOLOv8Pipeline:
    """Orchestrates the complete YOLOv8 pipeline from CVAT to training"""

    def __init__(self):
        self.temp_files = []

    def run_yolo_pipeline(self, cvat_xml: str, images_dir: str, output_dir: str,
                         train_split: float = 0.8, val_split: float = 0.1,
                         confidence_threshold: float = 0.5, sequence_length: int = 30,
                         stride: int = 10, train_yolo: bool = False):
        """Run the complete YOLOv8 pipeline"""

        print("🎯 Starting CVAT → YOLOv8 → Training Pipeline")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Input: {cvat_xml}")
        print(f"🖼️  Images: {images_dir}")
        print(f"📂 Output: {output_dir}")
        print(f"🎯 Train YOLOv8: {train_yolo}")
        print()

        try:
            # Step 1: Individual points to skeleton (shared with SLEAP pipeline)
            skeleton_xml = self.step1_points_to_skeleton(cvat_xml)

            # Step 2: Skeleton to YOLOv8 dataset
            yolo_dataset_dir = self.step2_skeleton_to_yolo8(skeleton_xml, images_dir,
                                                           train_split, val_split)

            # Step 3: Train YOLOv8 (optional)
            model_path = None
            if train_yolo:
                model_path = self.step3_train_yolo8(yolo_dataset_dir)

            # Step 4: Integrate to training format
            training_data = self.step4_yolo8_to_training(yolo_dataset_dir, model_path,
                                                        output_dir, confidence_threshold,
                                                        sequence_length, stride)

            # Cleanup temporary files
            self.cleanup()

            print("🎉 YOLOv8 pipeline completed successfully!")
            print(f"📂 Training data available at: {output_dir}")

        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            self.cleanup()
            raise

    def step1_points_to_skeleton(self, cvat_xml: str) -> str:
        """Step 1: Convert individual points to skeleton format (shared step)"""
        print("📍 Step 1: Converting individual points to skeleton format...")

        # Create temporary skeleton XML file
        input_path = Path(cvat_xml)
        skeleton_xml = input_path.parent / f"yolo_skeleton_{input_path.stem}.xml"
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

    def step2_skeleton_to_yolo8(self, skeleton_xml: str, images_dir: str,
                               train_split: float, val_split: float) -> str:
        """Step 2: Convert skeleton XML to YOLOv8 dataset"""
        print("🎯 Step 2: Converting skeleton to YOLOv8 dataset...")

        # Create YOLOv8 dataset directory
        xml_path = Path(skeleton_xml)
        yolo_dataset_dir = xml_path.parent / f"yolo_dataset_{xml_path.stem}"

        # Run the YOLOv8 converter
        cmd = [
            sys.executable, "cvat_to_yolo8_converter.py",
            "--cvat", str(skeleton_xml),
            "--images", str(images_dir),
            "--output", str(yolo_dataset_dir),
            "--train-split", str(train_split),
            "--val-split", str(val_split),
            "--create-training-script"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Step 2 failed: {result.stderr}")

        print(f"✅ Step 2 complete: {yolo_dataset_dir}")
        return str(yolo_dataset_dir)

    def step3_train_yolo8(self, yolo_dataset_dir: str) -> str:
        """Step 3: Train YOLOv8 model (optional)"""
        print("🏋️ Step 3: Training YOLOv8 model...")

        dataset_path = Path(yolo_dataset_dir)

        # Check if dataset.yaml exists
        config_file = dataset_path / "dataset.yaml"
        if not config_file.exists():
            raise RuntimeError(f"Dataset config not found: {config_file}")

        # Run YOLOv8 training
        train_script = dataset_path / "train_yolo8.py"

        if train_script.exists():
            # Use the generated training script
            cmd = [sys.executable, str(train_script)]
            cwd = str(dataset_path)
        else:
            # Use ultralytics CLI
            cmd = [
                "yolo", "detect", "train",
                f"data={config_file}",
                "model=yolov8n.pt",
                "epochs=50",
                "imgsz=640",
                "batch=16",
                "name=dog_detection"
            ]
            cwd = str(dataset_path)

        print(f"Running training command: {' '.join(cmd)}")

        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"⚠️ Training failed, using pretrained model: {result.stderr}")
            return "yolov8n.pt"  # Use pretrained model

        # Find the best model
        runs_dir = dataset_path / "runs" / "detect"
        if runs_dir.exists():
            # Find the most recent training run
            latest_run = max(runs_dir.glob("*"), key=os.path.getctime, default=None)
            if latest_run:
                best_model = latest_run / "weights" / "best.pt"
                if best_model.exists():
                    print(f"✅ Step 3 complete - trained model: {best_model}")
                    return str(best_model)

        print("⚠️ Trained model not found, using pretrained model")
        return "yolov8n.pt"

    def step4_yolo8_to_training(self, yolo_dataset_dir: str, model_path: str,
                               output_dir: str, confidence_threshold: float,
                               sequence_length: int, stride: int) -> str:
        """Step 4: Integrate YOLOv8 to training format"""
        print("🔄 Step 4: Integrating YOLOv8 to training format...")

        cmd = [
            sys.executable, "yolo8_to_training_integration.py",
            "--yolo-dataset", str(yolo_dataset_dir),
            "--output", str(output_dir),
            "--confidence", str(confidence_threshold),
            "--sequence-length", str(sequence_length),
            "--stride", str(stride),
            "--create-loader"
        ]

        if model_path:
            cmd.extend(["--model", str(model_path)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Step 4 failed: {result.stderr}")

        print(f"✅ Step 4 complete: {output_dir}")
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
    parser = argparse.ArgumentParser(description='Run complete CVAT to YOLOv8 training pipeline')
    parser.add_argument('--cvat', required=True, help='Path to CVAT XML annotations')
    parser.add_argument('--images', required=True, help='Directory containing images')
    parser.add_argument('--output', required=True, help='Output directory for training data')

    # YOLOv8 dataset parameters
    parser.add_argument('--train-split', type=float, default=0.8, help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')

    # Training integration parameters
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--sequence-length', type=int, default=30, help='Training sequence length')
    parser.add_argument('--stride', type=int, default=10, help='Sequence extraction stride')

    # Pipeline options
    parser.add_argument('--train-yolo', action='store_true', help='Train YOLOv8 model (slow)')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary files')

    args = parser.parse_args()

    pipeline = CVATYOLOv8Pipeline()

    try:
        # Validate inputs
        pipeline.validate_inputs(args.cvat, args.images)

        # Run pipeline
        pipeline.run_yolo_pipeline(
            args.cvat,
            args.images,
            args.output,
            args.train_split,
            args.val_split,
            args.confidence,
            args.sequence_length,
            args.stride,
            args.train_yolo
        )

        if not args.keep_temp:
            pipeline.cleanup()

        print(f"""
🎉 Complete YOLOv8 Pipeline Success!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📂 Training data: {args.output}

Your YOLOv8 detection sequences are ready for behavioral analysis!

Data format: [sequence_length, 5] features per frame
Features: [center_x, center_y, width, height, confidence] (normalized)

Next steps:
1. cd {args.output}
2. python yolo_dataloader.py  # Test the data loader
3. Integrate with TCN-VAE for movement analysis
4. Use for behavioral pattern detection
        """)

        return 0

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)