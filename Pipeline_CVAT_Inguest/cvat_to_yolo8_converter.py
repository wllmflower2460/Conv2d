#!/usr/bin/env python3
"""
CVAT to YOLOv8 Dataset Converter
Converts CVAT XML annotations to YOLOv8 format for object detection training
Extracts dog_bbox bounding boxes and converts to YOLO format

Usage:
  python cvat_to_yolo8_converter.py --cvat annotations.xml --images /path/to/images --output /path/to/yolo_dataset
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from datetime import datetime

try:
    from PIL import Image
except ImportError:
    print("❌ PIL not installed. Install with: pip install Pillow")
    sys.exit(1)

class CVATToYOLO8Converter:
    """Convert CVAT bounding box annotations to YOLOv8 format"""

    def __init__(self):
        # Class definitions - can be extended for multi-class detection
        self.classes = {
            'dog_bbox': 0  # YOLOv8 class index for dogs
        }

        self.class_names = ['dog']  # Human readable class names

        self.stats = {
            'images_processed': 0,
            'bboxes_extracted': 0,
            'images_copied': 0,
            'bbox_attributes': {}
        }

    def convert_cvat_to_yolo8(self, cvat_file: str, images_dir: str, output_dir: str,
                             train_split: float = 0.8, val_split: float = 0.1):
        """Convert CVAT annotations to YOLOv8 dataset structure"""
        print(f"📖 Converting CVAT annotations to YOLOv8 format...")
        print(f"Input: {cvat_file}")
        print(f"Images: {images_dir}")
        print(f"Output: {output_dir}")
        print(f"Train/Val/Test split: {train_split:.1%}/{val_split:.1%}/{1-train_split-val_split:.1%}")

        # Parse CVAT XML
        annotations = self.parse_cvat_xml(cvat_file)

        if not annotations:
            print("❌ No valid annotations found in CVAT file")
            return

        # Create YOLOv8 dataset structure
        self.create_yolo_structure(output_dir)

        # Process annotations and split into train/val/test
        self.process_and_split_data(annotations, images_dir, output_dir,
                                   train_split, val_split)

        # Create dataset configuration
        self.create_dataset_yaml(output_dir)

        # Print statistics
        self.print_conversion_stats()

        print(f"✅ YOLOv8 dataset created: {output_dir}")

    def parse_cvat_xml(self, cvat_file: str) -> List[Dict]:
        """Parse CVAT XML and extract bounding box annotations"""
        print(f"📖 Parsing CVAT annotations: {cvat_file}")

        tree = ET.parse(cvat_file)
        root = tree.getroot()

        annotations = []

        for image_elem in root.findall('.//image'):
            image_name = image_elem.get('name')
            image_width = int(image_elem.get('width', 640))
            image_height = int(image_elem.get('height', 480))

            bboxes = []

            # Find all bounding boxes in this image
            for box_elem in image_elem.findall('.//box'):
                label = box_elem.get('label', '')

                if label == 'dog_bbox':
                    bbox_data = self.parse_bbox_element(box_elem, image_width, image_height)
                    if bbox_data:
                        bboxes.append(bbox_data)
                        self.stats['bboxes_extracted'] += 1

            if bboxes:
                annotations.append({
                    'image_name': image_name,
                    'image_width': image_width,
                    'image_height': image_height,
                    'bboxes': bboxes
                })
                self.stats['images_processed'] += 1

        print(f"📊 Found {len(annotations)} images with {self.stats['bboxes_extracted']} bounding boxes")
        return annotations

    def parse_bbox_element(self, box_elem, img_width: int, img_height: int) -> Optional[Dict]:
        """Parse CVAT bounding box element and convert to YOLO format"""
        try:
            # Get bounding box coordinates
            xtl = float(box_elem.get('xtl', 0))
            ytl = float(box_elem.get('ytl', 0))
            xbr = float(box_elem.get('xbr', 0))
            ybr = float(box_elem.get('ybr', 0))

            # Convert CVAT format (xtl, ytl, xbr, ybr) to YOLO format (x_center, y_center, width, height)
            # All values normalized to [0, 1]
            bbox_width = xbr - xtl
            bbox_height = ybr - ytl
            x_center = xtl + bbox_width / 2
            y_center = ytl + bbox_height / 2

            # Normalize coordinates
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = bbox_width / img_width
            height_norm = bbox_height / img_height

            # Extract attributes for metadata
            attributes = {}
            for attr_elem in box_elem.findall('.//attribute'):
                attr_name = attr_elem.get('name', '')
                attr_value = attr_elem.text or ''
                attributes[attr_name] = attr_value

                # Track attribute statistics
                if attr_name not in self.stats['bbox_attributes']:
                    self.stats['bbox_attributes'][attr_name] = {}
                if attr_value not in self.stats['bbox_attributes'][attr_name]:
                    self.stats['bbox_attributes'][attr_name][attr_value] = 0
                self.stats['bbox_attributes'][attr_name][attr_value] += 1

            return {
                'class_id': self.classes['dog_bbox'],
                'x_center': x_center_norm,
                'y_center': y_center_norm,
                'width': width_norm,
                'height': height_norm,
                'attributes': attributes,
                'confidence': 1.0
            }

        except (ValueError, TypeError) as e:
            print(f"⚠️ Error parsing bbox: {e}")
            return None

    def create_yolo_structure(self, output_dir: str):
        """Create YOLOv8 dataset directory structure"""
        base_path = Path(output_dir)

        # Create main directories
        for split in ['train', 'val', 'test']:
            (base_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (base_path / split / 'labels').mkdir(parents=True, exist_ok=True)

        print(f"📁 Created YOLOv8 directory structure in: {output_dir}")

    def process_and_split_data(self, annotations: List[Dict], images_dir: str,
                              output_dir: str, train_split: float, val_split: float):
        """Process annotations and split into train/val/test sets"""

        base_path = Path(output_dir)
        images_path = Path(images_dir)

        # Calculate split indices
        total_images = len(annotations)
        train_end = int(total_images * train_split)
        val_end = train_end + int(total_images * val_split)

        # Shuffle for random split (optional - comment out for deterministic)
        import random
        random.seed(42)  # For reproducible splits
        random.shuffle(annotations)

        split_info = {
            'train': annotations[:train_end],
            'val': annotations[train_end:val_end],
            'test': annotations[val_end:]
        }

        for split_name, split_data in split_info.items():
            print(f"📦 Processing {split_name} split: {len(split_data)} images")

            for ann in split_data:
                image_name = ann['image_name']
                image_path = images_path / image_name

                if not image_path.exists():
                    print(f"⚠️ Image not found: {image_path}")
                    continue

                # Copy image
                dest_image_path = base_path / split_name / 'images' / image_name
                shutil.copy2(image_path, dest_image_path)
                self.stats['images_copied'] += 1

                # Create label file
                label_name = Path(image_name).stem + '.txt'
                label_path = base_path / split_name / 'labels' / label_name

                self.write_yolo_label(label_path, ann['bboxes'])

    def write_yolo_label(self, label_path: Path, bboxes: List[Dict]):
        """Write YOLO format label file"""
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                # YOLO format: class_id x_center y_center width height
                line = f"{bbox['class_id']} {bbox['x_center']:.6f} {bbox['y_center']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}\n"
                f.write(line)

    def create_dataset_yaml(self, output_dir: str):
        """Create YOLOv8 dataset configuration file"""
        config = {
            'path': str(Path(output_dir).absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_names),  # number of classes
            'names': self.class_names
        }

        yaml_path = Path(output_dir) / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            # Write YAML format
            f.write(f"# YOLOv8 Dataset Configuration\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"path: {config['path']}\n")
            f.write(f"train: {config['train']}\n")
            f.write(f"val: {config['val']}\n")
            f.write(f"test: {config['test']}\n\n")
            f.write(f"nc: {config['nc']}\n")
            f.write(f"names: {config['names']}\n")

        print(f"📝 Created dataset config: {yaml_path}")

    def create_training_script_template(self, output_dir: str):
        """Create YOLOv8 training script template"""
        template_code = '''#!/usr/bin/env python3
"""
YOLOv8 Training Script Template
Train YOLOv8 model on converted dataset
"""

from ultralytics import YOLO
import torch

def train_yolo8_model():
    """Train YOLOv8 model on the dataset"""

    # Initialize model
    model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

    # Train the model
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='dog_detection',
        patience=50,
        save=True,
        plots=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Validate the model
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

    # Export to different formats
    model.export(format='onnx')  # ONNX format
    model.export(format='engine')  # TensorRT (if available)

    return model, results

if __name__ == "__main__":
    model, results = train_yolo8_model()
    print("Training completed!")
'''

        script_path = Path(output_dir) / 'train_yolo8.py'
        with open(script_path, 'w') as f:
            f.write(template_code)

        print(f"📝 Created training script: {script_path}")

    def print_conversion_stats(self):
        """Print detailed conversion statistics"""
        print(f"\n📊 Conversion Statistics:")
        print(f"  Images processed: {self.stats['images_processed']}")
        print(f"  Images copied: {self.stats['images_copied']}")
        print(f"  Bounding boxes: {self.stats['bboxes_extracted']}")

        if self.stats['bbox_attributes']:
            print(f"\n📋 Bounding Box Attributes:")
            for attr_name, attr_values in self.stats['bbox_attributes'].items():
                print(f"  {attr_name}:")
                for value, count in sorted(attr_values.items()):
                    print(f"    - {value}: {count}")

def main():
    parser = argparse.ArgumentParser(description='Convert CVAT annotations to YOLOv8 format')
    parser.add_argument('--cvat', required=True, help='Path to CVAT XML annotations')
    parser.add_argument('--images', required=True, help='Directory containing images')
    parser.add_argument('--output', required=True, help='Output directory for YOLOv8 dataset')
    parser.add_argument('--train-split', type=float, default=0.8, help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--create-training-script', action='store_true', help='Create training script template')

    args = parser.parse_args()

    if not os.path.exists(args.cvat):
        print(f"❌ CVAT file not found: {args.cvat}")
        return 1

    if not os.path.exists(args.images):
        print(f"❌ Images directory not found: {args.images}")
        return 1

    # Validate split ratios
    if args.train_split + args.val_split >= 1.0:
        print(f"❌ Train + validation split must be < 1.0")
        return 1

    # Initialize converter
    converter = CVATToYOLO8Converter()

    try:
        # Convert to YOLOv8
        converter.convert_cvat_to_yolo8(
            args.cvat,
            args.images,
            args.output,
            args.train_split,
            args.val_split
        )

        # Create training script if requested
        if args.create_training_script:
            converter.create_training_script_template(args.output)

        print(f"""
🎉 YOLOv8 Conversion Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📂 Dataset: {args.output}
📊 Images: {converter.stats['images_processed']}
🎯 Bounding boxes: {converter.stats['bboxes_extracted']}

Next steps:
1. cd {args.output}
2. pip install ultralytics
3. python train_yolo8.py  # Start training
4. Monitor training: tensorboard --logdir runs/
        """)

        return 0

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)