#!/usr/bin/env python3
"""
YOLOv8 to Training Pipeline Integration Script
Converts YOLOv8 dataset and trained models to TCN-VAE training format
This integrates YOLOv8 detection results into your behavioral analysis pipeline

Usage:
  python yolo8_to_training_integration.py --yolo-dataset /path/to/yolo --model /path/to/best.pt --output /path/to/training/data
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from datetime import datetime

try:
    import torch
    from ultralytics import YOLO
    import cv2
    from PIL import Image
except ImportError as e:
    print(f"❌ Required packages not installed: {e}")
    print("Install with:")
    print("  pip install ultralytics torch opencv-python Pillow")
    sys.exit(1)

class YOLOv8TrainingIntegrator:
    """Integrate YOLOv8 detection results into TCN-VAE training pipeline"""

    def __init__(self):
        self.stats = {
            'images_processed': 0,
            'detections_found': 0,
            'sequences_created': 0,
            'cropped_regions': 0,
            'detection_confidence': {'min': 1.0, 'max': 0.0, 'total': 0.0, 'count': 0}
        }

    def integrate_yolo_to_training(self, yolo_dataset_path: str, model_path: str,
                                 output_dir: str, confidence_threshold: float = 0.5,
                                 sequence_length: int = 30, stride: int = 10):
        """Convert YOLOv8 results to training format"""
        print(f"🎯 Integrating YOLOv8 detections into training pipeline...")
        print(f"Dataset: {yolo_dataset_path}")
        print(f"Model: {model_path}")
        print(f"Output: {output_dir}")
        print(f"Confidence threshold: {confidence_threshold}")

        # Load YOLOv8 model
        model = self.load_yolo_model(model_path)

        # Create output directories
        output_path = Path(output_dir)
        self.create_output_structure(output_path)

        # Process dataset splits
        dataset_path = Path(yolo_dataset_path)
        for split in ['train', 'val', 'test']:
            split_path = dataset_path / split
            if split_path.exists():
                print(f"📦 Processing {split} split...")
                self.process_split(model, split_path, output_path / split,
                                 confidence_threshold, sequence_length, stride)

        # Create detection sequences
        detection_sequences = self.create_detection_sequences(output_path, sequence_length, stride)

        # Save integrated training data
        self.save_integrated_training_data(detection_sequences, output_path)

        # Generate metadata
        self.save_integration_metadata(output_path, yolo_dataset_path, model_path)

        print(f"\n🎉 YOLOv8 Integration Complete!")
        self.print_integration_stats()

    def load_yolo_model(self, model_path: str) -> YOLO:
        """Load YOLOv8 model"""
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found: {model_path}")
            print("Using YOLOv8n pretrained model instead...")
            return YOLO('yolov8n.pt')

        print(f"📖 Loading YOLOv8 model: {model_path}")
        return YOLO(model_path)

    def create_output_structure(self, output_path: Path):
        """Create output directory structure"""
        directories = [
            'detections', 'cropped_regions', 'sequences',
            'metadata', 'train', 'val', 'test'
        ]

        for dir_name in directories:
            (output_path / dir_name).mkdir(parents=True, exist_ok=True)

        print(f"📁 Created output structure: {output_path}")

    def process_split(self, model: YOLO, split_path: Path, output_split_path: Path,
                     confidence_threshold: float, sequence_length: int, stride: int):
        """Process a dataset split (train/val/test)"""

        images_path = split_path / 'images'
        if not images_path.exists():
            print(f"⚠️ Images directory not found: {images_path}")
            return

        # Process images
        detection_results = []

        for image_path in sorted(images_path.glob('*')):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                results = self.process_image(model, image_path, confidence_threshold)
                if results:
                    detection_results.extend(results)
                    self.stats['images_processed'] += 1

        # Save detection results for this split
        if detection_results:
            detections_file = output_split_path / 'detections.json'
            with open(detections_file, 'w') as f:
                json.dump(detection_results, f, indent=2)

    def process_image(self, model: YOLO, image_path: Path, confidence_threshold: float) -> List[Dict]:
        """Process single image with YOLOv8"""
        try:
            # Run inference
            results = model(str(image_path), verbose=False)

            detection_results = []

            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf.cpu().numpy())

                        if confidence >= confidence_threshold:
                            # Get bounding box coordinates
                            xyxy = box.xyxy.cpu().numpy().flatten()
                            x1, y1, x2, y2 = xyxy

                            # Calculate center and dimensions
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1

                            # Get image dimensions
                            img_height, img_width = r.orig_shape

                            detection_data = {
                                'image_path': str(image_path),
                                'image_name': image_path.name,
                                'bbox': {
                                    'x1': float(x1), 'y1': float(y1),
                                    'x2': float(x2), 'y2': float(y2),
                                    'center_x': float(center_x), 'center_y': float(center_y),
                                    'width': float(width), 'height': float(height)
                                },
                                'confidence': confidence,
                                'class_id': int(box.cls.cpu().numpy()),
                                'image_dimensions': {'width': img_width, 'height': img_height}
                            }

                            detection_results.append(detection_data)
                            self.stats['detections_found'] += 1

                            # Update confidence stats
                            self.update_confidence_stats(confidence)

            return detection_results

        except Exception as e:
            print(f"⚠️ Error processing {image_path}: {e}")
            return []

    def create_detection_sequences(self, output_path: Path, sequence_length: int, stride: int) -> List[Dict]:
        """Create temporal sequences from detections"""
        sequences = []

        for split in ['train', 'val', 'test']:
            detections_file = output_path / split / 'detections.json'

            if detections_file.exists():
                with open(detections_file, 'r') as f:
                    detections = json.load(f)

                # Group by video/sequence (assuming sequential frames)
                video_groups = self.group_detections_by_video(detections)

                for video_name, video_detections in video_groups.items():
                    video_sequences = self.create_video_sequences(
                        video_detections, sequence_length, stride, split, video_name
                    )
                    sequences.extend(video_sequences)

        return sequences

    def group_detections_by_video(self, detections: List[Dict]) -> Dict[str, List[Dict]]:
        """Group detections by video/image sequence"""
        groups = {}

        for detection in detections:
            # Extract video/sequence identifier from filename
            image_name = detection['image_name']

            # Simple grouping by common prefix (you may need to adjust this)
            video_id = self.extract_video_id(image_name)

            if video_id not in groups:
                groups[video_id] = []
            groups[video_id].append(detection)

        # Sort each group by image name
        for video_id in groups:
            groups[video_id].sort(key=lambda x: x['image_name'])

        return groups

    def extract_video_id(self, image_name: str) -> str:
        """Extract video identifier from image name"""
        # Simple approach - use first part before any number
        # You may need to customize this based on your naming convention
        import re

        # Remove file extension
        name = Path(image_name).stem

        # Extract base name before frame numbers
        match = re.match(r'([a-zA-Z_]+)', name)
        if match:
            return match.group(1)

        return "default_video"

    def create_video_sequences(self, detections: List[Dict], sequence_length: int,
                              stride: int, split: str, video_name: str) -> List[Dict]:
        """Create sequences from video detections"""
        sequences = []

        if len(detections) < sequence_length:
            return sequences

        for start_idx in range(0, len(detections) - sequence_length + 1, stride):
            sequence_detections = detections[start_idx:start_idx + sequence_length]

            # Extract bounding box features for the sequence
            bbox_sequence = self.extract_bbox_features(sequence_detections)

            sequence_data = {
                'sequence_id': f"{video_name}_{split}_{start_idx}",
                'video_name': video_name,
                'split': split,
                'start_frame': start_idx,
                'end_frame': start_idx + sequence_length - 1,
                'bbox_features': bbox_sequence,
                'detections': sequence_detections
            }

            sequences.append(sequence_data)
            self.stats['sequences_created'] += 1

        return sequences

    def extract_bbox_features(self, sequence_detections: List[Dict]) -> np.ndarray:
        """Extract bounding box features for temporal analysis"""
        # Features: [center_x, center_y, width, height, confidence] per frame
        features = []

        for detection in sequence_detections:
            bbox = detection['bbox']
            img_dims = detection['image_dimensions']

            # Normalize by image dimensions
            normalized_features = [
                bbox['center_x'] / img_dims['width'],      # Normalized center_x
                bbox['center_y'] / img_dims['height'],     # Normalized center_y
                bbox['width'] / img_dims['width'],         # Normalized width
                bbox['height'] / img_dims['height'],       # Normalized height
                detection['confidence']                     # Detection confidence
            ]

            features.append(normalized_features)

        return np.array(features, dtype=np.float32)

    def save_integrated_training_data(self, sequences: List[Dict], output_path: Path):
        """Save sequences in training format"""
        if not sequences:
            print("⚠️ No sequences to save")
            return

        training_data = {
            'sequences': [],
            'metadata': {
                'n_sequences': len(sequences),
                'feature_dim': 5,  # [center_x, center_y, width, height, confidence]
                'feature_names': ['center_x_norm', 'center_y_norm', 'width_norm', 'height_norm', 'confidence'],
                'sequence_length': sequences[0]['bbox_features'].shape[0] if sequences else 0,
                'created_at': datetime.now().isoformat(),
                'data_type': 'yolo_detections'
            }
        }

        for seq in sequences:
            training_data['sequences'].append({
                'id': seq['sequence_id'],
                'video': seq['video_name'],
                'split': seq['split'],
                'bbox_features': seq['bbox_features'].tolist(),
                'start_frame': seq['start_frame'],
                'end_frame': seq['end_frame']
            })

        # Save formats
        json_path = output_path / "yolo_training_sequences.json"
        with open(json_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        pkl_path = output_path / "yolo_training_sequences.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(training_data, f)

        # Save individual arrays
        arrays_path = output_path / "sequences"
        for i, seq in enumerate(sequences):
            np.save(arrays_path / f"yolo_sequence_{i:04d}.npy", seq['bbox_features'])

        print(f"💾 Saved YOLOv8 training data:")
        print(f"  JSON: {json_path}")
        print(f"  Pickle: {pkl_path}")
        print(f"  Arrays: {arrays_path}/ ({len(sequences)} files)")

    def save_integration_metadata(self, output_path: Path, dataset_path: str, model_path: str):
        """Save integration metadata"""
        metadata = {
            'integration_info': {
                'source_dataset': dataset_path,
                'yolo_model': model_path,
                'feature_extraction': 'bounding_box_temporal',
                'integration_type': 'yolo_to_tcnvae'
            },
            'processing_stats': self.stats,
            'feature_definition': {
                'dimensions': 5,
                'features': [
                    'center_x_normalized',
                    'center_y_normalized',
                    'width_normalized',
                    'height_normalized',
                    'detection_confidence'
                ],
                'normalization': 'per_image_dimensions'
            }
        }

        metadata_file = output_path / "metadata" / "yolo_integration_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"📋 Saved integration metadata: {metadata_file}")

    def update_confidence_stats(self, confidence: float):
        """Update confidence statistics"""
        stats = self.stats['detection_confidence']
        stats['min'] = min(stats['min'], confidence)
        stats['max'] = max(stats['max'], confidence)
        stats['total'] += confidence
        stats['count'] += 1

    def print_integration_stats(self):
        """Print detailed integration statistics"""
        print(f"\n📊 Integration Statistics:")
        print(f"  Images processed: {self.stats['images_processed']}")
        print(f"  Detections found: {self.stats['detections_found']}")
        print(f"  Sequences created: {self.stats['sequences_created']}")

        conf_stats = self.stats['detection_confidence']
        if conf_stats['count'] > 0:
            avg_conf = conf_stats['total'] / conf_stats['count']
            print(f"  Detection confidence - Min: {conf_stats['min']:.3f}, Max: {conf_stats['max']:.3f}, Avg: {avg_conf:.3f}")

    def create_dataloader_template(self, output_path: Path):
        """Create YOLOv8 detection data loader template"""
        template_code = '''#!/usr/bin/env python3
"""
YOLOv8 Detection DataLoader for TCN-VAE Training
Load YOLOv8 detection sequences for behavioral analysis
"""

import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class YOLODetectionDataset(Dataset):
    """Dataset for YOLOv8 detection sequences"""

    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (str): Path to yolo_training_sequences.pkl
            transform (callable, optional): Optional transform
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.sequences = self.data['sequences']
        self.metadata = self.data['metadata']
        self.transform = transform

        print(f"Loaded {len(self.sequences)} YOLOv8 detection sequences")
        print(f"Sequence length: {self.metadata['sequence_length']}")
        print(f"Feature dimensions: {self.metadata['feature_dim']}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_data = self.sequences[idx]

        # Convert to tensor
        bbox_features = torch.tensor(sequence_data['bbox_features'], dtype=torch.float32)

        sample = {
            'bbox_features': bbox_features,  # [seq_len, 5] (center_x, center_y, width, height, conf)
            'sequence_id': sequence_data['id'],
            'video': sequence_data['video'],
            'split': sequence_data['split']
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

# Example usage:
if __name__ == "__main__":
    # Load dataset
    dataset = YOLODetectionDataset('yolo_training_sequences.pkl')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Test loading
    for batch in dataloader:
        print(f"Batch bbox_features shape: {batch['bbox_features'].shape}")
        break
'''

        template_path = output_path / "yolo_dataloader.py"
        with open(template_path, 'w') as f:
            f.write(template_code)

        print(f"📝 Created YOLOv8 data loader template: {template_path}")

def main():
    parser = argparse.ArgumentParser(description='Integrate YOLOv8 results into training pipeline')
    parser.add_argument('--yolo-dataset', required=True, help='Path to YOLOv8 dataset directory')
    parser.add_argument('--model', help='Path to trained YOLOv8 model (.pt file)')
    parser.add_argument('--output', required=True, help='Output directory for integrated training data')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--sequence-length', type=int, default=30, help='Sequence length for training')
    parser.add_argument('--stride', type=int, default=10, help='Sequence extraction stride')
    parser.add_argument('--create-loader', action='store_true', help='Create data loader template')

    args = parser.parse_args()

    if not os.path.exists(args.yolo_dataset):
        print(f"❌ YOLOv8 dataset not found: {args.yolo_dataset}")
        return 1

    # Use pretrained model if none specified
    model_path = args.model if args.model else 'yolov8n.pt'

    # Initialize integrator
    integrator = YOLOv8TrainingIntegrator()

    try:
        # Run integration
        integrator.integrate_yolo_to_training(
            args.yolo_dataset,
            model_path,
            args.output,
            args.confidence,
            args.sequence_length,
            args.stride
        )

        # Create data loader template if requested
        if args.create_loader:
            integrator.create_dataloader_template(Path(args.output))

        print(f"""
🎉 YOLOv8 Integration Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📂 Integrated data: {args.output}
📊 Sequences: {integrator.stats['sequences_created']}
🎯 Features per frame: {5} (bbox + confidence)

This data represents temporal bounding box sequences
suitable for behavioral analysis in your TCN-VAE pipeline.

Next steps:
1. Review data: python yolo_dataloader.py
2. Integrate with TCN-VAE training scripts
3. Use for movement pattern analysis
        """)

        return 0

    except Exception as e:
        print(f"❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)