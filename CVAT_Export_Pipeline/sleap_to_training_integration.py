#!/usr/bin/env python3
"""
SLEAP to Training Pipeline Integration Script
Converts SLEAP dataset (.slp) to TCN-VAE training format
This is script #3 in the CVAT -> SLEAP -> Training pipeline

Usage:
  python sleap_to_training_integration.py --sleap dataset.slp --output /path/to/training/data
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
    import sleap
    import torch
    from sleap import Labels
except ImportError as e:
    print(f"❌ Required packages not installed: {e}")
    print("Install with: conda install sleap -c sleap -c nvidia -c conda-forge")
    print("And: pip install torch")
    sys.exit(1)

class SLEAPToTrainingConverter:
    """Convert SLEAP dataset to TCN-VAE training format"""

    def __init__(self):
        self.node_names = [
            # Head (5 points)
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            # Neck/Torso (3 points)
            "throat", "withers", "center",
            # Front legs (6 points)
            "left_front_shoulder", "left_front_elbow", "left_front_paw",
            "right_front_shoulder", "right_front_elbow", "right_front_paw",
            # Back legs (6 points)
            "left_hip", "left_knee", "left_back_paw",
            "right_hip", "right_knee", "right_back_paw",
            # Tail (4 points)
            "tail_base", "tail_mid_1", "tail_mid_2", "tail_tip"
        ]

        self.stats = {
            'frames_processed': 0,
            'sequences_created': 0,
            'valid_keypoints': 0,
            'missing_keypoints': 0,
            'confidence_stats': {'min': 1.0, 'max': 0.0, 'mean': 0.0}
        }

    def convert_sleap_to_training(self, sleap_file: str, output_dir: str,
                                sequence_length: int = 30, stride: int = 10):
        """Convert SLEAP dataset to training format"""
        print(f"📖 Loading SLEAP dataset: {sleap_file}")

        # Load SLEAP dataset
        labels = Labels.load_file(sleap_file)

        if not labels.labeled_frames:
            print("❌ No labeled frames found in SLEAP file")
            return

        print(f"✅ Loaded {len(labels.labeled_frames)} labeled frames")

        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        sequences_path = output_path / "sequences"
        sequences_path.mkdir(exist_ok=True)

        metadata_path = output_path / "metadata"
        metadata_path.mkdir(exist_ok=True)

        # Convert frames to keypoint sequences
        sequences = self.extract_keypoint_sequences(labels, sequence_length, stride)

        if not sequences:
            print("❌ No valid sequences could be extracted")
            return

        # Save training data
        self.save_training_data(sequences, output_path)

        # Generate metadata
        self.save_metadata(labels, output_path, sequence_length, stride)

        print(f"\n🎉 Conversion Complete!")
        self.print_conversion_stats()

    def extract_keypoint_sequences(self, labels: Labels, seq_length: int, stride: int) -> List[Dict]:
        """Extract keypoint sequences from SLEAP labels"""
        sequences = []

        # Group frames by video
        video_frames = {}
        for lf in labels.labeled_frames:
            video_name = lf.video.filename if lf.video else "default"
            if video_name not in video_frames:
                video_frames[video_name] = []
            video_frames[video_name].append(lf)

        for video_name, frames in video_frames.items():
            # Sort frames by frame index
            frames.sort(key=lambda x: x.frame_idx)

            print(f"📹 Processing video: {video_name} ({len(frames)} frames)")

            # Extract sequences with sliding window
            for start_idx in range(0, len(frames) - seq_length + 1, stride):
                sequence_frames = frames[start_idx:start_idx + seq_length]

                # Convert to keypoint array
                keypoint_sequence = self.frames_to_keypoints(sequence_frames)

                if keypoint_sequence is not None:
                    sequences.append({
                        'video': video_name,
                        'start_frame': sequence_frames[0].frame_idx,
                        'end_frame': sequence_frames[-1].frame_idx,
                        'keypoints': keypoint_sequence,
                        'sequence_id': f"{video_name}_{start_idx}"
                    })
                    self.stats['sequences_created'] += 1

        return sequences

    def frames_to_keypoints(self, frames: List) -> Optional[np.ndarray]:
        """Convert sequence of frames to keypoint array"""
        seq_length = len(frames)
        n_keypoints = len(self.node_names)

        # Initialize keypoint array: [sequence_length, n_keypoints, 3] (x, y, confidence)
        keypoints = np.zeros((seq_length, n_keypoints, 3))

        valid_frame_count = 0

        for frame_idx, labeled_frame in enumerate(frames):
            self.stats['frames_processed'] += 1

            if not labeled_frame.instances:
                continue

            # Use first instance (assuming single animal per frame)
            instance = labeled_frame.instances[0]
            frame_valid_points = 0

            for node_idx, node_name in enumerate(self.node_names):
                # Find corresponding node in skeleton
                skeleton_node = None
                for node in instance.skeleton.nodes:
                    if node.name == node_name:
                        skeleton_node = node
                        break

                if skeleton_node and skeleton_node in instance.points:
                    point = instance.points[skeleton_node]

                    if not np.isnan(point.x) and not np.isnan(point.y):
                        keypoints[frame_idx, node_idx, 0] = point.x
                        keypoints[frame_idx, node_idx, 1] = point.y
                        keypoints[frame_idx, node_idx, 2] = 1.0 if point.visible else 0.5

                        self.stats['valid_keypoints'] += 1
                        frame_valid_points += 1
                    else:
                        # Missing/invalid point
                        keypoints[frame_idx, node_idx, 2] = 0.0
                        self.stats['missing_keypoints'] += 1
                else:
                    # Node not found
                    keypoints[frame_idx, node_idx, 2] = 0.0
                    self.stats['missing_keypoints'] += 1

            if frame_valid_points > len(self.node_names) * 0.5:  # At least 50% points valid
                valid_frame_count += 1

        # Only return sequence if most frames have sufficient keypoints
        if valid_frame_count >= seq_length * 0.7:  # At least 70% of frames are valid
            return keypoints
        else:
            return None

    def save_training_data(self, sequences: List[Dict], output_path: Path):
        """Save sequences in training format"""

        # Convert to training format
        training_data = {
            'sequences': [],
            'metadata': {
                'n_sequences': len(sequences),
                'n_keypoints': len(self.node_names),
                'keypoint_names': self.node_names,
                'sequence_length': sequences[0]['keypoints'].shape[0] if sequences else 0,
                'created_at': datetime.now().isoformat()
            }
        }

        for seq in sequences:
            training_data['sequences'].append({
                'id': seq['sequence_id'],
                'video': seq['video'],
                'keypoints': seq['keypoints'].tolist(),  # Convert numpy to list for JSON
                'start_frame': seq['start_frame'],
                'end_frame': seq['end_frame']
            })

        # Save as JSON for easy loading
        json_path = output_path / "training_sequences.json"
        with open(json_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        # Save as pickle for faster loading in Python
        pkl_path = output_path / "training_sequences.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(training_data, f)

        # Save individual numpy arrays for direct use
        arrays_path = output_path / "sequences"
        for i, seq in enumerate(sequences):
            np.save(arrays_path / f"sequence_{i:04d}.npy", seq['keypoints'])

        print(f"💾 Saved training data:")
        print(f"  JSON: {json_path}")
        print(f"  Pickle: {pkl_path}")
        print(f"  Arrays: {arrays_path}/ ({len(sequences)} files)")

    def save_metadata(self, labels: Labels, output_path: Path, seq_length: int, stride: int):
        """Save dataset metadata"""
        metadata = {
            'dataset_info': {
                'sleap_version': sleap.__version__,
                'n_labeled_frames': len(labels.labeled_frames),
                'n_videos': len(set(lf.video.filename for lf in labels.labeled_frames if lf.video)),
                'sequence_length': seq_length,
                'stride': stride,
                'keypoint_names': self.node_names,
                'n_keypoints': len(self.node_names)
            },
            'conversion_stats': self.stats,
            'skeleton_definition': {
                'nodes': self.node_names,
                'connections': [
                    # Head
                    ('nose', 'left_eye'), ('nose', 'right_eye'),
                    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
                    ('nose', 'throat'),
                    # Torso
                    ('throat', 'withers'), ('withers', 'center'), ('center', 'tail_base'),
                    # Front legs
                    ('withers', 'left_front_shoulder'), ('left_front_shoulder', 'left_front_elbow'),
                    ('left_front_elbow', 'left_front_paw'),
                    ('withers', 'right_front_shoulder'), ('right_front_shoulder', 'right_front_elbow'),
                    ('right_front_elbow', 'right_front_paw'),
                    # Back legs
                    ('center', 'left_hip'), ('left_hip', 'left_knee'), ('left_knee', 'left_back_paw'),
                    ('center', 'right_hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_back_paw'),
                    # Tail
                    ('tail_base', 'tail_mid_1'), ('tail_mid_1', 'tail_mid_2'), ('tail_mid_2', 'tail_tip')
                ]
            }
        }

        metadata_file = output_path / "metadata" / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"📋 Saved metadata: {metadata_file}")

    def print_conversion_stats(self):
        """Print detailed conversion statistics"""
        print(f"\n📊 Conversion Statistics:")
        print(f"  Frames processed: {self.stats['frames_processed']}")
        print(f"  Sequences created: {self.stats['sequences_created']}")
        print(f"  Valid keypoints: {self.stats['valid_keypoints']}")
        print(f"  Missing keypoints: {self.stats['missing_keypoints']}")

        if self.stats['valid_keypoints'] + self.stats['missing_keypoints'] > 0:
            valid_ratio = self.stats['valid_keypoints'] / (self.stats['valid_keypoints'] + self.stats['missing_keypoints'])
            print(f"  Keypoint validity: {valid_ratio:.2%}")

    def create_data_loader_template(self, output_path: Path):
        """Create a template data loader for training"""
        template_code = '''#!/usr/bin/env python3
"""
Data Loader Template for TCN-VAE Training
Load the converted SLEAP data for training
"""

import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class KeypointSequenceDataset(Dataset):
    """Dataset for keypoint sequences"""

    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (str): Path to training_sequences.pkl
            transform (callable, optional): Optional transform to be applied on a sample
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.sequences = self.data['sequences']
        self.metadata = self.data['metadata']
        self.transform = transform

        print(f"Loaded {len(self.sequences)} sequences")
        print(f"Sequence length: {self.metadata['sequence_length']}")
        print(f"Keypoints: {self.metadata['n_keypoints']}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_data = self.sequences[idx]
        keypoints = np.array(sequence_data['keypoints'], dtype=np.float32)

        # Extract coordinates and confidence
        coords = keypoints[:, :, :2]  # [seq_len, n_keypoints, 2]
        confidence = keypoints[:, :, 2]  # [seq_len, n_keypoints]

        sample = {
            'keypoints': coords,
            'confidence': confidence,
            'sequence_id': sequence_data['id'],
            'video': sequence_data['video']
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

# Example usage:
if __name__ == "__main__":
    # Load dataset
    dataset = KeypointSequenceDataset('training_sequences.pkl')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Test loading
    for batch in dataloader:
        print(f"Batch keypoints shape: {batch['keypoints'].shape}")
        print(f"Batch confidence shape: {batch['confidence'].shape}")
        break
'''

        template_path = output_path / "keypoint_dataloader.py"
        with open(template_path, 'w') as f:
            f.write(template_code)

        print(f"📝 Created data loader template: {template_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert SLEAP dataset to training format')
    parser.add_argument('--sleap', required=True, help='Path to SLEAP dataset file (.slp)')
    parser.add_argument('--output', required=True, help='Output directory for training data')
    parser.add_argument('--sequence-length', type=int, default=30, help='Length of training sequences')
    parser.add_argument('--stride', type=int, default=10, help='Stride for sequence extraction')
    parser.add_argument('--create-loader', action='store_true', help='Create data loader template')

    args = parser.parse_args()

    if not os.path.exists(args.sleap):
        print(f"❌ SLEAP file not found: {args.sleap}")
        return 1

    # Initialize converter
    converter = SLEAPToTrainingConverter()

    try:
        # Convert SLEAP to training format
        converter.convert_sleap_to_training(
            args.sleap,
            args.output,
            args.sequence_length,
            args.stride
        )

        # Create data loader template if requested
        if args.create_loader:
            converter.create_data_loader_template(Path(args.output))

        print(f"""
🎉 Integration Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📂 Training data saved to: {args.output}
📊 Sequences: {converter.stats['sequences_created']}
🦴 Keypoints per sequence: {len(converter.node_names)}

Next steps:
1. Review the data: python keypoint_dataloader.py
2. Integrate with your TCN-VAE training:
   from keypoint_dataloader import KeypointSequenceDataset
3. Start training with your existing training scripts
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