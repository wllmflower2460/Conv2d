#!/usr/bin/env python3
"""
CVAT to SLEAP Dataset Converter
Converts CVAT XML annotations to SLEAP format for custom animal pose model training
Supports both skeleton and individual point annotations

Usage:
  python cvat_to_sleap_converter.py --cvat annotations.xml --images /path/to/images --output dataset.slp
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from datetime import datetime

try:
    import sleap
    import cv2
    from sleap import Skeleton, Node, Edge, Instance, LabeledFrame, Labels
    from sleap.io.dataset import load_video
except ImportError:
    print("❌ SLEAP not installed. Install with:")
    print("conda install sleap -c sleap -c nvidia -c conda-forge")
    sys.exit(1)

class CVATToSLEAPConverter:
    """Convert CVAT pose annotations to SLEAP format for training"""
    
    def __init__(self):
        self.skeleton = None
        # 24-point anatomically correct system
        self.node_names = [
            # Head (5 points)
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            # Neck/Torso (3 points) 
            "throat", "withers", "center",
            # Front legs (6 points)
            "left_front_shoulder", "left_front_elbow", "left_front_paw",
            "right_front_shoulder", "right_front_elbow", "right_front_paw",
            # Back legs (6 points) - ANATOMICALLY CORRECT
            "left_hip", "left_knee", "left_back_paw",
            "right_hip", "right_knee", "right_back_paw",
            # Tail (4 points)
            "tail_base", "tail_mid_1", "tail_mid_2", "tail_tip"
        ]
        
        # Define anatomical connections for skeleton
        self.connections = [
            # Head
            ("nose", "left_eye"), ("nose", "right_eye"), 
            ("left_eye", "left_ear"), ("right_eye", "right_ear"),
            ("nose", "throat"),
            # Torso
            ("throat", "withers"), ("withers", "center"), ("center", "tail_base"),
            # Front legs
            ("withers", "left_front_shoulder"), ("left_front_shoulder", "left_front_elbow"),
            ("left_front_elbow", "left_front_paw"),
            ("withers", "right_front_shoulder"), ("right_front_shoulder", "right_front_elbow"),
            ("right_front_elbow", "right_front_paw"),
            # Back legs
            ("center", "left_hip"), ("left_hip", "left_knee"),
            ("left_knee", "left_back_paw"),
            ("center", "right_hip"), ("right_hip", "right_knee"),
            ("right_knee", "right_back_paw"),
            # Tail
            ("tail_base", "tail_mid_1"), ("tail_mid_1", "tail_mid_2"), 
            ("tail_mid_2", "tail_tip")
        ]
        
        self.create_skeleton()
    
    def create_skeleton(self) -> None:
        """Create SLEAP skeleton from DataDogs joint definitions"""
        print("🦴 Creating DataDogs 24-point animal skeleton for SLEAP...")
        
        # Create nodes
        nodes = []
        for name in self.node_names:
            nodes.append(Node(name))
        
        # Create edges based on anatomical connections
        edges = []
        node_dict = {node.name: node for node in nodes}
        
        for src_name, dst_name in self.connections:
            if src_name in node_dict and dst_name in node_dict:
                edges.append(Edge(node_dict[src_name], node_dict[dst_name]))
        
        self.skeleton = Skeleton(nodes=nodes, edges=edges)
        
        print(f"✅ Skeleton created with {len(nodes)} nodes and {len(edges)} edges")
    
    def parse_cvat_xml(self, cvat_file: str) -> List[Dict]:
        """Parse CVAT XML annotations file - handles both skeleton and individual points"""
        print(f"📖 Parsing CVAT annotations: {cvat_file}")
        
        tree = ET.parse(cvat_file)
        root = tree.getroot()
        
        annotations = []
        
        for image_elem in root.findall('.//image'):
            image_name = image_elem.get('name')
            image_width = int(image_elem.get('width', 640))
            image_height = int(image_elem.get('height', 480))
            
            skeletons = []
            
            # First try to find skeleton elements
            skeleton_found = False
            for skeleton_elem in image_elem.findall('.//skeleton'):
                skeleton_data = self.parse_skeleton_element(skeleton_elem)
                if skeleton_data:
                    skeletons.append(skeleton_data)
                    skeleton_found = True
            
            # If no skeletons, look for individual point annotations
            if not skeleton_found:
                individual_points = self.parse_individual_point_annotations(image_elem)
                if individual_points:
                    skeletons.append(individual_points)
            
            if skeletons:
                annotations.append({
                    'image_name': image_name,
                    'image_width': image_width,
                    'image_height': image_height,
                    'skeletons': skeletons
                })
        
        print(f"📊 Parsed {len(annotations)} annotated images")
        return annotations
    
    def parse_skeleton_element(self, skeleton_elem) -> Optional[Dict]:
        """Parse CVAT skeleton element"""
        keypoints = {}
        
        # Get skeleton attributes
        attributes = {}
        for attr in skeleton_elem.findall('.//attribute'):
            attr_name = attr.get('name', '')
            attr_value = attr.text or ''
            attributes[attr_name] = attr_value
        
        # Parse points within skeleton
        for point_elem in skeleton_elem.findall('.//points'):
            label = point_elem.get('label', '')
            points_str = point_elem.get('points', '')
            
            # Parse points string: "x1,y1;x2,y2"
            if ',' in points_str:
                coords = points_str.split(';')[0].split(',')  # Take first point
                if len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    
                    # Map CVAT label to our node names
                    node_name = self.map_cvat_label_to_node(label)
                    if node_name:
                        keypoints[node_name] = {
                            'x': x, 'y': y, 
                            'visibility': self.get_visibility(point_elem),
                            'confidence': 1.0
                        }
        
        if keypoints:
            return {
                'keypoints': keypoints,
                'attributes': attributes
            }
        return None
    
    def parse_individual_point_annotations(self, image_elem) -> Optional[Dict]:
        """Parse individual point annotations (not part of a skeleton)"""
        keypoints = {}
        
        # Look for all point annotations in the image
        for points_elem in image_elem.findall('.//points'):
            # Skip if this is part of a skeleton
            parent = points_elem.getparent()
            if parent is not None and parent.tag == 'skeleton':
                continue
            
            label = points_elem.get('label', '')
            points_str = points_elem.get('points', '')
            
            if ',' in points_str:
                coords = points_str.split(',')
                if len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    
                    # Map the label to our node names
                    node_name = self.map_cvat_label_to_node(label)
                    if node_name:
                        keypoints[node_name] = {
                            'x': x, 
                            'y': y,
                            'visibility': self.get_visibility(points_elem),
                            'confidence': 1.0
                        }
        
        if keypoints:
            print(f"   Found {len(keypoints)} individual keypoints")
            return {'keypoints': keypoints, 'attributes': {}}
        return None
    
    def map_cvat_label_to_node(self, cvat_label: str) -> Optional[str]:
        """Map CVAT label names to SLEAP node names"""
        
        # Direct mapping for exact matches
        if cvat_label in self.node_names:
            return cvat_label
        
        # Handle variations and legacy naming
        label_mapping = {
            # Legacy naming from old configs (if you had old annotations)
            'neck': 'throat',  # If old annotations used 'neck'
            'spine_mid': 'center',  # If old annotations used 'spine_mid'
            
            # Handle 26-point to 24-point mapping
            'left_front_wrist': 'left_front_elbow',  # Skip wrist, use elbow
            'right_front_wrist': 'right_front_elbow',
            'left_ankle': 'left_knee',  # Skip ankle, use knee
            'right_ankle': 'right_knee',
            'tail_mid': 'tail_mid_1',  # Single mid to first mid
            
            # Alternative naming conventions
            'left_hind_paw': 'left_back_paw',
            'right_hind_paw': 'right_back_paw',
            'left_rear_paw': 'left_back_paw',
            'right_rear_paw': 'right_back_paw',
            
            # Handle incorrect anatomical naming from old systems
            'left_back_shoulder': 'left_hip',
            'right_back_shoulder': 'right_hip',
            'left_back_elbow': 'left_knee',
            'right_back_elbow': 'right_knee',
            'left_back_wrist': 'left_knee',  # Map to knee if no ankle
            'right_back_wrist': 'right_knee',
            
            # Common variations
            'spine_middle': 'center',
            'tail_end': 'tail_tip',
            'tail_middle': 'tail_mid_1',
            
            # Veterinary terms
            'left_stifle': 'left_knee',
            'right_stifle': 'right_knee',
            'left_hock': 'left_knee',
            'right_hock': 'right_knee'
        }
        
        mapped = label_mapping.get(cvat_label.lower())
        if not mapped:
            # Try to find partial matches
            cvat_lower = cvat_label.lower()
            for node in self.node_names:
                if node in cvat_lower or cvat_lower in node:
                    print(f"📝 Fuzzy match: {cvat_label} → {node}")
                    return node
            
            print(f"⚠️ Unmapped label: {cvat_label}")
        return mapped
    
    def get_visibility(self, point_elem) -> str:
        """Extract visibility from CVAT point attributes"""
        for attr in point_elem.findall('.//attribute'):
            if 'visibility' in attr.get('name', '').lower():
                return attr.text or 'visible'
        return 'visible'
    
    def convert_to_sleap(self, annotations: List[Dict], images_dir: str, output_path: str) -> None:
        """Convert annotations to SLEAP Labels format"""
        print(f"🔄 Converting to SLEAP format...")
        
        labeled_frames = []
        
        for ann in annotations:
            image_path = Path(images_dir) / ann['image_name']
            
            if not image_path.exists():
                print(f"⚠️  Image not found: {image_path}")
                continue
            
            # Load image to create video
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                    
                height, width = image.shape[:2]
            except Exception as e:
                print(f"❌ Error loading image {image_path}: {e}")
                continue
            
            # Create instances for each skeleton in the image
            instances = []
            
            for skeleton_data in ann['skeletons']:
                keypoints = skeleton_data['keypoints']
                
                # Create SLEAP points
                points = {}
                for node in self.skeleton.nodes:
                    if node.name in keypoints:
                        kp = keypoints[node.name]
                        
                        # Handle visibility
                        visible = kp.get('visibility', 'visible') != 'absent'
                        if visible:
                            points[node] = sleap.Point(x=kp['x'], y=kp['y'], visible=True)
                        else:
                            points[node] = sleap.Point(x=np.nan, y=np.nan, visible=False)
                    else:
                        # Missing keypoint
                        points[node] = sleap.Point(x=np.nan, y=np.nan, visible=False)
                
                # Create instance
                instance = sleap.Instance(skeleton=self.skeleton, points=points)
                instances.append(instance)
            
            if instances:
                # Create labeled frame
                video = sleap.Video.from_filename(str(image_path))
                frame = sleap.LabeledFrame(video=video, frame_idx=0, instances=instances)
                labeled_frames.append(frame)
        
        # Create Labels object and save
        labels = sleap.Labels(labeled_frames)
        labels.save(output_path)
        
        print(f"✅ SLEAP dataset saved: {output_path}")
        print(f"📊 Dataset statistics:")
        print(f"   • Images: {len(labeled_frames)}")
        print(f"   • Instances: {sum(len(lf.instances) for lf in labeled_frames)}")
        print(f"   • Skeleton: {len(self.skeleton.nodes)} nodes, {len(self.skeleton.edges)} edges")
    
    def create_training_config(self, dataset_path: str, output_dir: str) -> str:
        """Create SLEAP training configuration file"""
        config_path = Path(output_dir) / "training_config.json"
        
        config = {
            "data": {
                "labels": {
                    "training_labels": str(dataset_path),
                    "validation_labels": None,
                    "test_labels": None,
                    "split_by_video": True,
                    "training_fraction": 0.8,
                    "test_fraction": 0.1,
                    "validation_fraction": 0.1
                },
                "preprocessing": {
                    "ensure_rgb": True,
                    "ensure_grayscale": False,
                    "imagenet_mode": None,
                    "input_scaling": 1.0,
                    "pad_to_stride": 16,
                    "resize_and_pad_to_target": True,
                    "target_height": 384,
                    "target_width": 384
                },
                "instance_cropping": {
                    "center_on_part": "center",  # Using 'center' from 24-point system
                    "crop_size": 384,
                    "crop_size_detection_padding": 16
                }
            },
            "model": {
                "backbone": {
                    "leap": None,
                    "unet": {
                        "stem_stride": None,
                        "max_stride": 16,
                        "output_stride": 4,
                        "filters": 64,
                        "filters_rate": 2,
                        "middle_block": True,
                        "up_interpolate": True,
                        "stacks": 1
                    },
                    "hourglass": None,
                    "resnet": None,
                    "pretrained_encoder": None
                },
                "heads": {
                    "single_instance": None,
                    "centroid": None,
                    "centered_instance": {
                        "anchor_part": "center",  # Using 'center' from 24-point system
                        "part_names": self.node_names,
                        "sigma": 2.5,
                        "output_stride": 4,
                        "loss_weight": 1.0,
                        "offset_refinement": False
                    },
                    "multi_instance": None,
                    "multi_class_bottomup": None,
                    "multi_class_topdown": None
                },
                "base_loss_weight": 1.0,
                "regularization_loss_weight": 0.0001
            },
            "trainer": {
                "training_labels": str(dataset_path),
                "validation_labels": None,
                "test_labels": None,
                "base_checkpoint": None,
                "tensorboard": {
                    "write_logs": True,
                    "loss_frequency": "epoch",
                    "architecture_graph": False,
                    "profile_graph": False,
                    "log_dir": "logs"
                },
                "validation_frequency": 5,
                "early_stopping": {
                    "stop_training_on_plateau": True,
                    "plateau_min_delta": 1e-08,
                    "plateau_patience": 10,
                    "plateau_reduce_lr_on_plateau": True,
                    "plateau_reduce_lr_factor": 0.5,
                    "plateau_reduce_lr_patience": 5,
                    "plateau_reduce_lr_min": 1e-08
                },
                "steps_per_epoch": 200,
                "learning_rate": 0.0001,
                "epochs": 100,
                "batch_size": 4,
                "batches_per_epoch": None,
                "val_batches_per_epoch": None,
                "optimizer": "adam",
                "optimizer_params": {},
                "save_viz": True,
                "keep_viz": False,
                "viz_all_instances": True,
                "save_viz_frequency": 10,
                "split_by_video": True,
                "training_fraction": 0.8,
                "validation_fraction": 0.1,
                "test_fraction": 0.1,
                "seed": 1000
            },
            "inference": {
                "return_confmaps": False,
                "integral_refinement": True,
                "integral_patch_size": 5,
                "return_pafs": False,
                "return_paf_graph": False,
                "max_instances": 2,
                "multi_instance_peak_threshold": 0.2,
                "data_confidence_threshold": 0.5,
                "freeze_backbone": False,
                "force_grayscale": False
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📝 Training configuration saved: {config_path}")
        return str(config_path)

def main():
    parser = argparse.ArgumentParser(description='Convert CVAT annotations to SLEAP format')
    parser.add_argument('--cvat', required=True, help='Path to CVAT XML annotations file')
    parser.add_argument('--images', required=True, help='Directory containing training images')
    parser.add_argument('--output', required=True, help='Output SLEAP dataset path (.slp)')
    parser.add_argument('--config-dir', default='.', help='Directory to save training configuration')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.cvat):
        print(f"❌ CVAT file not found: {args.cvat}")
        return 1
    
    if not os.path.exists(args.images):
        print(f"❌ Images directory not found: {args.images}")
        return 1
    
    # Initialize converter
    converter = CVATToSLEAPConverter()
    
    try:
        # Parse CVAT annotations
        annotations = converter.parse_cvat_xml(args.cvat)
        
        if not annotations:
            print("❌ No valid annotations found in CVAT file")
            return 1
        
        # Convert to SLEAP
        converter.convert_to_sleap(annotations, args.images, args.output)
        
        # Create training configuration
        config_path = converter.create_training_config(args.output, args.config_dir)
        
        print(f"""
🎉 Conversion Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Dataset: {args.output}
📝 Config: {config_path}
🦴 Skeleton: {len(converter.skeleton.nodes)} keypoints

Next steps:
1. Review the dataset: sleap-label {args.output}
2. Train the model: sleap-train {config_path} {args.output}
3. Evaluate: sleap-track {args.output} --models /path/to/trained/model
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