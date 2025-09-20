#!/usr/bin/env python3
"""
CVAT Individual Points to Skeleton Converter
Converts individual point annotations to skeleton format for training
This bypasses CVAT's skeleton bug while maintaining annotation speed
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple

# Define the 24-point skeleton structure
SKELETON_STRUCTURE = {
    'points': [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'throat', 'withers', 
        'left_front_shoulder', 'left_front_elbow', 'left_front_paw',
        'right_front_shoulder', 'right_front_elbow', 'right_front_paw',
        'center',
        'left_hip', 'left_knee', 'left_back_paw',
        'right_hip', 'right_knee', 'right_back_paw',
        'tail_base', 'tail_mid_1', 'tail_mid_2', 'tail_tip'
    ],
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

class PointToSkeletonConverter:
    def __init__(self):
        self.skeleton = SKELETON_STRUCTURE
        self.stats = {
            'images_processed': 0,
            'skeletons_created': 0,
            'points_connected': 0,
            'missing_points': []
        }
    
    def convert_cvat_xml(self, input_path: str, output_path: str):
        """Convert CVAT individual points to skeleton format"""
        print(f"📖 Reading CVAT annotations from: {input_path}")
        
        tree = ET.parse(input_path)
        root = tree.getroot()
        
        # Process each image
        for image in root.findall('.//image'):
            self.stats['images_processed'] += 1
            self.process_image_annotations(image)
        
        # Save modified XML
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"✅ Saved skeleton format to: {output_path}")
        self.print_stats()
    
    def process_image_annotations(self, image_elem):
        """Group individual points into skeletons"""
        image_name = image_elem.get('name')
        
        # Collect all points in this image
        points_dict = {}
        points_to_remove = []
        
        for points_elem in image_elem.findall('.//points'):
            label = points_elem.get('label', '')
            if label in self.skeleton['points']:
                coords = points_elem.get('points', '').split(',')
                if len(coords) >= 2:
                    points_dict[label] = {
                        'x': float(coords[0]),
                        'y': float(coords[1]),
                        'elem': points_elem
                    }
                    points_to_remove.append(points_elem)
        
        if len(points_dict) >= 10:  # Minimum viable skeleton
            # Remove individual points
            for elem in points_to_remove:
                image_elem.remove(elem)
            
            # Create skeleton element
            skeleton_elem = self.create_skeleton_element(points_dict)
            image_elem.append(skeleton_elem)
            self.stats['skeletons_created'] += 1
            self.stats['points_connected'] += len(points_dict)
            
            print(f"  ✓ Created skeleton for {image_name} with {len(points_dict)} points")
        else:
            print(f"  ⚠ Skipped {image_name} - only {len(points_dict)} valid points found")
    
    def create_skeleton_element(self, points_dict: Dict) -> ET.Element:
        """Create a skeleton XML element from individual points"""
        skeleton = ET.Element('skeleton')
        skeleton.set('label', 'dog_pose')
        skeleton.set('source', 'manual')
        skeleton.set('occluded', '0')
        skeleton.set('z_order', '0')
        
        # Add each point as a sublabel
        for idx, point_name in enumerate(self.skeleton['points']):
            if point_name in points_dict:
                point_data = points_dict[point_name]
                point_elem = ET.SubElement(skeleton, 'points')
                point_elem.set('label', point_name)
                point_elem.set('points', f"{point_data['x']:.2f},{point_data['y']:.2f}")
                point_elem.set('occluded', '0')
                
                # Add visibility attribute
                attr_elem = ET.SubElement(point_elem, 'attribute')
                attr_elem.set('name', 'visibility')
                attr_elem.text = 'visible'
            else:
                # Add placeholder for missing points
                self.stats['missing_points'].append(point_name)
        
        return skeleton
    
    def print_stats(self):
        """Print conversion statistics"""
        print("\n📊 Conversion Statistics:")
        print(f"  Images processed: {self.stats['images_processed']}")
        print(f"  Skeletons created: {self.stats['skeletons_created']}")
        print(f"  Points connected: {self.stats['points_connected']}")
        if self.stats['missing_points']:
            missing_freq = {}
            for point in self.stats['missing_points']:
                missing_freq[point] = missing_freq.get(point, 0) + 1
            print(f"  Most frequently missing points:")
            for point, count in sorted(missing_freq.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    - {point}: {count} times")

def create_annotation_template():
    """Create a template for quick annotation"""
    template = {
        'annotation_order': [
            # Phase 1: Essential points (17 points, ~30 seconds)
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'withers', 'center',
            'left_front_paw', 'right_front_paw',
            'left_back_paw', 'right_back_paw',
            'left_hip', 'right_hip',
            'tail_base', 'tail_tip',
            'throat',
            
            # Phase 2: Details (7 points, +20 seconds)
            'left_front_shoulder', 'left_front_elbow',
            'right_front_shoulder', 'right_front_elbow',
            'left_knee', 'right_knee',
            'tail_mid_1', 'tail_mid_2'
        ],
        'keyboard_shortcuts': {
            '1': 'nose',
            '2': 'left_eye',
            '3': 'right_eye',
            '4': 'left_ear',
            '5': 'right_ear',
            '6': 'throat',
            '7': 'withers',
            'q': 'left_front_paw',
            'w': 'right_front_paw',
            'e': 'left_back_paw',
            'r': 'right_back_paw',
            't': 'tail_base',
            'y': 'tail_tip',
            'a': 'center',
            's': 'left_hip',
            'd': 'right_hip'
        }
    }
    
    with open('annotation_template.json', 'w') as f:
        json.dump(template, f, indent=2)
    
    print("📝 Created annotation_template.json with recommended workflow")

def main():
    parser = argparse.ArgumentParser(description='Convert CVAT individual points to skeleton format')
    parser.add_argument('--input', required=True, help='Input CVAT XML with individual points')
    parser.add_argument('--output', required=True, help='Output CVAT XML with skeletons')
    parser.add_argument('--create-template', action='store_true', help='Create annotation template')
    
    args = parser.parse_args()
    
    if args.create_template:
        create_annotation_template()
        return
    
    converter = PointToSkeletonConverter()
    converter.convert_cvat_xml(args.input, args.output)
    
    print("\n✨ Conversion complete!")
    print("You can now use this with your SLEAP/YOLOv8 converters")

if __name__ == "__main__":
    main()