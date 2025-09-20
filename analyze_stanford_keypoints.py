#!/usr/bin/env python3
"""Analyze Stanford Dogs keypoint structure"""

import json
import numpy as np
from pathlib import Path

# Load the Stanford Dogs annotations
json_path = Path('datasets/stanford_dogs/StanfordExtra_v12.json')
with open(json_path, 'r') as f:
    data = json.load(f)

# Examine the first few entries
print('Stanford Dogs Dataset Analysis')
print('=' * 60)
print(f'Total annotations: {len(data)}')
print()

# Check the structure of the first entry
first = data[0]
print('First entry structure:')
for key in first.keys():
    if key != 'joints':
        print(f'  {key}: {first[key]}')

# Analyze joints structure
joints = np.array(first['joints'])
print(f'\nJoints shape: {joints.shape}')
print(f'Number of keypoints: {joints.shape[0]}')
print()

# Show first few keypoints with their values
print('First 5 keypoints from sample 0:')
for i in range(min(5, len(joints))):
    x, y, vis = joints[i]
    print(f'  Point {i}: x={x:.1f}, y={y:.1f}, visibility={vis}')

print('\nLast 5 keypoints from sample 0:')
for i in range(max(0, len(joints)-5), len(joints)):
    x, y, vis = joints[i]
    print(f'  Point {i}: x={x:.1f}, y={y:.1f}, visibility={vis}')

# Check how many points are typically annotated (non-zero)
print('\nKeypoint visibility analysis (first 10 samples):')
print('-' * 60)
for i in range(min(10, len(data))):
    joints = np.array(data[i]['joints'])
    visible = joints[:, 2] > 0  # visibility flag
    annotated = np.sum((joints[:, 0] != 0) | (joints[:, 1] != 0))
    print(f'Sample {i}: {np.sum(visible)}/{len(joints)} visible, {annotated}/{len(joints)} have coordinates')

# Check all unique joint counts
print('\nKeypoint statistics across all samples:')
all_visible = []
all_annotated = []
for entry in data[:1000]:  # Check first 1000 for speed
    joints = np.array(entry['joints'])
    visible = np.sum(joints[:, 2] > 0)
    annotated = np.sum((joints[:, 0] != 0) | (joints[:, 1] != 0))
    all_visible.append(visible)
    all_annotated.append(annotated)

print(f'Visible points - Min: {min(all_visible)}, Max: {max(all_visible)}, Avg: {np.mean(all_visible):.1f}')
print(f'Annotated points - Min: {min(all_annotated)}, Max: {max(all_annotated)}, Avg: {np.mean(all_annotated):.1f}')

# Count frequency of different visibility patterns
from collections import Counter
vis_counter = Counter(all_visible)
print(f'\nVisibility distribution (top 5):')
for count, freq in vis_counter.most_common(5):
    print(f'  {count} visible points: {freq} samples ({freq/len(all_visible)*100:.1f}%)')

# Check which indices are commonly zero
print('\nChecking which keypoint indices are commonly unused:')
unused_counts = np.zeros(24)
for entry in data[:1000]:
    joints = np.array(entry['joints'])
    for i in range(len(joints)):
        if joints[i, 0] == 0 and joints[i, 1] == 0 and joints[i, 2] == 0:
            unused_counts[i] += 1

print('Keypoint usage (% of samples with this point annotated):')
for i in range(24):
    usage = (1000 - unused_counts[i]) / 1000 * 100
    status = 'COMMONLY USED' if usage > 50 else 'RARELY USED' if usage < 10 else 'SOMETIMES USED'
    print(f'  Point {i:2d}: {usage:5.1f}% - {status}')