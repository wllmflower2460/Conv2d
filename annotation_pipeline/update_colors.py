#!/usr/bin/env python3
"""
Update CVAT label colors to match the visualization documentation:
- Head points (red-yellow spectrum): nose, eyes, ears
- Front legs (green-cyan spectrum): shoulders, elbows, wrists, paws  
- Back legs (blue-purple spectrum): hips, knees, ankles, paws
- Spine/Torso (yellow-blue): neck, spine_mid, tail_base
- Tail (dark red/brown): base, mid, tip
"""

import json

# Define color mapping based on documentation
color_mapping = {
    # Head points (red-yellow spectrum) 🔴
    "nose": "#ff0000",           # Pure red
    "left_eye": "#ff3300",        # Red-orange
    "right_eye": "#ff6600",       # Orange-red
    "left_ear": "#ff9900",        # Orange
    "right_ear": "#ffcc00",       # Yellow-orange
    
    # Spine/Torso (yellow-blue gradient) 🟡
    "neck": "#ffff00",            # Yellow
    "spine_mid": "#00ccff",       # Light blue
    "tail_base": "#0099ff",       # Sky blue
    
    # Front left leg (green spectrum) 🟢
    "left_front_shoulder": "#00ff00",   # Pure green
    "left_front_elbow": "#00ff44",      # Green
    "left_front_wrist": "#00ff88",      # Light green
    "left_front_paw": "#00ffcc",        # Cyan-green
    
    # Front right leg (green-cyan spectrum) 🟢
    "right_front_shoulder": "#44ff00",   # Yellow-green
    "right_front_elbow": "#88ff00",      # Light yellow-green
    "right_front_wrist": "#ccff00",      # Yellow-green
    "right_front_paw": "#00ffff",        # Cyan
    
    # Back left leg (blue spectrum) 🔵
    "left_hip": "#0000ff",               # Pure blue
    "left_back_elbow": "#3333ff",        # Blue
    "left_back_wrist": "#6666ff",        # Light blue
    "left_back_paw": "#9999ff",          # Very light blue
    
    # Back right leg (purple spectrum) 🔵
    "right_hip": "#6600ff",              # Blue-purple
    "right_back_elbow": "#9900ff",       # Purple
    "right_back_wrist": "#cc00ff",       # Light purple
    "right_back_paw": "#ff00ff",         # Magenta
    
    # Tail (dark red/brown spectrum) 🟤
    "tail_mid": "#660000",               # Dark brown-red
    "tail_tip": "#330000",               # Very dark brown
}

# Load the JSON file
with open('/home/wllmflower/Development/tcn-vae-training/annotation_pipeline/cvat_labels_config.json', 'r') as f:
    data = json.load(f)

# Update colors
updated_count = 0
for label in data['labels']:
    if 'sublabels' in label:
        for point in label['sublabels']:
            point_name = point['name']
            if point_name in color_mapping:
                old_color = point.get('color', 'none')
                new_color = color_mapping[point_name]
                point['color'] = new_color
                updated_count += 1
                print(f"Updated {point_name:25} from {old_color} to {new_color}")

# Save the updated JSON
with open('/home/wllmflower/Development/tcn-vae-training/annotation_pipeline/cvat_labels_config.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n✅ Updated {updated_count} keypoint colors")
print("📁 Saved to cvat_labels_config.json")