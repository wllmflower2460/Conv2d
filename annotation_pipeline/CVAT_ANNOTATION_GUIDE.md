# CVAT Pose Annotation Guide for DataDogs

## 🎯 Overview

This guide covers **keypoint/skeleton annotation** for animal pose estimation in CVAT. We're annotating **25 anatomical landmarks** that correspond to iOS 17+ Vision framework joint detection.

## 📋 Setup Instructions

### 1. Import Label Configuration

```bash
# Import the label configuration into your CVAT project
# In CVAT web interface:
# Project Settings → Labels → Import → Upload cvat_labels_config.json
```

### 2. CVAT Project Setup

**Project Creation:**
- Project Name: `DataDogs Pose Estimation`
- Task Name: `Batch_YYYY-MM-DD` (e.g., `Batch_2025-09-11`)
- Labels: Import from `cvat_labels_config.json`

**Annotation Settings:**
- **Mode**: Skeleton annotation (not segmentation)
- **Tool**: Skeleton tool with keypoints
- **Labels**: Use the 25-point dog skeleton

## 🔧 Annotation Process

### **Step 1: Select Skeleton Tool**
1. Open image in CVAT annotation interface
2. Select **"Skeleton"** tool from toolbar
3. Choose **"dog_skeleton"** label

### **Step 2: Place Keypoints**
Place keypoints in this **recommended order** for consistency:

#### **Head Region (Start Here)**
1. **nose** - Tip of the nose/snout
2. **left_eye** - Center of left eye socket  
3. **right_eye** - Center of right eye socket
4. **left_ear** - Base of left ear attachment
5. **right_ear** - Base of right ear attachment

#### **Torso & Spine**
6. **neck** - Base of neck where it meets shoulders
7. **spine_mid** - Center of back/spine midpoint
8. **tail_base** - Where tail connects to body
9. **tail_mid** - Middle of tail (if visible)
10. **tail_tip** - Tip of tail (if visible)

#### **Front Legs (Left Side)**
11. **left_front_shoulder** - Shoulder joint
12. **left_front_elbow** - Elbow joint  
13. **left_front_wrist** - Wrist/ankle joint
14. **left_front_paw** - Paw contact point

#### **Front Legs (Right Side)**  
15. **right_front_shoulder** - Shoulder joint
16. **right_front_elbow** - Elbow joint
17. **right_front_wrist** - Wrist/ankle joint  
18. **right_front_paw** - Paw contact point

#### **Back Legs (Left Side)**
19. **left_hip** - Hip joint
20. **left_back_elbow** - Knee joint
21. **left_back_wrist** - Ankle joint
22. **left_back_paw** - Paw contact point

#### **Back Legs (Right Side)**
23. **right_hip** - Hip joint  
24. **right_back_elbow** - Knee joint
25. **right_back_wrist** - Ankle joint
26. **right_back_paw** - Paw contact point

### **Step 3: Handle Occlusions**

**Visibility States:**
- **visible** - Joint clearly visible and accurately placeable
- **occluded** - Joint blocked by another body part but location inferrable  
- **absent** - Joint completely out of frame or not determinable

**Best Practices:**
- When unsure, choose **"occluded"** rather than guess
- Place occluded joints at best estimated position
- Use anatomical knowledge to infer occluded joint positions

### **Step 4: Quality Attributes**

For each skeleton, set these attributes:

**Animal Classification:**
- **animal_type**: dog, cat
- **breed_size**: small, medium, large, giant

**Pose Assessment:**  
- **activity_type**: standing, sitting, lying, walking, running, playing, other
- **pose_quality**: excellent, good, fair, poor

**Quality Standards:**
- **excellent**: All joints visible, clear pose, high image quality
- **good**: Most joints visible, minor occlusions, good image quality  
- **fair**: Some joints occluded, acceptable for training
- **poor**: Heavy occlusions, blurry, should be rejected

## 🎨 Visual Guidelines

### **Keypoint Placement Principles**

1. **Anatomical Accuracy**: Place at actual joint centers, not surface features
2. **Sub-pixel Precision**: Zoom in for accurate placement
3. **Consistency**: Use same anatomical landmarks across all images
4. **Perspective Handling**: Account for camera angle and foreshortening

### **Common Mistakes to Avoid**

❌ **DON'T**: Place keypoints on fur/surface  
✅ **DO**: Place at underlying anatomical joint

❌ **DON'T**: Guess occluded joint positions wildly  
✅ **DO**: Use anatomical knowledge and mark as occluded

❌ **DON'T**: Skip difficult poses  
✅ **DO**: Annotate what's visible, mark rest as occluded

## 📊 Quality Control

### **Self-Check Before Submission**
- [ ] All visible joints annotated accurately
- [ ] Occluded joints marked appropriately  
- [ ] Skeleton connections look anatomically correct
- [ ] Activity type and pose quality attributes set
- [ ] No obvious mistakes in left/right assignment

### **Batch Processing Tips**
- **Start with easiest poses** (standing, clear visibility)
- **Save frequently** - CVAT auto-saves but manual saves prevent loss
- **Take breaks** - Annotation fatigue leads to errors
- **Review previous annotations** when returning to work

## 🔄 Integration with DataDogs Pipeline

### **Export Format**
When exporting annotations:
1. **Format**: CVAT for images 1.1
2. **Include**: All keypoints, attributes, and skeleton connections
3. **File naming**: Maintain original Firebase image IDs

### **Quality Metrics Tracking**
The pipeline tracks:
- **Annotation time per image**
- **Inter-annotator agreement**  
- **Model improvement metrics**
- **Annotation quality scores**

## 🚀 Pro Tips for Efficient Annotation

### **Keyboard Shortcuts**
- `N` - Next image
- `P` - Previous image  
- `Ctrl+S` - Save annotation
- `Space` - Play/pause video mode
- `+/-` - Zoom in/out

### **Workflow Optimization**
1. **Pre-scan batch** - Review all images quickly first
2. **Group similar poses** - Annotate similar poses together
3. **Use reference images** - Keep anatomical references open
4. **Calibrate regularly** - Check against other annotators

### **Dealing with Edge Cases**
- **Multiple animals**: Annotate primary/foreground animal
- **Partial frames**: Annotate visible portions accurately
- **Motion blur**: Mark affected joints as occluded
- **Unusual poses**: Focus on anatomical accuracy over appearance

## 📈 Success Metrics

**Target Metrics for Quality:**
- **Accuracy**: >95% keypoint placement within 5-pixel tolerance
- **Consistency**: <10% inter-annotator variance
- **Coverage**: >90% of visible joints annotated
- **Speed**: 2-5 minutes per high-quality annotation

**Pipeline Integration:**
- Annotations feed back to Firebase automatically
- Model retraining triggered at 50+ new annotations  
- Quality scores tracked for annotator performance
- A/B testing compares annotation sources

---

**🎯 Ready to Start Annotating!**

Your pose annotations directly improve the DataDogs behavioral analysis models used by thousands of pet owners and professional trainers. Quality matters - take your time and focus on anatomical accuracy!