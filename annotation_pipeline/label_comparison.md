# Label Configuration Comparison

## Current SLEAP Function (14 points)
Simple skeleton with basic keypoints:
1. nose
2. left_ear
3. right_ear
4. left_eye
5. right_eye
6. withers (shoulder area)
7. throat
8. tail_base
9. tail_tip
10. left_front_paw
11. right_front_paw
12. left_rear_paw
13. right_rear_paw
14. center

## Your CVAT Annotation Config (26 points)
Full skeleton with all joints (IDs 2-27 in the JSON):
1. **nose** ✓ (matches SLEAP)
2. **left_eye** ✓ (matches SLEAP)
3. **right_eye** ✓ (matches SLEAP)
4. **left_ear** ✓ (matches SLEAP)
5. **right_ear** ✓ (matches SLEAP)
6. **neck** (new - replaces throat?)
7. **left_front_shoulder** (new)
8. **left_front_elbow** (new)
9. **left_front_wrist** (new)
10. **left_front_paw** ✓ (matches SLEAP)
11. **right_front_shoulder** (new)
12. **right_front_elbow** (new)
13. **right_front_wrist** (new)
14. **right_front_paw** ✓ (matches SLEAP)
15. **spine_mid** (new - could map to center?)
16. **left_back_shoulder** (new - hip joint)
17. **left_back_elbow** (new - knee joint)
18. **left_back_wrist** (new - ankle joint)
19. **left_back_paw** ✓ (matches SLEAP left_rear_paw)
20. **right_back_shoulder** (new - hip joint)
21. **right_back_elbow** (new - knee joint)
22. **right_back_wrist** (new - ankle joint)
23. **right_back_paw** ✓ (matches SLEAP right_rear_paw)
24. **tail_base** ✓ (matches SLEAP)
25. **tail_mid** (new)
26. **tail_tip** ✓ (matches SLEAP)

## Key Differences:
1. **SLEAP** has "withers" and "throat" - not in CVAT config
2. **SLEAP** has "center" - not in CVAT config (but spine_mid might serve this purpose)
3. **CVAT** has full joint hierarchy (shoulder→elbow→wrist→paw) for all 4 limbs
4. **CVAT** has "neck" instead of "throat"
5. **CVAT** has "spine_mid" and "tail_mid" for more detailed spine/tail tracking

## Proposed Action:
Replace SLEAP's 14-point skeleton with your CVAT 26-point skeleton for:
- More detailed pose estimation
- Better tracking of joint angles
- Compatible with iOS Vision framework
- Suitable for TCN-VAE training pipeline