# PAMAP2 to Quadruped Mapping Documentation

## Processing Complete ✅

Successfully processed PAMAP2 human activity data for quadruped behavioral analysis.

### Dataset Statistics
- **Total Windows**: 27,281 samples
- **Data Shape**: (27,281, 9, 2, 100)
  - 9 channels: 3 accelerometer + 3 gyroscope + 3 zeros (padding)
  - 2 spatial dimensions: front/back leg pairs
  - 100 timesteps per window
- **Train/Val/Test Split**: 60/20/20
  - Train: 16,368 samples
  - Val: 5,456 samples  
  - Test: 5,457 samples

### Activity Distribution
| Activity | Samples | Percentage | Quadruped Behavior |
|----------|---------|------------|-------------------|
| Walk | 12,979 | 47.6% | Standard gait pattern |
| Stand | 3,798 | 13.9% | Static posture |
| Rest | 3,851 | 14.1% | Lying/resting |
| Sit | 3,703 | 13.6% | Sitting posture |
| Trot | 2,950 | 10.8% | Fast gait (from running) |

## Mapping Strategy

### Human to Quadruped Sensor Mapping

PAMAP2 has 3 IMU sensors on human body:
1. **Chest IMU** (torso/spine)
2. **Hand IMU** (dominant wrist)  
3. **Ankle IMU** (dominant ankle)

Mapped to quadruped configuration:
1. **Chest → Body reference** (not directly used in current Conv2d format)
2. **Hand → Front legs** (with phase shift for left/right)
3. **Ankle → Back legs** (with phase shift for left/right)

### Phase Coupling for Gait Simulation

To simulate realistic quadruped gaits from single-limb human data:
- **90-degree phase shift** (window_size/4 samples)
- **Diagonal coupling** for trot-like patterns:
  - Front-left ↔ Back-right (in phase)
  - Front-right ↔ Back-left (in phase)
  - Creates natural diagonal gait pattern

### Activity Mapping

Human activities mapped to quadruped behaviors:

| PAMAP2 Activity | Quadruped Behavior | Rationale |
|----------------|-------------------|-----------|
| Walking (4) | Walk | Direct mapping |
| Nordic Walking (7) | Walk | Similar gait pattern |
| Ascending Stairs (12) | Walk | Elevated gait variant |
| Descending Stairs (13) | Walk | Declined gait variant |
| Running (5) | Trot | Faster rhythmic gait |
| Rope Jumping (24) | Trot | Rhythmic bouncing |
| Playing Soccer (20) | Gallop | High energy, variable |
| Lying (1) | Rest | Prone position |
| Sitting (2) | Sit | Seated posture |
| Standing (3) | Stand | Static upright |

### Conv2d Format Adaptation

Final data format for Conv2d-VQ model:
```python
Shape: (N, 9, 2, 100)
- N: Number of samples
- 9: Channels (3 acc + 3 gyro + 3 padding)
- 2: Spatial dimensions
  - Row 0: Front leg data
  - Row 1: Back leg data
- 100: Temporal window (1 second at 100Hz)
```

## Next Steps

1. **Test with Conv2d-FSQ model**:
```bash
python evaluate_m15_real_data.py --data pamap2_quadruped
```

2. **Compare performance**:
- Expected: 65-70% accuracy (from 88.98% synthetic baseline)
- This is NORMAL and GOOD for real human→quadruped transfer

3. **Download LegKilo robot data** for better quadruped-specific training

## File Locations

- **Processed Data**: `/mnt/ssd/Conv2d_Datasets/quadruped_adapted/`
  - `pamap2_quadruped_processed.npz` - Full dataset
  - `pamap2_quadruped_train.npz` - Training split
  - `pamap2_quadruped_val.npz` - Validation split
  - `pamap2_quadruped_test.npz` - Test split
- **Processing Script**: `scripts/process_pamap2_quadruped.py`
- **Raw PAMAP2 Data**: `/mnt/ssd/Conv2d_Datasets/har_adapted/pamap2/`