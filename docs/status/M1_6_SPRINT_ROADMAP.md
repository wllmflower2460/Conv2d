# M1.6 Sprint Roadmap - Real-World Dataset Validation

**Sprint Duration**: 3 Weeks (Sept 23 - Oct 13, 2025)  
**Storage Available**: 1.7TB on RAID, 161GB on main drive  
**Current Data**: ~150MB (synthetic only)  
**Target**: 70-85% real-world accuracy  

---

## Executive Summary

The M1.5 committee review achieved 88.98% accuracy on synthetic quadruped data, but we need real-world validation. This sprint focuses on acquiring and validating against actual IMU datasets from quadruped robots, animals, and comparable platforms.

## Current State vs Target

### Current (M1.5)
- **Data**: Synthetic quadruped behavioral data only
- **Accuracy**: 88.98% on synthetic validation set
- **Size**: ~150MB total data
- **Validation**: No real-world testing

### Target (M1.6)
- **Data**: 5+ real-world datasets (100GB+)
- **Accuracy**: 70-85% on real data (expected 15-30% drop)
- **Validation**: Cross-dataset, cross-platform testing
- **Deployment**: Ready for physical Hailo-8 testing

---

## Week 1: Immediate Access Datasets (Sept 23-29)

### Day 1-2: Setup & TartanVO
```bash
# Install TartanVO toolkit
pip install tartanair

# Download simulated IMU data (bridge to real)
python scripts/download_tartanvo.py --env Downtown --modality imu
python scripts/download_tartanvo.py --env OldTown --modality imu
```
- **Size**: ~10GB per environment
- **IMU**: 1000Hz, realistic noise model
- **Purpose**: Semi-synthetic baseline

### Day 3-4: PAMAP2 Adaptation
```bash
# Download PAMAP2 (100Hz IMU, multi-position)
wget https://archive.ics.uci.edu/ml/machine-learning-databases/pamap2/
python scripts/adapt_pamap2_to_quadruped.py
```
- **Size**: 2.1GB
- **Adaptation**: Map chest→body, wrists→front limbs, ankles→rear limbs
- **Classes**: 18 activities including various locomotion modes

### Day 5-7: LegKilo Unitree Go1
```bash
# Direct download real quadruped data
python scripts/download_legkilo.py
python scripts/process_rosbags.py --dataset legkilo
```
- **Size**: 3GB per sequence (7 sequences)
- **IMU**: 50Hz from actual robot
- **Ground Truth**: Offline optimization

**Week 1 Deliverables**:
- [ ] 3 datasets downloaded and preprocessed
- [ ] Initial accuracy baseline on each
- [ ] Performance drop analysis report

---

## Week 2: Animal & Robot Datasets (Sept 30 - Oct 6)

### Day 1-2: Horse Gait Dataset
```python
# Scientific Reports horse dataset
python scripts/download_horse_gaits.py
# 120 horses, 7 IMUs per horse, 8 gaits
```
- **Size**: ~15GB
- **IMU**: 200-500Hz, 7 sensors per animal
- **Gaits**: Walk, trot, canter, tölt, pace, trocha, paso fino

### Day 3-4: Dog Behavior Dataset
```python
# Mendeley dog behavior dataset
python scripts/download_dog_behavior.py
# 45 dogs, collar + harness positions
```
- **Size**: ~5GB
- **IMU**: ActiGraph GT9X, 100Hz
- **Behaviors**: 7 classes validated with video

### Day 5-7: CEAR Mini Cheetah
```python
# MIT Mini Cheetah alternative
python scripts/download_cear_dataset.py
```
- **Size**: ~20GB
- **Features**: Backflips, bounding, pronking
- **Environments**: 31 different terrains

**Week 2 Deliverables**:
- [ ] Cross-species validation complete
- [ ] Failure mode analysis document
- [ ] Domain adaptation strategy defined

---

## Week 3: Robustness & Deployment (Oct 7-13)

### Day 1-2: Drone Datasets (Dynamic Validation)
```python
# UZH-FPV for aggressive trajectories
python scripts/download_uzh_fpv.py

# Blackbird for large-scale validation
python scripts/download_blackbird.py
```
- **Purpose**: Extreme dynamics testing
- **IMU**: 100Hz with ground truth

### Day 3-4: Domain Adaptation
```python
# Implement transfer learning pipeline
python train_domain_adaptation.py \
  --synthetic_ratio 0.5 \
  --real_datasets "tartanvo,legkilo,pamap2" \
  --adaptation_method "adversarial"
```

### Day 5-7: Deployment Testing
```python
# Export for Hailo-8
python export_for_hailo_m16.py

# Benchmark on real hardware
python benchmark_hailo_deployment.py
```

**Week 3 Deliverables**:
- [ ] Domain-adapted model trained
- [ ] 70-85% real-world accuracy achieved
- [ ] Hailo-8 deployment validated
- [ ] M1.6 gate review package

---

## Data Management Strategy

### Storage Layout
```
/mnt/ssd/Conv2d_Datasets/            # 1.8TB SSD (primary)
/mnt/raid1/Conv2d_Datasets_Backup/   # 1.7TB RAID (backup)
├── synthetic/                        # Current 150MB
├── semi_synthetic/
│   └── tartanvo/                    # 30GB
├── real_quadruped/
│   ├── legkilo/                     # 21GB
│   ├── cear_mini_cheetah/           # 20GB
│   └── spot_assessment/             # 10GB
├── animal_locomotion/
│   ├── horse_gaits/                 # 15GB
│   └── dog_behavior/                # 5GB
├── har_adapted/
│   ├── pamap2/                      # 2.1GB
│   └── opportunity/                 # 4GB
└── dynamic_validation/
    ├── uzh_fpv/                     # 10GB
    └── blackbird/                   # 25GB
```
**Total**: ~140GB (8% of available RAID storage)

### Download Scripts
```python
# scripts/dataset_manager.py
class DatasetManager:
    def __init__(self, base_path="/mnt/raid1/Conv2d_Datasets"):
        self.datasets = {
            'tartanvo': TartanVODownloader(),
            'legkilo': LegKiloDownloader(),
            'pamap2': PAMAP2Downloader(),
            'horse_gaits': HorseGaitDownloader(),
            # ... etc
        }
    
    def download_all_phase1(self):
        """Week 1 datasets"""
        self.datasets['tartanvo'].download(['Downtown', 'OldTown'])
        self.datasets['pamap2'].download()
        self.datasets['legkilo'].download()
    
    def validate_downloads(self):
        """Verify integrity"""
        for name, dataset in self.datasets.items():
            dataset.verify_checksum()
            dataset.validate_format()
```

---

## Evaluation Protocol

### Metrics Tracking
| Dataset | Expected Accuracy | Actual | Drop from Synthetic |
|---------|------------------|--------|---------------------|
| TartanVO (semi) | 80-85% | TBD | TBD |
| LegKilo (real) | 70-75% | TBD | TBD |
| PAMAP2 (adapted) | 65-70% | TBD | TBD |
| Horse Gaits | 75-80% | TBD | TBD |
| Dog Behavior | 70-75% | TBD | TBD |

### Success Criteria
- [ ] ≥70% accuracy on at least 3 real datasets
- [ ] ≥75% on robot quadruped data
- [ ] <30% drop from synthetic baseline
- [ ] Successful Hailo-8 deployment
- [ ] Cross-dataset generalization

---

## Risk Mitigation

### Storage Risks
- **Mitigation**: Use RAID storage (1.7TB available)
- **Cleanup**: Remove intermediate files after processing
- **Compression**: Store raw data compressed

### Performance Risks  
- **Expected**: 15-30% accuracy drop is NORMAL
- **Mitigation**: Domain adaptation, ensemble methods
- **Fallback**: Use semi-synthetic data for gradual transition

### Timeline Risks
- **Downloads**: May take 2-3 days for large datasets
- **Processing**: Parallelize preprocessing pipelines
- **Training**: Use overnight training scripts

---

## Implementation Checklist

### Week 1 Tasks
- [ ] Set up RAID storage structure
- [ ] Create `scripts/dataset_manager.py`
- [ ] Download TartanVO environments
- [ ] Process PAMAP2 for quadruped mapping
- [ ] Acquire LegKilo Unitree Go1 data
- [ ] Run initial evaluation baseline
- [ ] Document performance drops

### Week 2 Tasks
- [ ] Download horse gait dataset
- [ ] Process dog behavior data
- [ ] Acquire CEAR Mini Cheetah
- [ ] Cross-species validation
- [ ] Analyze failure modes
- [ ] Start domain adaptation

### Week 3 Tasks
- [ ] Integrate drone datasets
- [ ] Train domain-adapted model
- [ ] Export for Hailo-8
- [ ] Hardware deployment test
- [ ] Prepare M1.6 gate review
- [ ] Document final results

---

## Committee Notes Integration

Per M1.5 recommendations, while acquiring real data, also implement:
1. **5-Model Ensemble** - Use different data subsets per model
2. **Predictive Horizons** - Test on temporal sequences
3. **Background Mining** - Discover patterns in new datasets

---

## Next Steps

1. **Immediate** (Today):
   ```bash
   # Create storage structure
   sudo mkdir -p /mnt/raid1/Conv2d_Datasets
   sudo chown $USER:$USER /mnt/raid1/Conv2d_Datasets
   
   # Install dataset tools
   pip install tartanair rosbag pandas
   
   # Start first download
   python scripts/download_tartanvo.py
   ```

2. **Tomorrow**: Begin PAMAP2 adaptation
3. **This Week**: Complete Phase 1 datasets

The journey from 88.98% synthetic to 70-85% real-world accuracy is normal and expected. This sprint provides the data foundation for genuine deployment readiness.