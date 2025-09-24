# Quick Navigation Guide

## 🎯 Current Sprint: T0.5 Resubmission (Due Oct 1)

### Critical Files for T0.5 Work
```bash
# T0.5 Submission & Review
cd Shared/Conv2D/Agent_Reviews/
# - T0.5_SUBMISSION_PACKAGE.md
# - T0.5_COMMITTEE_REVIEW.md  
# - T0.5_FULL_REVIEW_DOCUMENTATION.md

# Implementation to fix
cd models/
# - t0_complete_implementation.py (needs real MI)
# - conv2d_vq_hdp_hsmm.py (HDP issue)

# Training scripts for ablation
cd scripts/training/
# - train_fsq_real_data_m13.py

# Data for validation
cd preprocessing/
# - enhanced_pipeline.py (PAMAP2 processing)
```

## 📍 Key Directories

### For Development
- `models/` - All model definitions and checkpoints
- `scripts/training/` - Training scripts
- `scripts/evaluation/` - Testing and evaluation
- `preprocessing/` - Data pipelines

### For Documentation
- `docs/architecture/` - System design
- `docs/deployment/` - Deployment guides
- `docs/status/` - Progress reports
- `Shared/Conv2D/Agent_Reviews/` - Gate reviews

### For Results
- `results/ablation/` - Ablation studies
- `results/training/` - Training logs
- `evaluation_data/` - Test datasets

## 🚀 Common Commands

```bash
# Run FSQ ablation on real PAMAP2
python scripts/training/train_fsq_real_data_m13.py

# Test the full pipeline
python scripts/evaluation/test_current_model.py

# Deploy to Hailo
python scripts/deployment/deploy_m13_fsq.py

# Check model size
python scripts/evaluation/check_actual_model_size.py
```

## 📊 Latest Results Location
- FSQ ablation: `results/fsq_ablation_final_20250921_232103.json`
- M15 training: `results/m15_training_results_20250922_141831.json`
- Best model: `models/m15_fsq_best_qa.pth`

## 🔧 Configuration Files
- `configs/enhanced_dataset_schema.yaml`
- `configs/improved_config.py`
- `config/training_config.py`

## 📦 Deployment Assets
- `hailo_export/` - Hailo compilation
- `CoreML_Pipeline/` - iOS integration
- `m13_fsq_deployment/` - Edge deployment

## 🗂️ Archive Location
- `archive/` - Old TCN work and legacy code