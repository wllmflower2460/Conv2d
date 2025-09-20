# Conv2d-VQ-HDP-HSMM Project Context Overview

**Created**: 2025-09-19
**Purpose**: Comprehensive shared understanding of project background and context
**Repository**: Conv2d (forked from TCN-VAE_training_pipeline)

## Project Evolution Timeline

### Foundation: TCN-VAE Pipeline (Completed)
- **Status**: Production-ready system with 86.53% accuracy on human behavior, 71.9% on quadruped
- **Architecture**: Temporal Convolutional Network + Variational Autoencoder
- **Deployment**: Hailo-8 compatible, EdgeInfer integration, Docker containerization
- **Key Achievement**: Real-time behavioral classification with clustering and ethogram visualization

### Current Evolution: Conv2d-VQ-HDP-HSMM (Active Development)
- **Innovation**: Unified discrete-continuous behavioral synchrony measurement
- **Core Components**: Vector Quantization + Hierarchical Dirichlet Process + Hidden Semi-Markov Model
- **Target**: Real-time human-animal synchrony detection with uncertainty quantification
- **Timeline**: 4-week implementation roadmap toward PhD application (December 2025)

## Technical Foundation

### Proven Components (Ready for Reuse)
From the TCN-VAE pipeline, the following are production-validated:

1. **HailoTemporalBlock** (`models/tcn_vae_hailo.py`)
   - Conv2d-based temporal convolutions with dilated causal structure
   - Hailo-8 validated and deployed
   - Supports multi-scale feature extraction

2. **DeviceAttention** (`models/device_attention.py`)
   - Phone+IMU dual-device processing
   - Cross-device attention mechanism for synchrony measurement
   - Handles temporal alignment and feature fusion

3. **EnhancedCrossSpeciesDataset** (`preprocessing/enhanced_pipeline.py`)
   - YAML-driven configuration system
   - Cross-species behavioral mappings (human ↔ dog)
   - Data augmentation and validation pipeline

4. **Training Infrastructure** (`training/`)
   - Monitoring scripts with real-time metrics
   - Overnight training setup with checkpointing
   - Progress reporting and visualization tools

### Current Performance Benchmarks
- **Human Behavior Classification**: 86.53% accuracy (WISDM + HAPT datasets)
- **Quadruped Behavior Classification**: 71.9% accuracy (21 behavioral classes)
- **Real-time Performance**: <100ms dashboard updates, <50ms inference latency
- **Behavioral Motif Discovery**: K-means clustering with 3 motifs (silhouette score 0.270)

## Synchrony Science Foundation

### Theoretical Integration
The Conv2d-VQ-HDP-HSMM architecture bridges three foundational approaches:

1. **Feldman's Bio-Behavioral Synchrony** (Discrete States)
   - States: Synchronized, Leading, Following, Disengaged
   - Focus on attachment-driven state transitions
   - Clinical assessment applications

2. **Kelso's Coordination Dynamics** (Continuous Phase)
   - Order parameters: relative phase (φ), Kuramoto synchronization
   - Bifurcations and critical transitions
   - Movement frequency and coupling strength

3. **Leclère's Measurement Standards** (Methodological)
   - Multi-level assessment requirements
   - Context-dependent optimal synchrony
   - Cross-study standardization needs

### Novel Contribution: Behavioral-Dynamical Coherence
**Core Innovation**: I(Z;Φ) - mutual information between discrete behavioral states and continuous phase dynamics
- High I(Z;Φ): Actions and timing tightly coupled (genuine synchrony)
- Low I(Z;Φ): Surface synchrony without behavioral alignment
- Optimal I(Z;Φ): Context-dependent balance between rigidity and flexibility

## Implementation Strategy

### 4-Week Development Plan

**Week 1: Vector Quantization Integration**
- Implement VectorQuantizer class with 512-code behavioral codebook
- Add commitment loss and perplexity monitoring
- Test codebook learning on existing encoder features
- Validate Hailo-8 compatibility

**Week 2: HDP Clustering Implementation**
- Implement HDPLayer with stick-breaking construction
- Add cluster assignment logic with Gumbel-softmax
- Monitor active clusters dynamically (target: 5-15 clusters)
- Test behavioral cluster discovery

**Week 3: HSMM Temporal Dynamics**
- Implement forward-backward algorithm for state inference
- Add duration modeling with negative binomial distributions
- Test temporal coherence and state transitions
- Validate duration-based relationship assessment

**Week 4: Full System Integration**
- Combine all components in unified architecture
- Implement dual-pathway processing (discrete + continuous)
- Add entropy-based uncertainty quantification
- Validate end-to-end performance and Hailo export

### Key Design Decisions

1. **Why VQ-VAE over Standard VAE?**
   - Discrete representations provide interpretable behavioral vocabulary
   - Natural clustering of similar behaviors
   - Better generalization through quantization
   - Foundation for hierarchical organization

2. **Why HDP over Fixed Clustering?**
   - Automatic discovery of optimal cluster count
   - Hierarchical organization of behaviors
   - Bayesian uncertainty quantification
   - Flexible complexity based on data

3. **Why HSMM over HMM?**
   - Explicit duration modeling captures behavior persistence
   - More realistic for behavioral sequences
   - Better handling of varying activity lengths
   - Duration distributions predictive of relationship quality

## Hardware and Deployment Context

### Current Infrastructure
- **Edge Device**: Raspberry Pi 5 + Hailo-8 accelerator (26 TOPS)
- **Development Environment**: GPUSRV with containerized training
- **Model Size Target**: <10MB (HEF format)
- **Latency Requirement**: <100ms end-to-end for real-time feedback
- **Power Budget**: <2W for edge deployment

### Integration Points
- **EdgeInfer System**: Docker Compose deployment ready
- **Feature Flag Architecture**: Safe production rollback capability
- **HTTP API**: RESTful endpoints for real-time inference
- **Session Management**: Structured data collection and analysis

## Research and Application Vision

### Immediate Applications (3-6 months)
1. **Real-time Training Feedback**: Live behavioral synchrony scoring during dog training sessions
2. **Clinical Assessment**: Automated attachment and relationship quality measurement
3. **Service Dog Matching**: Handler-dog compatibility prediction during training

### Academic Contributions (PhD Timeline)
1. **NeurIPS 2025**: Technical architecture and validation paper
2. **Current Biology**: Behavioral-dynamical coherence theoretical contribution
3. **PLOS Computational Biology**: Nonparametric behavioral state discovery

### Long-term Impact
- **Standardized Measurement**: Reproducible synchrony assessment across studies
- **Clinical Translation**: Real-time intervention capability for therapy
- **Cross-species Generalization**: Universal coordination principles
- **Open Science**: Democratized access to expert-level behavioral analysis

## Key Documents in This Context

### Design and Theory
- `Conv2d-VQ-HDP-HSMM_Architecture_Design.md` - Complete system architecture
- `Conv2d-VQ-HDP-HSMM_Implementation_Roadmap.md` - Step-by-step implementation guide
- `Unified_Theory_Computational_Behavioral_Synchrony.md` - Comprehensive theoretical framework

### Background and Foundation
- `Synchrony Dynamics - Two-Prong Improvement Strategy.md` - Algorithmic enhancement strategy
- `Behavioral_Pipeline_README.md` - Sprint management and current pipeline status
- `Enhanced_Pipeline_Implementation_Status.md` - Current model performance benchmarks
- `IMU_Behavioral_Analysis_Implementation.md` - 16-week staged development approach
- `TCN_VAE_EdgeInfer_Roadmap-2025-08-30.md` - Deployment and integration timeline
- `hailo_pipeline_project_status_and_next_actions.md` - Hardware integration status

## Success Metrics and Validation

### Technical Performance Targets
- **Model Quality**: Reconstruction MSE < 0.1, Codebook Perplexity > 100
- **Behavioral Classification**: >85% accuracy vs manual coding
- **Computational Efficiency**: <15ms inference latency on Hailo-8
- **Cluster Discovery**: 5-15 active behavioral clusters with coherent interpretation

### Scientific Validation
- **Kelso Replication**: Reproduce HKB phase transitions in coordination dynamics
- **Feldman Alignment**: Match parent-infant synchrony patterns from literature
- **Cross-species Consistency**: Stable metrics across dog/horse/cat behavioral data
- **Clinical Utility**: Detect known attachment issues with >85% sensitivity

### Deployment Readiness
- **Real-time Capability**: <100ms end-to-end latency for intervention timing
- **Edge Deployment**: Raspberry Pi 5 + Hailo-8 integration functional
- **Uncertainty Quantification**: Calibrated confidence intervals for clinical use
- **Professional Integration**: Trainer dashboard and feedback interface ready

---

This project represents the convergence of computational ethology, machine learning, and clinical applications - positioned to revolutionize how we understand and enhance human-animal relationships through real-time, scientifically grounded synchrony measurement.

**Status**: Ready for implementation with comprehensive foundation and clear roadmap to impact.