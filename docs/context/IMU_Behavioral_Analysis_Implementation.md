# Canine IMU Behavioral Analysis - Staged Implementation Plan

## 🎯 Project Success Framework

### Core Principles
1. **Work in Vertical Slices** - Get end-to-end working (even if simple) before optimizing
2. **Fail Fast** - Test critical assumptions early (especially hardware constraints)
3. **Document as You Go** - Keep a lab notebook with what worked/didn't
4. **Version Everything** - Data, models, configurations
5. **Set Time Boxes** - Don't perfect Stage 1 when Stage 2 is waiting

---

## 📊 [[Canine_Motifs_Obsidian_Vault/CDD Tasks/Stage 0 Data/0-verview|0-verview]] Stage 0: Foundation & Feasibility (Week 1-2)
**Goal**: Validate core assumptions and set up infrastructure

### Tasks
- [x] **Hardware Check**
  - [x] Verify Hailo-8 SDK installation and Raspberry Pi 5 setup ✅ 2025-08-17
  - [x] Run Hailo example models to confirm pipeline works ✅ 2025-08-17
  - [x] Test IMU sensor data collection at target sample rates ✅ 2025-08-17

- [x] **Data Inventory**
  - [x] Download PLOS One Working Dog dataset
  - [x] Download PAMAP2 for pretraining
  - [x] Create simple data loader that can read and visualize IMU streams
  - [x] Verify you can collect your own IMU data if needed

- [x] **Development Environment**
  - [x] Set up Python environment with PyTorch, NumPy, scikit-learn
  - [x] Install Hailo Dataflow Compiler (DFC)
  - [x] Create project structure with clear folders for data/models/configs
  - [x] Set up Git repository

**Gate Criteria**: Can load IMU data, visualize it, and run a simple model on Hailo

---

## 🔨 Stage 1: Baseline System (Week 3-5)
**Goal**: Build simplest working version of the complete pipeline

### Tasks
- [x]  **[[Simple TCN Encoder (Non-VAE)]]**
  - [x] Implement basic TCN with 3-4 layers (no VAE yet)
  - [x] Fixed window size (200 samples)
  - [x] Train as classifier on PAMAP2 activities (supervised)
  - [ ] Verify it learns something meaningful

- [ ] **Basic Clustering**
  - [ ] Extract features using trained TCN
  - [ ] Apply simple K-means clustering (K=5)
  - [ ] Visualize with t-SNE/UMAP
  - [ ] Create basic ethogram visualization

- [ ] **Hailo Deployment V1**
  - [x] Export TCN to ONNX
  - [x] Compile with Hailo DFC (may need troubleshooting)
  - [ ] Run inference on Pi with static test data
  - [ ] Measure latency and confirm it works

**Gate Criteria**: Can train model, cluster behaviors, and run on Hailo (even if not real-time)

---

## 🚀 Stage 2: VAE Implementation (Week 6-8)
**Goal**: Add VAE components and improve representation learning

### Tasks
- [ ] **TCN-VAE Architecture**
  - [ ] Add VAE components (μ, σ, reparameterization)
  - [ ] Implement ELBO loss (reconstruction + KL)
  - [ ] Train on PAMAP2 first, then canine data
  - [ ] Monitor for posterior collapse

- [ ] **Triplet Loss Addition**
  - [ ] Implement temporal triplet mining
  - [ ] Add to loss function with warm-up schedule
  - [ ] Tune β and λ hyperparameters
  - [ ] Verify latent space structure improves

- [ ] **HMM Integration**
  - [ ] Implement basic HMM on latent codes
  - [ ] Sweep K from 8-14, use BIC/AIC for selection
  - [ ] Add transition probability constraints
  - [ ] Compare to simple K-means baseline

**Gate Criteria**: VAE produces structured latent space, HMM finds meaningful states

---

## 🔄 Stage 3: Real-Time Pipeline (Week 9-11)
**Goal**: Achieve real-time streaming inference

### Tasks
- [ ] **Streaming Data Handler**
  - [ ] Implement circular buffer for IMU data
  - [ ] Sliding window with 50% overlap
  - [ ] Real-time preprocessing (normalization)
  - [ ] Handle dropped samples gracefully

- [ ] **Online HMM**
  - [ ] Forward algorithm implementation
  - [ ] Add dwell time constraints (0.3s)
  - [ ] Confidence thresholding (0.6)
  - [ ] Smoothing with lag

- [ ] **Causal TCN Training**
  - [ ] Modify architecture for causal convolutions
  - [ ] Teacher-student distillation from non-causal
  - [ ] Verify minimal accuracy loss
  - [ ] Test receptive field calculations

- [ ] **System Integration**
  - [ ] Connect IMU → Buffer → Hailo → HMM → Output
  - [ ] Add watchdog timer for stalls
  - [ ] Implement fallback modes
  - [ ] Create monitoring dashboard

**Gate Criteria**: System runs continuously at target FPS with <2.5s latency

---

## 🎨 Stage 4: Optimization & Refinement (Week 12-14)
**Goal**: Improve accuracy and robustness

### Tasks
- [ ] **Model Optimization**
  - [ ] Quantization-aware training (QAT)
  - [ ] Hyperparameter optimization
  - [ ] Ablation studies (SE blocks, depth, etc.)
  - [ ] Cross-validation on different dogs

- [ ] **Advanced Training**
  - [ ] Data augmentation strategies
  - [ ] Semi-hard triplet mining
  - [ ] Pretrain on TartanIMU if available
  - [ ] Fine-tune on combined datasets

- [ ] **Robustness Testing**
  - [ ] Test with sensor dropout
  - [ ] Axis misalignment simulation
  - [ ] Different mounting positions
  - [ ] Long-duration stability tests

**Gate Criteria**: Model generalizes across dogs, handles edge cases

---

## 📱 Stage 5: Production Features (Week 15-16)
**Goal**: Add user-facing features and polish

### Tasks
- [ ] **Visualization Suite**
  - [ ] Real-time UMAP with moving dot
  - [ ] IMU trace + reconstruction view
  - [ ] Behavioral ethogram timeline
  - [ ] Session comparison tools

- [ ] **Dual-Mode System**
  - [ ] Implement low-latency head (155ms)
  - [ ] High-fidelity head (2.5s)
  - [ ] Late fusion strategy
  - [ ] Mode switching logic

- [ ] **Deployment Package**
  - [ ] Configuration management
  - [ ] Model versioning system
  - [ ] Performance monitoring
  - [ ] User documentation

**Gate Criteria**: System is deployable and usable by others

---

## 🛠️ Critical Path Items (Do These First!)

1. **Week 1**: Confirm Hailo-8 can compile a simple TCN
2. **Week 2**: Verify you can stream IMU data at 100Hz
3. **Week 3**: Get ANY model producing behavioral labels
4. **Week 4**: Achieve first Hailo deployment (even if slow)

---

## 📝 Success Maximizers

### Daily Practices
- **Morning Planning** (15 min): Review today's specific task
- **Evening Documentation** (15 min): Log what worked/failed
- **Weekly Demo**: Show someone your progress (even if it's broken)

### Tools to Use
- **Weights & Biases / MLflow**: Track experiments
- **Jupyter Notebooks**: For exploration (convert to scripts later)
- **pytest**: Write tests for data pipeline early
- **Docker**: Containerize your environment

### When You're Stuck
1. **Simplify**: Can you make the problem 10x simpler?
2. **Visualize**: Plot everything - data, gradients, latents
3. **Baseline**: Compare to simplest possible approach
4. **Ask**: Post specific errors to Hailo forums early

### Red Flags to Watch
- 🚨 Posterior collapse in VAE (z becomes meaningless)
- 🚨 Hailo compilation failures (address immediately)
- 🚨 Latency exceeding 2.5s (architectural problem)
- 🚨 HMM states fragmenting (K too high or features poor)

---

## 📊 Progress Tracking

### Weekly Milestones
- [ ] Week 2: Data pipeline complete
- [ ] Week 4: First model trained
- [ ] Week 6: Hailo inference working
- [ ] Week 8: VAE producing good latents
- [ ] Week 10: Real-time streaming achieved
- [ ] Week 12: HMM behavioral states stable
- [ ] Week 14: Cross-dog generalization proven
- [ ] Week 16: Production-ready system

### Success Metrics
- **Technical**: <2.5s latency, >10 FPS, <5W power
- **Scientific**: 8-14 distinct behavioral motifs, >80% temporal consistency
- **Practical**: Runs 24 hours without intervention

---

## 💡 Escape Hatches

If you get severely blocked:

1. **Hailo Won't Compile TCN?** 
   → Fall back to regular CNN with larger kernel
   → Try depth-wise separable convolutions
   → Consider two-stage: CNN on Hailo, RNN on CPU

2. **VAE Won't Learn?**
   → Start with regular autoencoder
   → Use pre-trained features from PAMAP2
   → Reduce latent dimension to 16

3. **Real-time Too Slow?**
   → Reduce sampling rate to 50Hz
   → Smaller model (3 layers)
   → Increase buffer stride (more skip)

4. **No Meaningful Behaviors?**
   → Collect more diverse data
   → Try supervised pre-training
   → Use simpler features (statistics over windows)

Remember: **A working simple system is better than a perfect complex one that doesn't exist!**