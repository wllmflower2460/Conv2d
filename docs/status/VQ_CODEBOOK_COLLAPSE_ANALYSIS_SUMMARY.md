# VQ Codebook Collapse: Setup, Fixes & Training/Eval Discrepancy

## Current VQ Architecture (`models/VectorQuantizerEMA2D_Stable.py`)
- **Codebook**: 512 codes × 64 dimensions, EMA updates with decay=0.95
- **Gradient Flow**: Straight-through estimator: `z_q = z + (z_q - z).detach()`
- **Loss Components**: Commitment loss (β=0.4) + Codebook loss
- **Dead Code Handling**: Reinitializes codes with <0.001 usage from random batch samples
- **L2 Normalization**: Applied to encoder outputs before quantization

## Implemented Fixes in Ablation Framework

### 1. Data-Driven Initialization (`initialize_codebook_from_data`)
```python
# Collect encoder outputs from first batch
embeddings = model.encoder(x).detach()
# Initialize codebook with random subset
indices = torch.randperm(embeddings.size(0))[:model.vq.num_codes]
model.vq.embedding.data = embeddings[indices]
```

### 2. VQ Loss Weighting with Warmup
```python
vq_weight = min(1.0, epoch / 10)  # 10-epoch warmup
loss = classification_loss + vq_weight * 0.1 * vq_loss["vq"]
```

### 3. Training Configuration Adjustments
- Reduced batch size: 64 → 32 (more gradient updates)
- Extended epochs: 100 → 150 (more time to learn codes)
- Lower learning rate: 1e-3 → 5e-4 (stable convergence)
- EMA decay: 0.99 → 0.95 (faster codebook updates)

### 4. Diagnostic Monitoring (Every 10 Epochs)
```python
usage = model.vq.ema_cluster_size / model.vq.ema_cluster_size.sum()
active_codes = (usage > 0.001).sum().item()
perplexity = torch.exp(-torch.sum(usage * torch.log(usage + 1e-10)))
```

## Critical Finding: Training Success vs Evaluation Collapse

### Training Behavior (Working)
```
vq_hdp @ Epoch 40:  Active codes: 183/512, Perplexity: 157.2
vq_hdp @ Epoch 90:  Active codes: 174/512, Perplexity: 150.8  
vq_hdp @ Epoch 140: Active codes: 136/512, Perplexity: 88.9
```
- Successfully learns diverse codebook usage
- Maintains 100-180 active codes throughout training
- Perplexity stays in healthy 50-160 range

### Evaluation Behavior (Failing)
```
vq_only:     Perplexity: 1.06, Accuracy: 22.9%
vq_hdp:      Perplexity: 1.06, Accuracy: 22.9%
vq_hsmm:     Perplexity: 2.09, Accuracy: 22.9%
vq_hdp_hsmm: Perplexity: 2.07, Accuracy: 22.9%
```
- Complete collapse to 1-2 codes during evaluation
- All VQ variants show identical 22.9% accuracy (random chance for 10 classes)
- Non-VQ variants achieve 82-89% accuracy

## Hypothesis: Why Training Works but Evaluation Fails

### Potential Causes:
1. **EMA Statistics Not Updated During Eval**: VQ uses EMA-updated codebook which may not generalize
2. **Distribution Shift**: Training augmentation creates diversity not present in clean eval data
3. **Overfitting to Training Batches**: Codebook specialized to training data distribution
4. **Missing eval() Mode Handling**: VQ may need special handling when model.eval() is called

### Key Diagnostic Questions:
1. Does codebook collapse happen immediately at eval start or gradually?
2. Are the same 1-2 codes used across all eval samples?
3. Do encoder features change distribution between train and eval?
4. Is the straight-through gradient properly detached during eval?

## Next Steps for Deep Dive:
1. Add verbose logging specifically during evaluation phase
2. Track per-batch perplexity during evaluation
3. Visualize encoder feature distributions train vs eval
4. Test with model.train() mode during evaluation (diagnostic only)
5. Check if freezing encoder for initial epochs helps codebook stability