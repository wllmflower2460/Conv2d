# VQ Codebook Collapse: Deep Analysis & Findings

## Executive Summary
The VQ codebook is collapsing to use only 1 code (perplexity=1.03) even during training, indicating a fundamental issue beyond the eval mode problem. The eval mode fix from the Claude artifact is necessary but not sufficient.

## Test Results

### Quick Test (`test_vq_eval_fix.py`)
- **Training**: Immediately collapses from 44.47 to 1.03 perplexity
- **Standard Eval**: Stays at 1.03 (already collapsed)
- **Fixed Eval** (VQ in train mode): Still 1.03 (no improvement)
- **Conclusion**: The collapse happens during training itself

### Previous Ablation Results
- **Without VQ**: 82-89% accuracy
- **With VQ**: 22.9% accuracy (random chance)
- **Training behavior**: Shows good perplexity (50-160) during monitoring
- **Eval behavior**: Collapses to ~1-2 perplexity

## Root Cause Analysis

### 1. Immediate Training Collapse
```
Step   0: Perplexity=44.47, Usage=0.297  # Good initial diversity
Step  10: Perplexity=1.03, Usage=0.004   # Immediate collapse
```
The VQ collapses within 10 training steps, suggesting:
- Gradient issues preventing proper learning
- Loss weighting problems
- Initialization not holding

### 2. Dimension Mismatch Issues
Our encoder outputs 128D but VQ expects 64D. Current reshaping:
```python
features.view(batch_size, 64, 2)  # Splits 128 into 64x2
```
This artificial split may not preserve meaningful features.

### 3. EMA Update Problems
The VectorQuantizerEMA2D_Stable has EMA updates but they may be:
- Too aggressive (decay=0.95 might be wrong)
- Not receiving proper gradients
- Being overwhelmed by commitment loss

## Solutions to Implement

### Priority 1: Fix Training Collapse
1. **Reduce commitment cost**: β=0.4 → 0.1 or 0.01
2. **Increase VQ loss weight**: Current 0.1x might be too low
3. **Remove L2 normalization**: May be constraining too much
4. **Slower EMA decay**: 0.95 → 0.99 for stability

### Priority 2: Architecture Alignment
1. **Add projection layer**: 128D → 64D learnable projection
2. **Or increase code_dim**: 64 → 128 to match encoder
3. **Remove artificial reshaping**: Keep natural dimensions

### Priority 3: Advanced Techniques
1. **Codebook reset on collapse**: If perplexity < 5, reinitialize
2. **Gradient clipping**: Prevent large updates
3. **Warmup period**: Freeze encoder for first 1000 steps
4. **Alternative VQ**: Try VQ-VAE-2 or FSQ (Finite Scalar Quantization)

## Immediate Next Steps

### Option A: Quick Parameter Tuning
```python
VectorQuantizerEMA2D_Stable(
    num_codes=512,
    code_dim=128,  # Match encoder
    decay=0.99,     # Slower EMA
    commitment_cost=0.01,  # Much lower
    l2_normalize_input=False,  # Remove constraint
    restart_dead_codes=True,
    dead_code_threshold=0.01  # More aggressive restart
)
```

### Option B: Add Projection Layer
```python
class AblationModel(nn.Module):
    def __init__(self, config):
        # ...
        if config.use_vq:
            self.vq_proj = nn.Linear(128, 64)  # Project to VQ dim
            self.vq = VectorQuantizerEMA2D_Stable(...)
```

### Option C: Implement Fallback System
From `local_handoff/vq_recovery/fallback_tokenizer.py`:
- Detect collapse (perplexity < 1.5)
- Switch to k-means clustering
- Continue training/inference

## Key Insight from External Research
The eval mode fix is correct but addresses a different problem:
- **Problem 1**: VQ collapses during training (our current issue)
- **Problem 2**: VQ collapses during eval even if training was good (what the fix solves)

We need to solve Problem 1 first, then apply the eval mode fix for Problem 2.

## Recommended Immediate Action
1. Test with much lower commitment cost (0.01)
2. Match dimensions (code_dim=128)
3. Remove L2 normalization
4. If still failing, implement k-means fallback

The VQ collapse is more fundamental than initially thought and requires addressing the training dynamics before the eval mode fix becomes relevant.