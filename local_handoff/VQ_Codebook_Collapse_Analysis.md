# VQ Codebook Collapse Analysis and Solutions

## Problem Summary
Ablation study results show complete VQ (Vector Quantization) failure:
- Without VQ: 88-97% accuracy 
- With VQ: 5-23% accuracy (worse than random)
- Perplexity: 1.0-1.1 (only 1 code used out of 256/512)
- Complete codebook collapse despite multiple attempted fixes

## Root Causes Identified

### 1. Missing Straight-Through Estimator
**Problem**: Gradient flow blocked through VQ layer
**Solution**: 
```python
z_q = z_e + (z_q - z_e).detach()  # Critical for gradient flow
```

### 2. Incorrect Loss Weighting  
**Problem**: VQ loss added directly without scaling
**Current Code**: `loss = loss + out["vq_loss"]["vq"]`
**Solution**: `loss = loss + 0.1 * out["vq_loss"]["vq"]`

### 3. EMA Initialization Issue
**Problem**: With decay=0.99, needs ~100 updates to change significantly
- Only 50-100 epochs with batch_size=32 insufficient for EMA convergence
- Codebook never properly initialized from data distribution

### 4. Dimensional Averaging Masking Collapse
**Problem**: Classifier uses averaged features, hiding codebook issues during training

## Implementation Fixes

### Fix 1: Proper Gradient Flow
```python
class VectorQuantizerEMA2D(nn.Module):
    def forward(self, z_e):
        # Quantization logic...
        
        # CRITICAL: Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        
        # Commitment loss uses undetached z_e
        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        
        return z_q, losses, info
```

### Fix 2: VQ Loss Warmup
```python
def train_model(self, model, config, epochs=50):
    for epoch in range(epochs):
        vq_weight = min(1.0, epoch / 10)  # Warmup over 10 epochs
        
        # In training loop:
        if config.use_vq:
            loss = loss + vq_weight * 0.1 * out["vq_loss"]["vq"]
```

### Fix 3: Data-Driven Initialization
```python
def initialize_codebook_from_data(model, data_loader):
    """Initialize VQ codebook using actual encoder outputs"""
    if not hasattr(model, 'vq') or model.vq is None:
        return
    
    with torch.no_grad():
        embeddings = []
        for x, _ in data_loader:
            z_e = model.encoder(x.to(model.device))
            z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, z_e.size(1))
            embeddings.append(z_e_flat)
            if len(embeddings) * z_e_flat.size(0) >= model.vq.num_codes * 10:
                break
        
        embeddings = torch.cat(embeddings, dim=0)
        # K-means or random sampling initialization
        indices = torch.randperm(embeddings.size(0))[:model.vq.num_codes]
        model.vq.embedding.data = embeddings[indices]
        model.vq.ema_cluster_size.data.fill_(1.0)
```

### Fix 4: Dynamic Dead Code Reinitialization
```python
def reinit_dead_codes(model, z_e_flat, threshold=0.01):
    """Reinitialize codes with very low usage during training"""
    if not hasattr(model, 'vq') or model.vq is None:
        return
    
    usage = model.vq.ema_cluster_size / model.vq.ema_cluster_size.sum()
    dead_codes = (usage < threshold).nonzero(as_tuple=True)[0]
    
    if len(dead_codes) > 0:
        with torch.no_grad():
            random_indices = torch.randint(0, z_e_flat.size(0), (len(dead_codes),))
            model.vq.embedding[dead_codes] = z_e_flat[random_indices] + 0.02 * torch.randn_like(z_e_flat[random_indices])
```

## Diagnostic Monitoring
Add to training loop after each epoch:
```python
if config.use_vq:
    with torch.no_grad():
        usage = model.vq.ema_cluster_size / model.vq.ema_cluster_size.sum()
        active_codes = (usage > 0.001).sum().item()
        print(f"Active codes: {active_codes}/{model.vq.num_codes}, "
              f"Max usage: {usage.max():.3f}, "
              f"Perplexity: {torch.exp(-torch.sum(usage * torch.log(usage + 1e-10))):.1f}")
```

## Priority Implementation Order
1. **First**: Straight-through gradient fix (most critical)
2. **Second**: Proper loss weighting (0.1 * vq_loss)
3. **Third**: Data-driven initialization
4. **Fourth**: Warmup schedule
5. **Optional**: Dead code reinitialization if still issues

## Expected Results After Fixes
- Perplexity should be 50-200 (using 50-200 codes actively)
- Accuracy should match or exceed non-VQ baseline (>85%)
- Codebook usage should stabilize around 40-80% of codes
- ECE should remain low (<0.05)

## Additional Considerations
- May need to increase training epochs to 200+ for EMA stabilization
- Consider reducing decay rate to 0.95 for faster adaptation
- Monitor gradient norms through VQ layer to ensure flow
- Test with smaller batch sizes (16) for more frequent updates

## Related Files
- [[conv2d_vq_hdp_hsmm.py]] - Main model implementation
- [[conv2d_vq_model.py]] - VQ encoder/decoder components  
- [[Ablation_Framework_Upgraded_backup.py]] - Ablation testing framework
- [[ablation_report.md]] - Latest ablation results
- [[ablation_results.json]] - Raw ablation data

---
*Created: September 2025*
*Context: Conv2d-VQ-HDP-HSMM ablation study debugging*
*Status: Critical bug - VQ codebook collapse preventing model training*