# ADR-001: Drop HDP → Post-hoc Clustering

**Status**: Accepted  
**Priority**: High  
**Project**: Conv2d  
**Component**: decision-record  
**Created**: 2025-09-25  
**Updated**: 2025-09-25  
**Version**: v1.0  
**Authors**: Development Team  

---

## Context

The original Conv2d-VQ-HDP-HSMM pipeline used Hierarchical Dirichlet Process (HDP) after FSQ/VQ codes to cluster behaviors. HDP was attractive because:
- Non-parametric: doesn't require a fixed number of clusters
- Theoretically elegant: automatic discovery of behavioral states
- Hierarchical: natural multi-level behavior organization

However, in practice we observed severe issues:

1. **Accuracy Degradation**: 52% drop when HDP was integrated (100% → 48.3%)
2. **Cluster Fragmentation**: Behaviors split into 50+ unstable micro-states
3. **Irreproducibility**: Label maps varied across runs despite fixed seeds
4. **Gradient Flow**: Broken between FSQ and HDP components
5. **HSMM Instability**: Temporal modeling suffered due to unstable inputs

## Decision

We **removed HDP** and adopted a **post-hoc deterministic clustering pipeline**:

### New Pipeline Architecture
```
FSQ (64 codes) → K-means/GMM → Temporal Smoothing → HSMM
```

### Implementation Details

1. **FSQ with Reduced Codebook**
   - Levels: [4, 4, 4] = 64 total codes
   - Target: >80% utilization (was 7.4% with 512 codes)
   - Vectorized operations for efficiency

2. **Deterministic Clustering**
   ```python
   # Fixed K-means with BIC/AIC selection
   best_k = select_k_bic(fsq_codes, k_range=[8, 16, 24, 32])
   clusters = KMeans(n_clusters=best_k, random_state=42)
   motifs = clusters.fit_predict(fsq_codes)
   ```

3. **Rare Cluster Handling**
   - Merge clusters with <1% of samples
   - Drop clusters with <0.1% support
   - Maintain cluster→label mapping

4. **Temporal Smoothing**
   - Median filter (window=5) for noise reduction
   - Hysteresis thresholding for state transitions
   - Minimum dwell time enforcement (≥3 frames)

## Consequences

### Pros ✅
- **Stability**: Reproducible motifs across runs
- **Performance**: 78.12% accuracy restored on quadruped task
- **Simplicity**: Easier debugging and QA
- **Edge-friendly**: Deterministic operations suitable for Hailo-8
- **Interpretability**: Fixed motif definitions

### Cons ⚠️
- **Less Flexible**: Cannot discover rare behaviors automatically
- **Parameter Tuning**: Requires K selection
- **Over-merging Risk**: May combine distinct but similar behaviors

### Mitigation Strategies
1. Use BIC/AIC for automatic K selection
2. Maintain FSQ code diversity through proper levels
3. Implement merge-then-split refinement post-hoc
4. Preserve raw FSQ codes for future re-clustering

## Alternatives Considered

### 1. Tuning HDP Priors
- **Attempted**: Alpha/gamma hyperparameter sweeps
- **Result**: Improved slightly but still unstable
- **Rejected**: Fundamental gradient flow issue remained

### 2. Sticky HDP-HMM
- **Attempted**: Added self-transition bias
- **Result**: Better duration modeling but fragmentation persisted
- **Rejected**: Didn't address core clustering instability

### 3. Larger Codebooks
- **Attempted**: 2048 codes for more granularity
- **Result**: Worse sparsity (3.1% utilization), poor calibration
- **Rejected**: Exacerbated the problem

### 4. Direct VQ without HDP
- **Attempted**: Use VQ codes directly as behaviors
- **Result**: Too granular, no semantic grouping
- **Rejected**: Lost behavioral abstraction

## Implementation Status

### Completed ✅
- HDP components moved to `archive/legacy_models/`
- FSQ optimized to 64 codes with vectorized operations
- Validation on real PAMAP2 data confirms improvement

### In Progress 🚧
- Post-hoc clustering pipeline implementation
- Temporal smoothing module development
- Integration tests for new pipeline

### Future Work 📋
- Investigate learnable clustering (DEC, JULE)
- Explore attention-based motif discovery
- Consider hierarchical clustering for multi-scale behaviors

## Code References

- **Original HDP**: `archive/legacy_models/hdp_components.py`
- **Optimized FSQ**: `models/conv2d_fsq_optimized.py:46-148`
- **Clustering Pipeline**: `models/fsq_clustering.py` (to be implemented)
- **Validation Results**: `scripts/validate_fsq_real_data.py:89-156`

## Review Notes

Per D1 Gate Review (2025-09-25):
> "HDP integration causes 52% accuracy drop - gradient flow broken between FSQ and HDP. Debug gradient propagation or remove HDP until fixed."

Decision to remove was validated by both:
1. Synchrony Advisor Committee Review
2. PhD-level Technical Review

## References

1. Teh et al. (2006) - "Hierarchical Dirichlet Processes"
2. Fox et al. (2011) - "Sticky HDP-HMM"
3. Van Haaren et al. (2019) - "Deep Embedded Clustering for HAR"
4. D1 Gate Review Documentation - `docs/design/D1_CONSOLIDATED_REVIEWS.md`

---

*This ADR documents the critical architectural decision to remove HDP in favor of deterministic post-hoc clustering, addressing the primary blocker identified in the D1 Design Gate review.*