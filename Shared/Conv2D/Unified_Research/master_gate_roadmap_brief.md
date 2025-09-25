# Master Gate & Roadmap Brief (M1.3 → Causal Intervention System)

[[Master_MOC]] • [[02__MVP/README]] • [[04__Operations/README]]

---

## Purpose
Unify all active documentation (Checkpoint Bundle, Migration Guide, Technical Architecture v2, Roadmap v2) into a single execution brief. Serves as source of truth for post-M1.3 development.

---

## Current Status: ✅ M1.3 Passed
- **Model**: Conv2d → FSQ → HSMM → Classifier + Causal Rules + Calibration【45†M1_3_CHECKPOINT_BUNDLE.md】
- **Accuracy**: 86.4% (≥85% target met)
- **ECE**: 2.3% (≤3% target met)
- **Conformal Coverage**: 91% (≥90% target met)
- **Latency**: 9ms (≪100ms target)
- **HDP**: Removed (harmful, +43% accuracy after removal)
- **Causal Intervention System**: Implemented (92% precision, <1ms overhead)
- **Deployment**: Hailo-8 compiled and validated
- **Clinical Readiness**: Safety checks passed, regulatory prep ongoing

---

## Architecture v2.0 (Production)

### High-Level Flow【44†TECHNICAL_ARCHITECTURE_V2.md】
```
Conv2d Encoder (5ms) → FSQ Quantizer (1ms) → HSMM Temporal Model (2ms) → Classifier (0.5ms)
                                         ↘
                                      Causal Rules Engine (<1ms)
                                         ↘
                                   Intervention Trigger
```

### Key Decisions
1. **Finite Scalar Quantization (FSQ)** replaces VQ-VAE → deterministic, no collapse.
2. **Hidden Semi-Markov Model (HSMM)** retained for temporal dynamics.
3. **Causal Rules Engine** added for interventions (<1ms overhead).
4. **Calibration Layer** (temperature scaling + conformal prediction) ensures clinical reliability.

### Performance
- Accuracy: 86.4% top-1, 97.2% top-3
- Intervention Precision: 92%
- Latency: 9ms (p95: 9.1ms)
- Memory: 98MB (32% lower vs M1.2)
- Power: 1.4W active inference

---

## Migration Guide (Summary)【43†MIGRATION_GUIDE.md】

### Codebase Updates
- **Removed**: `models/hdp_component.py`, `models/vq_vae_model.py`, `models/vector_quantizer_ema.py`
- **Added**: `models/fsq_model.py`, `models/causal_rules_engine.py`, `calibration/calibrated_fsq_model.py`, `intervention/rule_engine.py`, `intervention/intervention_selector.py`
- **Config**: Replace `hdp/vq` blocks with `fsq/causal/calibration`
- **Train Scripts**: Swap HDP/VQ imports → FSQ + Causal + Calibration

### Database Schema
- **Added**: `interventions` + `calibration_metrics` tables
- **Removed**: `hdp_topics`, `vq_codebook_usage`

### Performance After Migration
| Metric        | M1.2 (HDP) | M1.3 (Causal) | Gain |
|---------------|------------|---------------|------|
| Accuracy      | 78.1%      | 86.4%         | +8.3% |
| Latency       | 25ms       | 9ms           | -64% |
| Memory        | 145MB      | 98MB          | -32% |
| Power         | 2.3W       | 1.4W          | -39% |

---

## Roadmap v2 (12 Weeks)【46†causal_intervention_roadmap.md】

### Phase 1 (Weeks 1–2): ✅ M1.3 Sprint
- HDP removal complete
- Calibration integrated (ECE 2.3%, coverage 91%)
- Ensemble + augmentation lifted accuracy to 86.4%
- M1.3 Gate passed ✅

### Phase 2 (Weeks 3–5): Causal Pattern Infrastructure
- [ ] **Week 3**: Implement `CausalPatternMiner` + validation system
- [ ] **Week 4**: Build `EdgeRuleEngine` (<1ms, buffer=100, conflict resolution)
- [ ] **Week 5**: Develop `CausalTCN` (predictive interventions >90% acc)
- **Milestone**: Causal Detection System Ready

### Phase 3 (Weeks 6–8): Clinical Mapping
- Define intervention taxonomy (levels 0–3)
- Clinical validation protocol (safety, A/B tests)
- InterventionSelector (bandit-based, personalization)
- End-to-end pipeline test + simulation
- **Milestone**: Intervention System Validated

### Phase 4 (Weeks 9–12): Deployment
- Hailo-8 optimization (quantization, power management)
- Edge monitoring & telemetry
- Cloud retraining & analytics pipeline
- OTA update system
- Beta test (10 users), production launch with gradual rollout
- **Milestone**: Production Launch Complete

---

## Risks & Mitigations
- **Rule conflicts** → resolved via priority system
- **Intervention fatigue** → cooldown periods implemented
- **Pattern drift** → continuous learning planned
- **Clinical safety** → conservative thresholds, IRB protocol in place

---

## Gates & Decision Points
- **Week 2 (M1.3 Gate)**: ✅ Passed
- **Week 5 (Causal Detection)**: Pending
- **Week 8 (Clinical Validation)**: Pending
- **Week 11 (Beta Results)**: Pending

---

## Next Immediate Sprint (Weeks 3–5)
**Focus**: Build causal infrastructure
- Deliverables: `pattern_miner.py`, `pattern_validator.py`, `edge_rule_engine.py`, `rule_compiler.py`, `causal_tcn.py`
- Validation: <1ms rule engine, >90% causal TCN accuracy
- Gate: **Causal Detection System Ready**

---

## Archival Note
The following are now historical references and moved to `/06__Archive/`:
- VQ collapse analysis
- Conv2d-VQ-HDP-HSMM quick reference, roadmap, architecture design
- Unified Theory of Computational Behavioral Synchrony

These inform theoretical foundation but are **not active implementation docs**.

---

**Document Version:** 1.0  
**Last Updated:** Sept 2025  
**Next Review:** Week 5 (Causal Detection Gate)

