# Execution Brief: M1.3 → Causal Intervention System (12-Week Plan)

[[Master_MOC]] • [[02__MVP/README]] • [[04__Operations/README]]

---

## Purpose
Consolidated roadmap aligning the **M1.2 checkpoint**, **committee report**, and **causal intervention system roadmap**. Provides execution guidance from **M1.3 requirements** through full **12-week deployment**.

---

## Current Status (M1.2)
- ✅ **FSQ adoption**: Fixed VQ collapse (stable, deterministic).
- ✅ **Ablation evidence**: FSQ+HSMM optimal, HDP harmful.
- ⚠️ **Accuracy plateau**: ~78% on real data (target ≥85%).
- ❌ **Calibration missing**: No ECE, conformal, temp scaling.
- 🟡 **Committee vote**: Conditional Pass to M1.3【21†M1_2_CHECKPOINT_BUNDLE.md】【22†M1_2_COMMITTEE_REPORT.md】.

---

## Phase 1 (Weeks 1–2): M1.3 Sprint
**Goal: Clear M1.3 Gate.**

**Tasks**
- [ ] Remove HDP from pipeline (simplify Conv2d → FSQ → HSMM → Classifier).
- [ ] Integrate calibration.py → validate ECE ≤3%, generate reliability diagrams.
- [ ] Add conformal predictor with 90% coverage.
- [ ] Implement temperature scaling.
- [ ] Build augmentation pipeline (time-warp, jitter, rotation).
- [ ] Train ensemble of FSQ models with varied quantization levels.
- [ ] Validate ≥85% accuracy, <100 ms latency (Hailo-8 profiling).
- [ ] Prepare **M1.3 Gate Package** (documentation, metrics, validation logs).

**Acceptance Criteria (M1.3 Gate)**
- ✅ ECE ≤3%
- ✅ ≥85% accuracy on real behavioral data
- ✅ Latency <100ms (p95, Hailo-8)
- ✅ Conformal prediction (90% coverage)

---

## Phase 2 (Weeks 3–5): Causal Pattern Infrastructure
**Goal: Replace HDP with real-time causal inference.**

**Tasks**
- Pattern mining framework (discover predictive sequences).
- Pattern validation (precision, recall, statistical tests).
- Lightweight edge rule engine (<1 ms evaluation, circular buffer).
- Rule compiler/optimizer → deploy rules to Hailo-8.
- Temporal Causal Network (TCN) with causal masking → predict interventions in <10 ms.

**Milestone: Causal Detection System Ready.**

---

## Phase 3 (Weeks 6–8): Clinical Intervention Mapping
**Goal: Define and validate therapeutic interventions.**

**Tasks**
- Build intervention taxonomy (levels 0–3, modalities, latencies).
- Design clinical validation protocol (safety, A/B testing).
- Intervention selection algorithm (multi-armed bandit, personalization).
- End-to-end pipeline test (FSQ → pattern → intervention).
- Clinical simulations and system review.

**Milestone: Intervention System Validated.**

---

## Phase 4 (Weeks 9–12): Production Deployment
**Goal: Edge optimization, cloud support, beta test, launch.**

**Tasks**
- Quantize and optimize TCN for Hailo-8 (INT8, memory/power optimized).
- Implement monitoring + telemetry.
- Cloud analytics pipeline (mining, retraining, dashboards).
- OTA update system (rules, model versioning, rollback).
- Controlled beta (10 users, feedback, performance monitoring).
- Production readiness: safety audit, load testing, gradual rollout.
- Launch retrospective + optimization roadmap.

**Milestone: Production Launch Complete.**

---

## Risks & Mitigations
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Calibration not ready | Medium | High | Dedicated sprint, gate criteria strict |
| Accuracy stuck at 78% | Medium | High | Augmentation + ensemble, semi/active learning |
| Latency >100ms | Low | Medium | FSQ efficient, early profiling on Hailo-8 |
| False positive interventions | Medium | Medium | Conservative thresholds, cooldown periods |
| Clinical rejection | Low | High | Calibration + safety dashboards |

---

## Go/No-Go Gates
- **Week 2 (M1.3 Gate)** → must meet calibration + accuracy + latency.
- **Week 5 (Causal Ready)** → pattern mining + TCN trained.
- **Week 8 (Clinical Validated)** → safety + effectiveness confirmed.
- **Week 11 (Beta Results)** → positive user feedback, no safety flags.

---

## Review Cadence
- **Weekly reviews** with metrics.
- **Gate reviews** at weeks 2, 5, 8, 11.
- **Final review** at week 12 before launch.

---

## Next Step (Immediate)
Start **Phase 1 sprint**:
- Owner: ML Engineer (calibration + augmentation).
- Deliverables: `calibrated_fsq_model.py`, `augmentation_pipeline.py`, `ensemble_fsq_model.py`, `M1_3_GATE_PACKAGE.md`.

---

**Document Version:** 1.0  
**Last Updated:** Sept 2025  
**Next Review:** Week 1 (post-calibration integration)

