# Entropy & Marginals Module — Synchrony System
**Updated:** 2025-09-19  
**Purpose:** Keep uncertainty front-and-center while coding. Drop-in formulas, code stubs, API fields, and UI rules for discrete (states) + continuous (phase) fusion.

---

## 1) Mental Model (where it lives)
```
Sensors -> Preprocess ->
  |- Discrete Path (HDP-HSMM) -> p(z_t) -> H_state
  |- Continuous Path (Phase/Order Params) -> p(phi_t) -> H_phase
                 -> Joint p(z_t, phi_t) -> MI(Z;Phi)
Fusion -> SynchronyOutput { score, PI, risk, entropies, MI, marginals }
UI -> 'Transparent by default' (show intervals, entropy chips, drift)
```

---

## 2) Core Definitions (copy-paste)
- Shannon entropy (discrete):  H(Z_t) = - sum_i p(z_t=i) log p(z_t=i)
- Entropy (binned phase):  H(Phi_t) = - sum_j p(phi_t in B_j) log p(phi_t in B_j)
- Joint entropy:  H(Z,Phi) = - sum_{i,j} p(i,B_j) log p(i,B_j)
- Mutual information:  I(Z;Phi) = H(Z) + H(Phi) - H(Z,Phi)
- Normalized entropy:  Hbar = H / log K   (map to [0,1] for K states/bins)
- Circular concentration (proxy): resultant length R, precision kappa_hat (von Mises approx)

---

## 3) Minimal Numpy Stubs
```python
import numpy as np

EPS = 1e-12

def entropy(p):
    p = np.clip(np.asarray(p, float), EPS, 1.0)
    p /= p.sum()
    return -np.sum(p * np.log(p))

def normalized_entropy(p):
    H = entropy(p)
    K = len(np.asarray(p))
    return H / np.log(K)

def joint_entropy(joint):
    J = np.clip(np.asarray(joint, float), EPS, 1.0)
    J /= J.sum()
    return -np.sum(J * np.log(J))

def mutual_information(joint):
    J = np.clip(np.asarray(joint, float), EPS, 1.0); J /= J.sum()
    pz = J.sum(axis=1)   # states
    pphi = J.sum(axis=0) # phase bins
    return entropy(pz) + entropy(pphi) - joint_entropy(J)

def circular_confidence(angles_rad):
    angles = np.asarray(angles_rad, float)
    C, S = np.cos(angles).sum(), np.sin(angles).sum()
    R = np.sqrt(C*C + S*S) / max(len(angles), 1)
    kappa_hat = (R*(2 - R**2)) / (1 - R**2 + EPS)  # fast approx
    return dict(R=R, kappa_hat=float(kappa_hat), circ_var=1-R)
```

---

## 4) Fusion Rules (programmer-ready)
- Precision-weighted fuse discrete/continuous scores:
```python
score = (w_d * score_disc + w_c * score_cont) / (w_d + w_c + EPS)
# Suggest w_d = 1 / Var_disc,  w_c = 1 / Var_cont, or map kappa_hat -> precision for phase
```
- Confidence class from interval width or entropy:
  - green: PI_90_width <= 0.20 or Hbar <= 0.3
  - yellow: 0.20–0.35 or 0.3–0.6
  - red: >0.35 or >0.6

---

## 5) API Additions (SynchronyOutput)
```json
{
  "entropy": {
    "state_entropy": 0.22,
    "phase_entropy": 0.18,
    "joint_entropy": 1.72,
    "mutual_information": 0.41
  },
  "marginals": {
    "p_state_topk": [{"id":7,"p":0.62},{"id":3,"p":0.19}],
    "p_phase_hist": [0.05,0.08,0.12]
  }
}
```
Tip: store 'bins' (edges) with p_phase_hist so plots are reproducible.

---

## 6) UI Rules (drop into app)
- Header chip: "Sync 0.63 • +-0.12 (90% PI) • Hs 0.22 • Hp 0.18"
- "Why this score?" popover:
  - State posterior bars (+ entropy)
  - Phase histogram + kappa_hat
  - Small MI sparkline (last 30s)
- Trust gates: if PI width>tau or OOD high -> "Low confidence, collect more data."

---

## 7) Checklist (in-code comments)
- [ ] Compute p(z_t), H_state each frame; log top-k & entropy
- [ ] Bin relative phase; compute H_phase, kappa_hat, and MI with recent window
- [ ] Calibrate probabilities (temperature / Dirichlet); track ECE/Brier
- [ ] Produce conformal PI for synchrony_score; tag coverage in logs
- [ ] Persist marginals & entropies with model/version IDs
- [ ] UI shows intervals, entropy chips, drift status; fail-closed on low confidence

---

## 8) Quick Example (end-to-end prototype)
```python
def summarise_window(state_post, phase_angles):
    Hs = normalized_entropy(state_post)
    circ = circular_confidence(phase_angles)
    bins = np.linspace(-np.pi, np.pi, 13)  # 12 bins
    hist, _ = np.histogram(phase_angles, bins=bins, density=True)
    Hp = normalized_entropy(hist + EPS)
    joint = np.outer(state_post / state_post.sum(), hist / hist.sum())
    MI = mutual_information(joint)
    return dict(H_state=Hs, H_phase=Hp, kappa=float(circ["kappa_hat"]), MI=float(MI), p_phase_hist=hist.tolist())
```
Remember: entropy & marginals are not extra. They are the explanation layer. Keep them in every API, plot, and decision.
