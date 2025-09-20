---
template_type: development_session_input
content_type: development_session
category: documentation
use_case: Plan a focused session to integrate Stanford Dogs + StanfordExtra and OKS into tcn-vae-training-pipeline
section: development
created: 2025-09-13
tags: [session, dataset, stanford_dogs, stanfordextra, yolo, oks, tcn-vae, evaluation, pipeline]
version: 1.0
---

[[Master_MOC]] • [[04__Operations/README]] • [[Development Sessions]]

# Development Session (INPUT): Stanford Dogs + StanfordExtra + OKS → tcn-vae-training-pipeline

## 1) Objectives
- **Load** Stanford Dogs images and **StanfordExtra** keypoints into a clean PyTorch `Dataset`/`DataLoader`.
- **Implement OKS** (Object Keypoint Similarity) metric (vectorized PyTorch + NumPy parity tests).
- **Convert** StanfordExtra JSON → **YOLOv8 keypoint format** (txt) for cross-compat and spot checks.
- **Wire an Evaluation Module** in `tcn-vae-training-pipeline` to compute OKS over predicted keypoints (stub model OK now; replace later).
- **Ship**: CLI utilities + small unit tests + example notebook + sample metrics table.

**Success Criteria**
- `pip install -e .` enables `tcn-vae eval-oks --help` and works end-to-end on a **50-image** smoke subset.
- Reproducible **OKS@0.5:0.95** curve and **OKS mAP** computed on validation split.
- Visual sanity-check script outputs **10 images** with drawn keypoints + boxes to `/artifacts/oks_viz/`.

## 2) Context
- Current pipeline (TCN‑VAE) focuses on IMU/time-series. This session adds a **vision-side evaluation** module using dog keypoints.
- Near-term goal: use pose supervision to shape latent **behavioral motifs** and align with IMU motifs later.

## 3) Scope & Non‑Goals
**In**: Dataset ingest, annotation conversion, OKS metric, eval CLI, minimal visualization.
**Out** (this session): Training a pose model; fusing image keypoints into TCN‑VAE loss; video sequencing.

## 4) Prerequisites / Materials
- Repo: `tcn-vae-training-pipeline` (branch: `feature/oks-eval`).
- Disk: ~5 GB free for images + labels + artifacts.
- Python ≥3.10; PyTorch; OpenCV; Pandas; NumPy; tqdm; pydantic; tyro/click for CLI.
- Data:
  - **Stanford Dogs** images
  - **StanfordExtra** (`StanfordExtra_v12.json`, `*_split.npy`, `keypoint_definitions.csv`)

## 5) Architecture & Integration
```
datasets/
  stanford_dogs_extra.py     # PyTorch Dataset returning (image, bbox, keypoints, vis, meta)
metrics/
  oks.py                      # OKS computation (per-instance, per-image, vectorized batch)
evaluation/
  oks_eval.py                 # Loops over dataset with a predictor fn → OKS stats, PR curves
cli/
  eval_oks.py                 # `tcn-vae eval-oks --data ... --subset smoke|val`
viz/
  draw.py                     # draw_boxes, draw_keypoints, grid writer
contrib/yolo_export/
  to_yolo_kpts.py             # optional: JSON → YOLO keypoint .txt writer
tests/
  test_oks.py, test_dataset.py
notebooks/
  01_oks_quickstart.ipynb
```

**Predictor Abstraction (temporary)**
```python
from typing import Dict
def predict_keypoints(batch) -> Dict[str, np.ndarray]:
    """Return dict with 'kpts': (B, K, 2), 'kvis': (B, K), 'bbox': (B, 4). 
    For now: use ground-truth passthrough or simple noise model to exercise OKS path."""
```

## 6) Detailed Steps (Time‑boxed)

### A. Data Setup (45m)
1. Create `data/stanford_dogs` & `data/stanford_extra` folders.
2. Drop `StanfordExtra_v12.json`, splits, `keypoint_definitions.csv` into `data/stanford_extra/`.
3. Script `contrib/prepare_stanford.py`:
   - Validate file presence, print counts for train/val/test.
   - Optional: copy a **smoke subset (≈50 imgs)** into `data/stanford_smoke/` for quick runs.

**CLI**
```bash
python contrib/prepare_stanford.py --root data --make-smoke 50
```

### B. Dataset Class (60m)
- Implement `StanfordDogsExtraDataset(root, split, transforms=None, smoke=False)`
  - Returns: `image(T,H,W,C|CHW), bbox(4: xyxy), kpts(K,2), vis(K,)∈{0,1}, meta(dict)`
  - Handles NaNs → 0; visibility per spec; single-instance images only.
- Unit test: 5 random samples; assert shapes, vis flags ∈{0,1}.

### C. OKS Metric (60m)
- Implement per-annotation OKS:
  - Inputs: `pred_kpts(K,2)`, `gt_kpts(K,2)`, `vis(K,)`, object scale `s` (area of GT box).
  - Use per-keypoint constants `k_i` (default from COCO; allow override for 20-dog set).
- Vectorized batch OKS returning:
  - **per-instance OKS**, **thresholded matches**, and **OKS mAP** over τ∈{0.50:0.05:0.95}.

**API**
```python
def oks(pred, gt, vis, scale, k: "array[K]" ) -> float
def oks_map(preds, gts, viss, scales, k) -> Dict[str, float]
```

### D. Evaluation Loop + CLI (60m)
- Wire `evaluation/oks_eval.py` to iterate DataLoader, call `predict_keypoints` and compute:
  - mean OKS, OKS@0.50, OKS@0.75, OKS mAP@[.50:.95]
  - dump JSON to `artifacts/oks_metrics.json`
- Add `cli/eval_oks.py` using tyro/click.

### E. Visualization (30m)
- `viz/draw.py`: draw keypoints by visibility; label OKS on image; save grid of 10.
- Save under `artifacts/oks_viz/`

### F. YOLO Export (Optional, 30m)
- `contrib/yolo_export/to_yolo_kpts.py` to emit per-image `.txt` with class id 0 and normalized bbox + kpts + vis (0/1/2 mapping).
- Validate with 3 files; include README note.

## 7) OKS Notes (Dog Keypoints)
- StanfordExtra has **20 annotated kpts**; the 4 missing (eyes, throat, withers) are zeros → **mask out via visibility**.
- Start with COCO `k` constants fallback; add a dict for dog‑specific calibration later.
- Scale `s`: use GT box area; confirm against definition used in reference OKS papers.

## 8) Deliverables
- PR: `feature/oks-eval`
- Artifacts: `oks_metrics.json`, `oks_viz/*.png`, `smoke_results.md` (numbers & notes)
- Tests: `pytest -q` green on dataset + oks
- Docs: `notebooks/01_oks_quickstart.ipynb` + README snippets

## 9) Risks & Mitigations
- **Annotation anomalies** → add `--strict` (skip dubious) vs `--lenient` (accept) flag.
- **Single-instance only** → assert and warn; future multi-instance extension.
- **Keypoint constants (k_i) uncertainty** → config file with overrides; sensitivity check.

## 10) Validation & Exit Criteria
- Run smoke subset → OKS mAP computed without crash; sample viz looks sensible.
- Full **val split** runs within budget (≤ 15 min on laptop GPU/CPU).
- Code quality: mypy/ruff pre-commit passes; tests green.

## 11) Session Timing (3.5 hours total)
- A 0:45  Data setup
- B 1:00  Dataset class + tests
- C 1:00  OKS metric + tests
- D 1:00  Eval loop + CLI
- E 0:30  Visualization (in parallel if possible)
- F 0:30  YOLO export (optional/backlog)

## 12) After-Session Next Steps (Backlog)
- Plug a real keypoint predictor (YOLOv8-pose, RTMPose) into `predict_keypoints`.
- Add **OKS-driven curriculum** to influence VAE latent shaping (contrastive pairs by OKS bins).
- Explore small synthetic sequences by augmenting static images to probe temporal encoders.
- Integrate with **EdgeInfer** path once pose → kinematic features stabilizes.

---

### Quick Commands

```bash
# Create branch
git checkout -b feature/oks-eval

# Install deps
pip install -e .[dev]  # includes torch, opencv-python, tyro/click, numpy, pandas, pytest, ruff, mypy

# Smoke run
python cli/eval_oks.py --data data/stanford_extra --split val --subset smoke --out artifacts

# Tests
pytest -q

# Lint/type
ruff check . && mypy .
```
