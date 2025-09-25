# VQ Recovery & Calibration Quick Fix
Updated: 2025-09-21

## What you get
1) A robust **VQ-EMA** module (`VectorQuantizerEMA2D_Stable.py`) with dead-code reinit and proper perplexity.
2) A tiny **fallback tokenizer** (`fallback_tokenizer.py`) to avoid blocked runs when VQ collapses.
3) **Temperature scaling** (`temperature_scaler.py`) to reduce ECE at evaluation.
4) A small **model patch** (`conv2d_vq_model.patch`) that adds pre-VQ normalization and the fallback.

## How to use
**Option A (replace VQ):**
```python
from VectorQuantizerEMA2D_Stable import VectorQuantizerEMA2D_Stable as VectorQuantizerEMA2D
```
Then re-train VQ variants. Targets: perplexity in [50, 200], usage > 0.30.

**Option B (guard existing VQ):**
Place `fallback_tokenizer.py` in your repo root (or `models/`) and apply the patch:
```bash
patch -p0 < conv2d_vq_model.patch
```
This normalizes features before VQ and auto-falls back to k-means tokens if VQ is unhealthy.

## Calibration (ECE)
Fit temperature on a held-out split and report ECE:
```python
from temperature_scaler import TemperatureScaler, expected_calibration_error
scaler = TemperatureScaler()
T = scaler.fit(val_logits, val_labels)               # fit on 40-50% calibration subset
probs_ts = torch.softmax(scaler(val_logits), 1)      # apply to eval
ece = expected_calibration_error(probs_ts, val_labels, n_bins=15)
print("Temp:", T, "ECE:", ece)
```

## Debug checklist if VQ still collapses
- Check reshape: VQ sees (B*T, D) from (B, D, 1, T).
- Ensure EMA updates use `z_e.detach()` (no grad through EMA stats).
- Try `beta=0.25` or `decay=0.95`.
- Freeze encoder for 1–2k steps to let EMA stabilize.
- Inspect token index histogram; if >80% same index, codebook is still collapsed.
