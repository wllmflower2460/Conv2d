# AI Handoff Instructions — Conv2d-VQ-HDP-HSMM (Week 1)

**Date:** 2025-09-21  
**Purpose:** Run the new vector-quantization layer on the remote machine, produce a quick health report (usage, diversity, transitions), and commit the artifacts.

---

## 1) Context (what matters)

- **Input tensor shape:** `(batch, 9 channels, 2 devices, 100 time steps)` kept **static**.  
- **Convolution style:** two-dimensional only (use height = 1 and time = 100).  
- **Disallowed:** one-dimensional conv, group normalization, layer normalization, standard softmax.  
- **New module:** vector quantization (VQ) with straight-through gradients and exponential moving average (EMA) codebook updates.  
- **Typical settings:** codebook size ≈ 512, code dimension ≈ 64.  
- **Key files:**  
  - `models/vq_ema_2d.py` (quantizer)  
  - `models/conv2d_vq_hdp_hsmm.py` (integration skeleton)  
  - `codebook_analysis.py` (produces JSON + PNG artifacts)

**Goal of this run:** verify shapes, run the analyzer on a small batch, commit results.

---

## 2) Requirements on the remote machine

- Python 3.10+  
- Git  
- NVIDIA graphics card drivers installed (for speed; not strictly required to pass this smoke test)  
- Internet access for package installation

---

## 3) One-shot command block (copy/paste)

> Adjust the repository directory if needed (default `~/repos/Conv2d`).

```bash
# 0) Basic system info
nvidia-smi || true
python3 --version

# 1) Get or update the repository
cd ~/repos || mkdir -p ~/repos && cd ~/repos
if [ -d "Conv2d" ]; then
  cd Conv2d && git fetch origin && git switch main && git pull --ff-only
else
  git clone https://github.com/wllmflower2460/Conv2d.git && cd Conv2d
fi

# 2) New working branch for artifacts
git switch -c chore/remote-sprint1-vq-report || git switch chore/remote-sprint1-vq-report

# 3) Python virtual environment
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
# If requirements.txt exists, use it; otherwise install core packages
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install numpy scipy matplotlib pandas scikit-learn xarray
fi

# 4) Quick quantizer self-test
python - <<'PY'
import torch
from pathlib import Path
from models.vq_ema_2d import VectorQuantizerEMA2D
B,D,H,T = 32,64,1,100
z = torch.randn(B,D,H,T)
vq = VectorQuantizerEMA2D(num_codes=512, code_dim=D, decay=0.99, commitment_cost=0.25)
vq.train()
zq, losses, info = vq(z)
print("OK shape:", zq.shape, "loss:", float(losses["vq"]), "usage:", float(info["usage"]))
Path("analysis/codebook_results").mkdir(parents=True, exist_ok=True)
with open("analysis/codebook_results/selftest.txt","w") as f:
    f.write(str({"shape": tuple(zq.shape), "loss": float(losses["vq"]), "usage": float(info["usage"])}))
PY

# 5) Run analyzer on a small batch (uses your checkpoint path if present)
mkdir -p analysis/codebook_results
python codebook_analysis.py --checkpoint models/best_conv2d_vq_model.pth   --save_dir analysis/codebook_results --max_batches 10 || true

# 6) Save environment details
pip freeze > analysis/codebook_results/pip-freeze.txt

# 7) Commit and push artifacts
git add analysis/codebook_results || true
git add *.md || true
git commit -m "remote: Sprint1 VQ analyzer artifacts and environment snapshot" || true
git push -u origin HEAD
```

---

## 4) Expected outputs

Created under `analysis/codebook_results/`:
- `codebook_analysis.json` — metrics: number of active codes, fraction of codebook used, simple cluster summaries, short duration statistics.  
- `codebook_visualization.png` — plots: usage distribution, transition heatmap, low-dimensional view.  
- `selftest.txt` — shape and quick loss from the quantizer self-test.  
- `pip-freeze.txt` — exact package list for repeatability.

If the checkpoint file does not exist, the analyzer will skip model loading gracefully; keep the self-test and environment snapshot.

---

## 5) Acceptance criteria (merge when all are true)

1. Quantizer self-test prints the expected shape `(32, 64, 1, 100)` and a finite loss.  
2. Analyzer produces a JSON file and at least one image file in `analysis/codebook_results/`.  
3. The new branch is pushed to the remote repository with these artifacts.  
4. No failures related to prohibited layers or shape mismatches.

---

## 6) If something fails (quick triage)

- **Import or shape error:** confirm the repository is on the latest `main` branch before creating the work branch.  
- **Package errors:** re-run the virtual environment steps and reinstall packages; check Python version.  
- **No graphics card available:** proceed anyway; this run is light and does not require high speed.  
- **Analyzer missing checkpoint:** skip or provide a path to an available checkpoint and re-run step 5.  
- **Very low code usage:** later we will adjust the commitment weight or codebook update rate; it is not a blocker for this handoff.

---

## 7) Notes for future runs

- Keep the input window shape and the two-dimensional convolution pattern unchanged.  
- Do not introduce one-dimensional convolution, group normalization, layer normalization, or standard softmax.  
- Save all analyzer artifacts in `analysis/codebook_results/` so they can be compared across runs.
