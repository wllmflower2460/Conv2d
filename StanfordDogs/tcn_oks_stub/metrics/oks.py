from __future__ import annotations
import numpy as np
from typing import Dict, Iterable, Tuple

# COCO-style keypoint constants. For dogs, start with human defaults and allow override.
COCO_K = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
    0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
], dtype=np.float32)

def ensure_k(k: np.ndarray, K: int) -> np.ndarray:
    if k is None or len(k) == 0:
        return np.ones((K,), dtype=np.float32) * 0.05  # neutral fallback
    if len(k) < K:
        # pad by repeating last value
        pad = np.full((K - len(k),), k[-1], dtype=np.float32)
        return np.concatenate([k.astype(np.float32), pad], axis=0)
    return k[:K].astype(np.float32)

def oks_single(
    pred_kpts: np.ndarray,  # (K,2)
    gt_kpts: np.ndarray,    # (K,2)
    vis: np.ndarray,        # (K,) in {0,1}
    scale: float,           # area of GT object
    k: np.ndarray | None = None,  # (K,)
) -> float:
    """Compute OKS for a single instance.

    OKS = mean_i[exp( - d_i^2 / (2 * s^2 * k_i^2) )] over visible keypoints i.
    """
    K = gt_kpts.shape[0]
    k_use = ensure_k(k if k is not None else np.full(K, 0.05, dtype=np.float32), K)
    vis_mask = (vis > 0).astype(bool)
    if vis_mask.sum() == 0:
        return 0.0
    d2 = np.sum((pred_kpts - gt_kpts) ** 2, axis=1)  # (K,)
    denom = 2.0 * (scale + 1e-6) * (k_use ** 2)      # (K,)
    oks_all = np.exp(-d2 / denom)
    return float(oks_all[vis_mask].mean())

def oks_map(
    preds: Iterable[np.ndarray],
    gts: Iterable[np.ndarray],
    visses: Iterable[np.ndarray],
    scales: Iterable[float],
    k: np.ndarray | None = None,
    thresholds: Iterable[float] = tuple(np.arange(0.50, 0.96, 0.05)),
) -> Dict[str, float]:
    """Compute mean OKS and mAP-style OKS over thresholds.

    Returns dict with mean_oks, oks@t for each threshold, and oks_map.
    """
    preds = list(preds); gts = list(gts); visses = list(visses); scales = list(scales)
    assert len(preds) == len(gts) == len(visses) == len(scales), "Mismatched lengths"
    oks_vals = []
    oks_hits = {t: [] for t in thresholds}
    for p, g, v, s in zip(preds, gts, visses, scales):
        val = oks_single(p, g, v, s, k=k)
        oks_vals.append(val)
        for t in thresholds:
            oks_hits[t].append(1.0 if val >= t else 0.0)
    mean_oks = float(np.mean(oks_vals)) if oks_vals else 0.0
    results = {"mean_oks": mean_oks}
    for t in thresholds:
        results[f"oks@{t:.2f}"] = float(np.mean(oks_hits[t])) if oks_hits[t] else 0.0
    results["oks_map"] = float(np.mean([results[f"oks@{t:.2f}"] for t in thresholds])) if thresholds else 0.0
    return results
