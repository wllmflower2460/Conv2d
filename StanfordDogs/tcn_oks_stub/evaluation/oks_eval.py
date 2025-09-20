from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from tqdm import tqdm

from datasets.stanford_dogs_extra import StanfordDogsExtraDataset
from metrics.oks import oks_map

def load_predictor(kind: str = "passthrough", noise_px: float = 2.0) -> Callable:
    """Return a callable predictor(batch) -> dict with 'kpts', 'kvis', 'bbox' arrays.

    'passthrough' uses GT keypoints to exercise the OKS pipeline.
    'gaussian' adds small Gaussian noise to GT kpts to simulate imperfect predictions.
    """
    def fn(batch):
        kpts = batch["kpts"].copy()
        if kind == "gaussian":
            kpts = kpts + np.random.normal(scale=noise_px, size=kpts.shape).astype(np.float32)
        return {"kpts": kpts, "kvis": batch["vis"], "bbox": batch["bbox"]}
    return fn

def iterate_dataset(ds: StanfordDogsExtraDataset, batch_size: int = 1):
    # simple generator yielding dict batches
    for i in range(len(ds)):
        s = ds[i]
        w, h = s.meta["img_wh"]
        scale = float((s.bbox_xyxy[2] - s.bbox_xyxy[0]) * (s.bbox_xyxy[3] - s.bbox_xyxy[1]))
        yield {
            "img_wh": np.array([w, h], dtype=np.float32),
            "bbox": s.bbox_xyxy.astype(np.float32),
            "kpts": s.keypoints.astype(np.float32),
            "vis": s.visibility.astype(np.int32),
            "scale": scale,
        }

def run_eval(data_root: str | Path, split: str, subset: str, out_dir: str | Path, predictor_kind: str = "passthrough") -> Dict[str, float]:
    ds = StanfordDogsExtraDataset(root=data_root, split="train" if split == "train" else "val",
                                  smoke=50 if subset == "smoke" else 0)
    pred = load_predictor(predictor_kind)
    preds, gts, visses, scales = [], [], [], []
    for batch in tqdm(iterate_dataset(ds), total=len(ds)):
        out = pred(batch)
        preds.append(out["kpts"])
        gts.append(batch["kpts"])
        visses.append(batch["vis"])
        scales.append(batch["scale"])

    metrics = oks_map(preds, gts, visses, scales, k=None)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "oks_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
