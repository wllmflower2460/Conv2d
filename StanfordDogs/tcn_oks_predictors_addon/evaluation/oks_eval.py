
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

from datasets.stanford_dogs_extra import StanfordDogsExtraDataset
from metrics.oks import oks_map
from .predictors import load_predictor_by_name

def iterate_dataset(ds: StanfordDogsExtraDataset):
    for i in range(len(ds)):
        s = ds[i]
        w, h = s.meta["img_wh"]
        scale = float((s.bbox_xyxy[2] - s.bbox_xyxy[0]) * (s.bbox_xyxy[3] - s.bbox_xyxy[1]))
        yield {
            "image": s.image,
            "img_wh": np.array([w, h], dtype=np.float32),
            "bbox": s.bbox_xyxy.astype(np.float32),
            "kpts": s.keypoints.astype(np.float32),
            "vis": s.visibility.astype(np.int32),
            "scale": scale,
            "rel_path": s.meta["path"],
        }

def run_eval(data_root: str | Path, split: str, subset: str, out_dir: str | Path, predictor_name: str, model_path: str, **pred_kw) -> Dict[str, float]:
    ds = StanfordDogsExtraDataset(root=data_root, split="train" if split == "train" else "val",
                                  smoke=50 if subset == "smoke" else 0)
    predictor = load_predictor_by_name(predictor_name, model_path=model_path, **pred_kw)

    preds, gts, visses, scales = [], [], [], []
    for batch in tqdm(iterate_dataset(ds), total=len(ds)):
        out = predictor(batch)
        preds.append(out["kpts"])
        gts.append(batch["kpts"])
        visses.append(batch["vis"])
        scales.append(batch["scale"])

    metrics = oks_map(preds, gts, visses, scales, k=None)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "oks_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
