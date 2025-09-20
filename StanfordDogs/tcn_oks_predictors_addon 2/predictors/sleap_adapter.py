
from __future__ import annotations
from typing import Dict, Any
import numpy as np

from .base import KeypointPredictor, harmonize_kpts, harmonize_vis

class SLEAPPredictor(KeypointPredictor):
    """SLEAP model adapter."""

    def __init__(self, model_path: str, kpt_conf: float = 0.5):
        try:
            import sleap  # type: ignore
        except Exception as e:
            raise RuntimeError("SLEAP is not installed. See https://sleap.ai/") from e

        self.sleap = sleap
        self.model = None
        for loader_name in ("load_model", "load_file"):
            loader = getattr(sleap, loader_name, None)
            if loader is None:
                continue
            try:
                self.model = loader(model_path)
                break
            except Exception:
                continue
        if self.model is None:
            raise RuntimeError(f"Could not load SLEAP model from: {model_path}")
        self.kpt_conf = kpt_conf

    def _infer_np(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        pred = self.model.predict(img)  # type: ignore[attr-defined]
        instances = getattr(pred, "instances", pred)
        n = len(instances)
        if n == 0:
            return {"kpts": np.zeros((0,0,2), dtype=np.float32), "conf": np.zeros((0,0), dtype=np.float32), "boxes": np.zeros((0,4), dtype=np.float32)}
        kpts_list, conf_list, boxes = [], [], []
        for inst in instances:
            points = getattr(inst, "points_array", None) or getattr(inst, "points", None)
            if points is None:
                continue
            pts = np.array(points, dtype=np.float32)
            scores = getattr(inst, "scores", None)
            if scores is None:
                scores = np.ones((pts.shape[0],), dtype=np.float32)
            else:
                scores = np.array(scores, dtype=np.float32)
            if pts.size > 0:
                x1, y1 = np.min(pts[:,0]), np.min(pts[:,1])
                x2, y2 = np.max(pts[:,0]), np.max(pts[:,1])
                box = np.array([x1, y1, x2, y2], dtype=np.float32)
            else:
                box = np.zeros((4,), dtype=np.float32)
            kpts_list.append(pts); conf_list.append(scores); boxes.append(box)
        if not kpts_list:
            return {"kpts": np.zeros((0,0,2), dtype=np.float32), "conf": np.zeros((0,0), dtype=np.float32), "boxes": np.zeros((0,4), dtype=np.float32)}
        K = max(p.shape[0] for p in kpts_list)
        kpts_arr = np.stack([np.pad(p, ((0, K - p.shape[0]), (0,0)), constant_values=0) for p in kpts_list], axis=0)
        conf_arr = np.stack([np.pad(c, (0, K - c.shape[0]), constant_values=0) for c in conf_list], axis=0)
        boxes_arr = np.stack(boxes, axis=0)
        return {"kpts": kpts_arr, "conf": conf_arr, "boxes": boxes_arr}

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        img = batch.get("image", None)
        if img is None:
            raise ValueError("Batch missing 'image' ndarray for SLEAPPredictor.")
        out = self._infer_np(img)
        if out["kpts"].shape[0] == 0:
            K = batch["kpts"].shape[0] if "kpts" in batch else 24
            return {"kpts": np.zeros((K,2), dtype=np.float32), "kvis": np.zeros((K,), dtype=np.int32), "bbox": np.zeros((4,), dtype=np.float32)}

        idx = int(np.argmax(out["conf"].mean(axis=1)))
        kpts = out["kpts"][idx]
        conf = out["conf"][idx]
        bbox = out["boxes"][idx]

        target_K = batch["kpts"].shape[0] if "kpts" in batch else kpts.shape[0]
        kpts = harmonize_kpts(kpts, target_K)
        vis = (conf >= self.kpt_conf).astype(np.int32)
        vis = harmonize_vis(vis, target_K)

        return {"kpts": kpts.astype(np.float32), "kvis": vis.astype(np.int32), "bbox": bbox.astype(np.float32)}
