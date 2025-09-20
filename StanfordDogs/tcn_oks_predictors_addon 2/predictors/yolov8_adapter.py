
from __future__ import annotations
from typing import Dict, Any
import numpy as np

from .base import KeypointPredictor, harmonize_kpts, harmonize_vis

class YOLOv8Predictor(KeypointPredictor):
    """Ultralytics YOLOv8 Pose adapter."""

    def __init__(self, weights: str, conf: float = 0.25, iou: float = 0.5, kpt_conf: float = 0.5, device: str | None = None):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError("Ultralytics is not installed. `pip install ultralytics`") from e
        self.YOLO = YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.kpt_conf = kpt_conf
        self.device = device

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        img = batch.get("image", None)
        if img is None:
            raise ValueError("Batch missing 'image' ndarray for YOLOv8Predictor.")
        results = self.model.predict(img, conf=self.conf, iou=self.iou, verbose=False, device=self.device)
        res = results[0]

        if res.keypoints is None or len(res.keypoints) == 0:
            K = batch["kpts"].shape[0] if "kpts" in batch else 20
            return {
                "kpts": np.zeros((K,2), dtype=np.float32),
                "kvis": np.zeros((K,), dtype=np.int32),
                "bbox": np.zeros((4,), dtype=np.float32)
            }

        if res.boxes is not None and len(res.boxes) > 0:
            idx = int(np.argmax(res.boxes.conf.cpu().numpy()))
            box_xyxy = res.boxes.xyxy.cpu().numpy()[idx]
        else:
            idx = 0
            box_xyxy = np.zeros((4,), dtype=np.float32)

        kpts_xy = res.keypoints.xy.cpu().numpy()[idx]
        kpts_conf = res.keypoints.conf.cpu().numpy()[idx]

        vis = (kpts_conf >= self.kpt_conf).astype(np.int32)

        target_K = batch["kpts"].shape[0] if "kpts" in batch else kpts_xy.shape[0]
        kpts_xy = harmonize_kpts(kpts_xy, target_K)
        vis = harmonize_vis(vis, target_K)

        return {
            "kpts": kpts_xy.astype(np.float32),
            "kvis": vis.astype(np.int32),
            "bbox": box_xyxy.astype(np.float32)
        }
