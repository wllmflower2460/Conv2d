from __future__ import annotations
from typing import Tuple
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

def draw_keypoints(img: np.ndarray, kpts: np.ndarray, vis: np.ndarray, radius: int = 3) -> np.ndarray:
    if cv2 is None:
        return img
    out = img.copy()
    for i, (p, v) in enumerate(zip(kpts, vis)):
        if v <= 0:
            continue
        x, y = int(p[0]), int(p[1])
        cv2.circle(out, (x, y), radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    return out

def draw_bbox(img: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return img
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2, lineType=cv2.LINE_AA)
    return out
