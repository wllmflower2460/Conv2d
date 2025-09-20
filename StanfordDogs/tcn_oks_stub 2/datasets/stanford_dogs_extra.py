from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


@dataclass
class Sample:
    image: np.ndarray  # HxWxC (BGR if cv2, RGB if PIL)
    bbox_xyxy: np.ndarray  # (4,)
    keypoints: np.ndarray  # (K, 2)
    visibility: np.ndarray  # (K,) in {0,1}
    meta: Dict[str, Any]


class StanfordDogsExtraDataset:
    """Single-instance StanfordExtra dataset loader.

    Expects directory structure:
    - root/StanfordExtra_V12/StanfordExtra_v12.json
    - root/StanfordExtra_V12/train_stanford_StanfordExtra_v12.npy
    - root/StanfordExtra_V12/test_stanford_StanfordExtra_v12.npy
    - images under root/Images/* (Stanford Dogs layout)

    Args:
        root: dataset root folder containing StanfordExtra_V12 and Images
        split: "train" | "val" (val uses *test* indices from upstream paper/tooling)
        smoke: if >0, limits to first N samples for quick runs
        prefer_cv2: if True and cv2 is available, images are read via cv2 (BGR). Else PIL (RGB).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "val",
        smoke: int = 0,
        prefer_cv2: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.prefer_cv2 = prefer_cv2 and (cv2 is not None)
        ann_root = self.root / "StanfordExtra_V12"
        json_path = ann_root / "StanfordExtra_v12.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing {json_path}")

        self.data = json.loads(json_path.read_text())
        split_file = "train_stanford_StanfordExtra_v12.npy" if split == "train" else "test_stanford_StanfordExtra_v12.npy"
        ids = np.load(ann_root / split_file)
        self.ids = ids.tolist()
        if smoke and smoke > 0:
            self.ids = self.ids[: int(smoke)]
        self.images_root = self.root / "Images"

    def __len__(self) -> int:
        return len(self.ids)

    def _read_image(self, rel_path: str) -> np.ndarray:
        path = self.images_root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if self.prefer_cv2 and cv2 is not None:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"cv2 failed to read {path}")
            return img  # BGR
        else:
            if Image is None:
                raise RuntimeError("PIL not available and cv2 disabled")
            im = Image.open(path).convert("RGB")
            return np.array(im)  # RGB

    def __getitem__(self, idx: int) -> Sample:
        j = self.data[self.ids[idx]]
        img = self._read_image(j["img_path"])

        w, h = int(j["img_width"]), int(j["img_height"])
        # StanfordExtra bbox stored as [xmin, ymin, width, height]
        xmin, ymin, bw, bh = [float(v) for v in j["img_bbox"]]
        bbox_xyxy = np.array([xmin, ymin, xmin + bw, ymin + bh], dtype=np.float32)

        joints = np.array(j["joints"], dtype=np.float32)  # shape (24, 3) inc. 4 un-annotated
        # Keep only first 20 valid dog keypoints per article; treat vis in {0,1}
        kpts = joints[:, :2]
        vis = joints[:, 2]
        # Replace NaNs with 0 coords and vis=0
        nan_mask = np.isnan(kpts).any(axis=1)
        kpts[nan_mask] = 0.0
        vis[nan_mask] = 0.0
        vis = np.clip(vis, 0, 1)

        meta = {
            "path": j["img_path"],
            "img_wh": (w, h),
            "is_multiple_dogs": bool(j.get("is_multiple_dogs", False)),
        }

        return Sample(
            image=img, bbox_xyxy=bbox_xyxy, keypoints=kpts, visibility=vis.astype(np.int32), meta=meta
        )
