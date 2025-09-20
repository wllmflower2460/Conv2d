
from __future__ import annotations
from typing import Optional

from predictors.yolov8_adapter import YOLOv8Predictor
from predictors.sleap_adapter import SLEAPPredictor

def load_predictor_by_name(name: str, model_path: Optional[str] = None, **kw):
    name = name.lower()
    if name in ("yolo", "yolov8", "yolo8"):
        if not model_path:
            raise ValueError("YOLOv8 predictor requires --model path to weights (.pt)")
        return YOLOv8Predictor(model_path, **kw)
    if name in ("sleap",):
        if not model_path:
            raise ValueError("SLEAP predictor requires --model path to a SLEAP model file")
        return SLEAPPredictor(model_path, **kw)
    raise ValueError(f"Unknown predictor: {name}")
