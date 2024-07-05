"""YOLO ONNX engine."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List, Union

import cv2
import numpy as np

from src.engine.onnx_engine import CommonOnnxEngine
from src.utils.logger import get_logger

log = get_logger()


class YoloxOnnxEngine(CommonOnnxEngine):
    """YoloX ONNX engine module."""

    def __init__(
        self,
        engine_path: str,
        categories: List[str] = ["face"],
        provider: str = "cpu",
        end2end: bool = False,
        max_det_end2end: int = 100,
    ) -> None:
        """Initialize YOLO ONNX engine."""
        super().__init__(engine_path, provider)
        self.categories = categories
        self.end2end = end2end
        self.max_det_end2end = max_det_end2end
