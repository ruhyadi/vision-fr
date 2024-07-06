"""Arcface ONNX engine."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

import cv2
import numpy as np

from src.engine.onnx_engine import CommonOnnxEngine
from src.utils.logger import get_logger

log = get_logger()


class ArcfaceOnnxEngine(CommonOnnxEngine):
    """Arcface ONNX engine module."""

    def __init__(self, engine_path: str, provider: str = "cpu") -> None:
        """Initialize Arcface ONNX engine."""
        super().__init__(engine_path, provider)

    def predict(self, imgs: List[np.ndarray]) -> np.ndarray:
        """Predict embeddings from image(s)."""
        imgs = self.preprocess_imgs(imgs)
        outputs = self.engine.run(None, {self.metadata[0].input_name: imgs})

        return outputs[0]

    def preprocess_imgs(self, imgs: List[np.ndarray]) -> np.ndarray:
        """Preprocess images (faces)."""
        # resize faces
        dst_h, dst_w = self.img_shape
        resized_imgs = np.zeros((len(imgs), dst_h, dst_w, 3), dtype=np.float32)
        for i, img in enumerate(imgs):
            resized_imgs[i] = cv2.resize(img, (dst_w, dst_h))

        # normalize faces
        resized_imgs = resized_imgs.transpose(0, 3, 1, 2)
        resized_imgs /= 255.0

        return resized_imgs
