"""YOLO ONNX engine."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

import cv2
import numpy as np

from src.engine.onnx_engine import CommonOnnxEngine
from src.schema.yolo_schema import YoloE2EResultSchema, YoloResultSchema
from src.utils.logger import get_logger

log = get_logger()


class YoloxOnnxEngine(CommonOnnxEngine):
    """YoloX ONNX engine module."""

    def __init__(
        self,
        engine_path: str,
        categories: List[str] = ["face"],
        provider: str = "cpu",
        max_det_end2end: int = 100,
    ) -> None:
        """Initialize YOLO ONNX engine."""
        super().__init__(engine_path, provider)
        self.categories = categories
        self.max_det_end2end = max_det_end2end

    def predict(
        self, imgs: List[np.ndarray], conf: float = 0.25, nms: float = 0.45
    ) -> List[YoloResultSchema]:
        """Detect objects from image(s)."""
        imgs, ratios, pads = self.preprocess_imgs(imgs)
        outputs = self.engine.run(None, {self.metadata[0].input_name: imgs})
        results = self.postprocess_end2end(outputs, ratios, pads, conf)

        return results

    def preprocess_imgs(self, imgs: List[np.ndarray]):
        """Preprocess images."""
        # resize and pad
        dst_h, dst_w = self.img_shape
        resized_imgs = np.ones((len(imgs), dst_h, dst_w, 3), dtype=np.float32) * 114
        ratios = np.ones((len(imgs)), dtype=np.float32)
        pads = np.ones((len(imgs), 2), dtype=np.float32)
        for i, img in enumerate(imgs):
            src_h, src_w = img.shape[:2]
            ratio = min(dst_w / src_w, dst_h / src_h)
            resized_w, resized_h = int(src_w * ratio), int(src_h * ratio)
            dw, dh = (dst_w - resized_w) / 2, (dst_h - resized_h) / 2
            img = cv2.resize(img, (resized_w, resized_h))
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114
            )
            resized_imgs[i] = img

            pads[i] = np.array([dw, dh], dtype=np.float32)
            ratios[i] = ratio

        # transpose to NCHW
        resized_imgs = resized_imgs.transpose((0, 3, 1, 2))

        return resized_imgs, ratios, pads

    def postprocess_end2end(
        self,
        outputs: List[np.ndarray],
        ratios: np.ndarray,
        pads: np.ndarray,
        conf: float = 0.25,
    ):
        """Postprocess end2end outputs."""
        # parse outputs
        e2e_outputs: List[YoloE2EResultSchema] = []
        for i in range((outputs[0].shape[0])):  # batch size
            num_dets = int(outputs[0][i][0])
            boxes = outputs[1][i][:num_dets]
            scores = outputs[2][i][:num_dets]
            classes = outputs[3][i][:num_dets]
            e2e_outputs.append(
                YoloE2EResultSchema(
                    num_dets=num_dets,
                    boxes=boxes,
                    scores=scores,
                    classes=classes,
                )
            )

        # scaling and filtering
        results: List[YoloResultSchema] = []
        for i, out in enumerate(e2e_outputs):
            # scale bbox to original image size
            out.boxes[:, 0::2] -= pads[i][0]
            out.boxes[:, 1::2] -= pads[i][1]
            out.boxes /= ratios[i]

            # filter by conf
            mask = out.scores > conf
            out.boxes = out.boxes[mask].astype(np.int32)
            out.scores = out.scores[mask]
            out.classes = out.classes[mask]

            # filter by class
            mask = out.classes < len(self.categories)
            out.boxes = out.boxes[mask]
            out.scores = out.scores[mask]
            out.classes = out.classes[mask]

            results.append(
                YoloResultSchema(
                    categories=[self.categories[int(i)] for i in out.classes],
                    scores=out.scores,
                    boxes=out.boxes,
                )
            )

        return results
