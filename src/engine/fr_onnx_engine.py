"""Face recognition onnx engine."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

import numpy as np

from src.engine.arcface_onnx_engine import ArcfaceOnnxEngine
from src.engine.yolo_onnx_engine import YoloxOnnxEngine
from src.schema.fr_schema import FrResultSchema
from src.schema.yolo_schema import YoloResultSchema
from src.utils.logger import get_logger

log = get_logger()


class FrOnnxEngine:
    """Face recognition ONNX engine module."""

    def __init__(
        self,
        det_engine_path: str,
        rec_engine_path: str,
        det_max_end2end: int = 100,
        provider: str = "cpu",
    ) -> None:
        """Initialize face recognition ONNX engine."""
        self.det_engine_path = det_engine_path
        self.rec_engine_path = rec_engine_path
        self.det_max_end2end = det_max_end2end
        self.provider = provider

    def setup(self) -> None:
        """Setup face recognition ONNX engine."""
        log.info(f"Setup face recognition ONNX engine")

        # setup face detection engine
        self.det_engine = YoloxOnnxEngine(
            engine_path=self.det_engine_path,
            categories=["face"],
            provider=self.provider,
            max_det_end2end=self.det_max_end2end,
        )
        self.det_engine.setup()

        # setup face recognition engine
        self.rec_engine = ArcfaceOnnxEngine(
            engine_path=self.rec_engine_path, provider=self.provider
        )
        self.rec_engine.setup()

        log.info(f"Face recognition ONNX engine setup complete")

    def predict(
        self, imgs: List[np.ndarray], det_conf: float = 0.25, det_nms: float = 0.45
    ) -> List[FrResultSchema]:
        """Predict embeddings from image(s)."""
        # detect faces
        det_results = self.detect_faces(imgs, det_conf, det_nms)

        # get embeddings
        results: List[FrResultSchema] = []
        batch_faces = self.preprocess_rec(det_results, imgs)
        for faces, dets in zip(batch_faces, det_results):
            if len(dets.boxes) == 0:
                results.append(FrResultSchema())
                continue

            embds = self.get_embds(faces)
            result = FrResultSchema(
                boxes=dets.boxes,
                scores=dets.scores,
                categories=dets.categories,
                embeddings=embds,
            )

            results.append(result)

        return results

    def detect_faces(
        self, imgs: List[np.ndarray], conf: float = 0.25, nms: float = 0.45
    ) -> List[YoloResultSchema]:
        """Detect faces from image(s)."""
        return self.det_engine.predict(imgs, conf, nms)

    def get_embds(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        """Get embeddings from image(s)."""
        return self.rec_engine.predict(imgs)

    def preprocess_rec(
        self, batch_dets: List[YoloResultSchema], imgs: List[np.ndarray]
    ):
        """Preprocess faces for recognition."""
        # crop faces
        batch_faces: List[List[np.ndarray]] = []
        for i, dets in enumerate(batch_dets):
            if len(dets.boxes) == 0:
                batch_faces.append([])
                continue
            faces: List[np.ndarray] = []
            for box in dets.boxes:
                face = imgs[i][box[1] : box[3], box[0] : box[2]]
                faces.append(face)
            batch_faces.append(faces)

        return batch_faces
