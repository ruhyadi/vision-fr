"""Face recognition api router."""

import rootutils

ROOT = rootutils.autosetup()

from datetime import datetime
from typing import List, Literal

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from sqlmodel import select

from src.api.base_api import BaseApi
from src.engine.fr_onnx_engine import FrOnnxEngine
from src.schema.configs import Configs
from src.schema.fr_schema import FacesFrSqlSchema, ReadFacesFrSchema
from src.utils.logger import get_logger

log = get_logger()


class FrApi(BaseApi):
    """Face recognition API router."""

    def __init__(self, cfg: Configs) -> None:
        """Initialize the face recognition API router."""
        super().__init__(cfg)
        self.router = APIRouter()

        self.setup_engine()
        self.setup()

    def setup_engine(self) -> None:
        """Setup the face recognition engine."""
        self.engine = FrOnnxEngine(
            det_engine_path=self.cfg.FR_DET_ENGINE_PATH,
            rec_engine_path=self.cfg.FR_REC_ENGINE_PATH,
            det_max_end2end=self.cfg.FR_DET_MAX_END2END,
            provider=self.cfg.FR_PROVIDER,
        )
        self.engine.setup()

    def setup(self) -> None:
        """Setup the face recognition API router."""

        @self.router.get(
            "/face",
            response_model=List[ReadFacesFrSchema],
        )
        async def list_faces() -> List[ReadFacesFrSchema]:
            """List all faces."""
            log.log(21, f"Request to list all faces")
            faces = self.pg.session.exec(select(FacesFrSqlSchema)).all()

            log.log(21, f"Founds {len(faces)} faces")

            return [ReadFacesFrSchema(**face.model_dump(), box=None) for face in faces]

        @self.router.post(
            "/face/register",
            response_model=ReadFacesFrSchema,
        )
        async def register_face(
            name: str,
            image: UploadFile = File(...),
            detConf: float = 0.25,
            detNms: float = 0.45,
        ) -> ReadFacesFrSchema:
            """Register a face."""
            log.log(21, f"Request to register a face with name: {name}")

            # check if name already exists
            faces_db = self.pg.session.exec(select(FacesFrSqlSchema)).all()
            for face_db in faces_db:
                if face_db.name == name:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Name already exists",
                    )

            # bytes to numpy
            img_np = await self.preprocess_img_bytes(await image.read())

            # recognize faces
            faces = self.engine.predict([img_np], det_conf=detConf, det_nms=detNms)
            # single batch
            faces = faces[0]

            # check if face detected
            if len(faces.boxes) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No face detected",
                )

            # only allow one face
            if len(faces.boxes) > 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only one face allowed",
                )

            # only iterate once
            for embd in faces.embeddings:
                face = FacesFrSqlSchema(
                    name=name,
                    embedding=embd,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                self.pg.session.add(face)
                self.pg.session.commit()
                self.pg.session.refresh(face)

            log.log(21, f"Face registered with id: {face.id}")

            return ReadFacesFrSchema(**face.model_dump(), box=faces.boxes[0])

        @self.router.post(
            "/face/recognize",
            response_model=List[ReadFacesFrSchema],
        )
        async def recognize_faces(
            image: UploadFile = File(...),
            method: Literal["cosine"] = "cosine",
            distance: float = 0.5,
            detConf: float = 0.25,
            detNms: float = 0.45,
        ) -> List[ReadFacesFrSchema]:
            """Recognize faces."""
            log.log(21, f"Request to recognize faces")

            # bytes to numpy
            img_np = await self.preprocess_img_bytes(await image.read())

            # recognize faces
            faces = self.engine.predict([img_np], det_conf=detConf, det_nms=detNms)
            # single batch
            faces = faces[0]

            # check if face detected
            if len(faces.boxes) == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No face detected",
                )

            # query similar faces
            results: List[ReadFacesFrSchema] = []
            for box, embd in zip(faces.boxes, faces.embeddings):
                # TODO: add more methods
                if method == "cosine":
                    responses = self.pg.session.exec(
                        select(FacesFrSqlSchema)
                        .filter(
                            FacesFrSqlSchema.embedding.cosine_distance(embd) < distance
                        )
                        .limit(1)
                    ).all()
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Method not supported",
                    )
                response = responses[0] if responses else None
                if response:
                    results.append(ReadFacesFrSchema(**response.model_dump(), box=box))
                else:
                    results.append(ReadFacesFrSchema(box=box))

            return results

        @self.router.delete(
            "/face/{id}",
            status_code=status.HTTP_204_NO_CONTENT,
        )
        async def delete_face(id: int):
            """Delete a face."""
            log.log(21, f"Request to delete face with id: {id}")

            face = self.pg.session.exec(
                select(FacesFrSqlSchema).filter(FacesFrSqlSchema.id == id)
            ).first()

            if not face:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Face not found",
                )

            self.pg.session.delete(face)
            self.pg.session.commit()

            log.log(21, f"Face deleted with id: {id}")

