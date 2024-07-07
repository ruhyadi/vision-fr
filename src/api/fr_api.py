"""Face recognition api router."""

import rootutils

ROOT = rootutils.autosetup()

from datetime import datetime

import cv2
from fastapi import APIRouter, File, UploadFile
from sqlmodel import select

from src.api.base_api import BaseApi
from src.schema.configs import Configs
from src.schema.fr_schema import FacesFrSqlSchema
from src.utils.logger import get_logger

log = get_logger()


class FrApi(BaseApi):
    """Face recognition API router."""

    def __init__(self, cfg: Configs) -> None:
        """Initialize the face recognition API router."""
        super().__init__(cfg)
        self.router = APIRouter()

        self.setup()

    def setup(self) -> None:
        """Setup the face recognition API router."""

        @self.router.get("/test")
        async def test():

            face = FacesFrSqlSchema(
                name="test",
                embedding=[1 for _ in range(512)],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            self.pg.session.add(face)
            self.pg.session.commit()

            log.warning(f"Added face: {face}")

        @self.router.get("/distance")
        async def distance():

            faces = self.pg.session.exec(
                select(FacesFrSqlSchema).filter(
                    FacesFrSqlSchema.embedding.l2_distance([2 for _ in range(512)])
                    < 0.1
                )
            ).all()

            log.warning(f"Found faces: {faces}")

