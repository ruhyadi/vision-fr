"""Base API module."""

import rootutils

ROOT = rootutils.autosetup()

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, UploadFile
from PIL import Image

from src.db.pg_db import PgSyncDb
from src.schema.configs import Configs
from src.utils.logger import get_logger

log = get_logger()


class BaseApi:
    """Base API router."""

    def __init__(self, cfg: Configs) -> None:
        """Initialize the API router."""
        self.cfg = cfg
        self.app = FastAPI()
        self.router = APIRouter()

        # db
        self.pg = PgSyncDb(
            host=self.cfg.POSTGRES_HOST,
            port=self.cfg.POSTGRES_PORT,
            user=self.cfg.POSTGRES_USER,
            password=self.cfg.POSTGRES_PASSWORD,
            db=self.cfg.POSTGRES_DB,
        )

        self.setup()

    def setup(self) -> None:
        """Setup the API router."""

        @self.router.on_event("startup")
        async def startup_event():
            """Startup event."""
            log.log(21, f"Load startup event")
            self.pg.setup()
            self.pg.create_all()

            log.log(21, f"Startup event complete")

        @self.router.get("/app")
        async def app():
            """App route."""
            return {"status": "ok"}