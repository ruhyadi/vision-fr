"""Face recognition api router."""

import rootutils

ROOT = rootutils.autosetup()

import cv2
from fastapi import APIRouter, File, UploadFile

from src.api.base_api import BaseApi

from src.utils.logger import get_logger
from src.schema.configs import Configs

log = get_logger()


class FrApi:
    """Face recognition API router."""

    def __init__(self, cfg: Configs) -> None:
        """Initialize the face recognition API router."""
        self.cfg = cfg
        self.router = APIRouter()