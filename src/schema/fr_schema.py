"""Face recognition schema."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

from pydantic import Field

from src.schema.yolo_schema import YoloResultSchema


class FrResultSchema(YoloResultSchema):
    """Face recognition result schema."""

    embeddings: List[List[float]] = Field([], example=[[0.0, 1.0, 10.0]])
