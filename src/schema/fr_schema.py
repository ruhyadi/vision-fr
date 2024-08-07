"""Face recognition schema."""

import rootutils

ROOT = rootutils.autosetup()

from datetime import datetime
from typing import Any, List, Optional

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, Field
from sqlalchemy import Column
from sqlmodel import Field as SqlField
from sqlmodel import SQLModel

from src.schema.yolo_schema import YoloResultSchema


class FrResultSchema(YoloResultSchema):
    """Face recognition result schema."""

    embeddings: List[List[float]] = Field([], example=[[0.0, 1.0, 10.0]])


class FacesFrSqlSchema(SQLModel, table=True):
    """Faces face recognition schema."""

    __tablename__ = "faces"

    id: Optional[int] = SqlField(default=None, primary_key=True)
    name: str = SqlField(..., max_length=100)
    embedding: Any = SqlField([], sa_column=Column(Vector(512)))
    created_at: datetime = SqlField(...)
    updated_at: datetime = SqlField(...)


class ReadFacesFrSchema(BaseModel):
    """Read faces face recognition schema."""

    id: Optional[int] = Field(None, example=1)
    name: str = Field(None, example="John Doe")
    box: Optional[List[int]] = Field(None, example=[0, 0, 100, 100])
    created_at: Optional[datetime] = Field(None)
    updated_at: Optional[datetime] = Field(None)
