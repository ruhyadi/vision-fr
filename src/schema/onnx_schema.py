"""ONNX engine schema."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

import numpy as np
from pydantic import BaseModel, Field, model_validator


class OnnxMetadataSchema(BaseModel):
    """ONNX metadata schema."""

    input_name: str = Field(..., example="images")
    input_shape: List = Field(..., example=[1, 3, 224, 224])
    output_name: str = Field(..., example="output")
    output_shape: List = Field(..., example=[1, 8400, 85])

    @model_validator(mode="after")
    def validator(self) -> "OnnxMetadataSchema":
        """Check input and output shape."""
        if isinstance(self.input_shape, str):
            self.input_shape = [-1]
        if isinstance(self.output_shape, str):
            self.output_shape = [-1]

        return self
