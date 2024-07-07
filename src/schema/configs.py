"""Application settings."""

import rootutils

ROOT = rootutils.autosetup()

from typing import Any, List, Literal, Union

from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_cors(v: Any) -> Union[List[str], str]:
    """Parse CORS origins."""
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, Union[list, str]):
        return v
    raise ValueError(v)


class Configs(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    # api settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 6090
    API_WORKERS: int = 1
    SERVER: Literal["uvicorn", "gunicorn"] = "uvicorn"

    # postgres settings
    POSTGRES_HOST: str = "vision-fr-pg"
    POSTGRES_PORT: int = 7031
    POSTGRES_USER: str = "didi"
    POSTGRES_PASSWORD: str = "didi123"
    POSTGRES_DB: str = "vision-fr"

    # fr engine
    FR_DET_ENGINE_PATH: str = "assets/yoloxs_face.onnx"
    FR_REC_ENGINE_PATH: str = "assets/w600k_mbf.onnx"
    FR_DET_MAX_END2END: int = 100
    FR_PROVIDER: Literal["cpu", "gpu"] = "cpu"


cfg = Configs()
