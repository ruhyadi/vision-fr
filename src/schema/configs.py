"""Application settings."""

import rootutils

ROOT = rootutils.autosetup()

import secrets
from typing import Annotated, Any, List, Literal, Union

from pydantic import AnyUrl, BeforeValidator, PostgresDsn, computed_field
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
    API_JWT_ALGORITHM: str = "HS256"
    API_JWT_SECRET: str = secrets.token_urlsafe(32)
    API_RESET_PWD_SECRET: str = secrets.token_urlsafe(32)
    API_JWT_EXPIRE: int = 3600
    API_CORS_ORIGINS: Annotated[list[AnyUrl] | str, BeforeValidator(parse_cors)] = []
    ENVIRONMENT: Literal["devel", "prod"] = "devel"
    SERVER: Literal["uvicorn", "gunicorn"] = "uvicorn"

    # postgres settings
    POSTGRES_HOST: str = "vision-fr-pg"
    POSTGRES_PORT: int = 7031
    POSTGRES_USER: str = "didi"
    POSTGRES_PASSWORD: str = "didi123"
    POSTGRES_DB: str = "vision-fr"

    @computed_field
    @property
    def POSTGRES_URI(self) -> PostgresDsn:
        return PostgresDsn.build(
            scheme="postgresql",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_HOST,
            port=self.POSTGRES_PORT,
            path=self.POSTGRES_DB,
        )
    
    # fr engine
    FR_DET_ENGINE_PATH: str = None
    FR_REC_ENGINE_PATH: str = None
    FR_DET_MAX_END2END: int = 100
    FR_PROVIDER: Literal["cpu", "gpu"] = "cpu"

cfg = Configs()
