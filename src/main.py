"""Main function."""

import rootutils

ROOT = rootutils.autosetup()

from src.schema.configs import Configs, cfg
from src.utils.logger import get_logger

log = get_logger()


def main_api(cfg: Configs) -> None:
    """Main API function."""
    from fastapi import FastAPI

    from src.api.base_api import BaseApi
    from src.api.server import GunicornServer, UvicornServer

    log.info(f"Starting API server on {cfg.API_HOST}:{cfg.API_PORT}")

    app = FastAPI(
        title="Face Recognition API",
        description="API for face recognition",
        version="1.0.0",
        docs_url="/",
    )

    # base api
    base_api = BaseApi(cfg)
    app.include_router(base_api.router)

    # server
    if cfg.SERVER == "gunicorn":
        server = GunicornServer(
            app=app,
            host=cfg.API_HOST,
            port=cfg.API_PORT,
            workers=cfg.API_WORKERS,
        )
    elif cfg.SERVER == "uvicorn":
        server = UvicornServer(
            app=app,
            host=cfg.API_HOST,
            port=cfg.API_PORT,
            workers=cfg.API_WORKERS,
        )
    else:
        raise ValueError(f"Invalid server: {cfg.SERVER}")

    server.run()


if __name__ == "__main__":
    """Main function."""

    main_api(cfg)
