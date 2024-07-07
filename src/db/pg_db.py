"""Postgresql db module."""

import rootutils

ROOT = rootutils.autosetup()

from pgvector.psycopg2 import register_vector
from sqlmodel import Session, SQLModel, create_engine, text

from src.utils.logger import get_logger

log = get_logger()


class PgSyncDb:
    """Postgresql syncronous db module."""

    def __init__(self, host: str, port: int, user: str, password: str, db: str) -> None:
        """Initialize SQL database."""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = db

        self.url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    def setup(self) -> None:
        """Setup SQL database."""
        log.log(22, f"Connecting SQL to {self.host}:{self.port}/{self.db}...")
        self.engine = create_engine(
            url=self.url,
            echo=False,
        )
        self.session = Session(self.engine)

        # test connection
        self.session.exec(text("SELECT 1"))

        # pg vector extension
        self.session.exec(text("CREATE EXTENSION IF NOT EXISTS vector"))

        log.log(22, f"Connected SQL to {self.host}:{self.port}/{self.db}")

    def create_all(self) -> None:
        """Create all tables."""
        log.log(22, "Creating all tables...")
        SQLModel.metadata.create_all(self.engine)
