from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration (environment variables prefixed EP_, or a .env file)."""

    model_config = SettingsConfigDict(env_prefix="EP_", env_file=".env", extra="ignore")

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'data' / 'experimentpilot.db').as_posix()}"
    data_dir: Path = BACKEND_DIR / "data"
    max_upload_mb: int = Field(25, ge=1, le=500)
    max_rows: int = Field(200_000, ge=100)
    max_columns: int = Field(500, ge=2)
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"
    executor: str = Field("process", pattern="^(process|thread)$")
    experiment_timeout_s: int = Field(180, ge=1)
    benchmark_workers: int = Field(2, ge=1, le=16)
    environment: str = "local"
    expose_docs: bool = True
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    get_settings.cache_clear()
