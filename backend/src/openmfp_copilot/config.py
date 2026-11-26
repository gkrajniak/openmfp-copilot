from functools import lru_cache
from typing import Literal

from pydantic import PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

type Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENMFP_COPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    app_name: str = "OpenMFP Copilot"
    environment: Literal["local", "test", "production"] = "local"
    cors_origins: list[str] = ["http://localhost:4200"]

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-haiku-5-5"
    anthropic_effort: Effort = "low"

    voyage_api_key: SecretStr | None = None
    voyage_model: str = "voyage-4"

    database_url: PostgresDsn = PostgresDsn(
        "postgresql+asyncpg://copilot:copilot@localhost:5432/openmfp_copilot"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
