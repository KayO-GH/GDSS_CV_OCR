"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Streamlit application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    vlm_api_key: str | None = Field(default=None, env="VLM_API_KEY")
    vlm_api_url: AnyHttpUrl | None = Field(default=None, env="VLM_API_URL")
    vlm_model: str = Field(default="gpt-4o-mini", env="VLM_MODEL")
    request_timeout_seconds: int = Field(default=60, env="REQUEST_TIMEOUT_SECONDS")
    default_confidence_threshold: float = Field(default=0.55, env="CONFIDENCE_THRESHOLD")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()


settings = get_settings()

