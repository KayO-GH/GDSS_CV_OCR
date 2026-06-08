"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import AliasChoices, AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Streamlit application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False)

    vlm_api_key: str | None = Field(default=None, validation_alias=AliasChoices("VLM_API_KEY", "vlm_api_key"))
    vlm_api_url: AnyHttpUrl | None = Field(default=None, validation_alias=AliasChoices("VLM_API_URL", "vlm_api_url"))
    vlm_model: str = Field(default="gpt-4o-mini", validation_alias=AliasChoices("VLM_MODEL", "vlm_model"))
    request_timeout_seconds: int = Field(
        default=60,
        validation_alias=AliasChoices("REQUEST_TIMEOUT_SECONDS", "request_timeout_seconds"),
    )
    default_confidence_threshold: float = Field(
        default=0.55,
        validation_alias=AliasChoices("CONFIDENCE_THRESHOLD", "confidence_threshold", "default_confidence_threshold"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()


settings = get_settings()
