"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import AliasChoices, AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Streamlit application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False)

    vlm_provider: str = Field(default="cohere", validation_alias=AliasChoices("VLM_PROVIDER", "vlm_provider"))
    cohere_api_key: str | None = Field(default=None, validation_alias=AliasChoices("COHERE_API_KEY", "cohere_api_key"))
    cohere_api_url: AnyHttpUrl | None = Field(
        default="https://api.cohere.com/v2/chat",
        validation_alias=AliasChoices("COHERE_API_URL", "cohere_api_url"),
    )
    cohere_model: str = Field(
        default="command-a-vision-07-2025",
        validation_alias=AliasChoices("COHERE_MODEL", "cohere_model"),
    )
    OPENAI_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_KEY", "OPENAI_KEY", "VLM_API_KEY", "vlm_api_key"),
    )
    openai_api_url: AnyHttpUrl | None = Field(
        default="https://api.openai.com/v1/responses",
        validation_alias=AliasChoices("OPENAI_API_URL", "openai_api_url", "VLM_API_URL", "vlm_api_url"),
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("OPENAI_MODEL", "openai_model", "VLM_MODEL", "vlm_model"),
    )
    hf_token: str | None = Field(default=None, validation_alias=AliasChoices("HF_TOKEN", "hf_token"))
    hf_api_url: AnyHttpUrl | None = Field(
        default="https://router.huggingface.co/v1/chat/completions",
        validation_alias=AliasChoices("HF_API_URL", "hf_api_url"),
    )
    default_model_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DEFAULT_MODEL_KEY", "default_model_key"),
    )
    request_timeout_seconds: int = Field(
        default=60,
        validation_alias=AliasChoices("REQUEST_TIMEOUT_SECONDS", "request_timeout_seconds"),
    )
    default_confidence_threshold: float = Field(
        default=0.55,
        validation_alias=AliasChoices("CONFIDENCE_THRESHOLD", "confidence_threshold", "default_confidence_threshold"),
    )
    group_processing_concurrency: int = Field(
        default=3,
        validation_alias=AliasChoices("GROUP_PROCESSING_CONCURRENCY", "group_processing_concurrency"),
    )
    request_retry_attempts: int = Field(
        default=3,
        validation_alias=AliasChoices("REQUEST_RETRY_ATTEMPTS", "request_retry_attempts"),
    )
    request_retry_base_delay_seconds: float = Field(
        default=2.0,
        validation_alias=AliasChoices("REQUEST_RETRY_BASE_DELAY_SECONDS", "request_retry_base_delay_seconds"),
    )

    @property
    def vlm_api_key(self) -> str | None:
        """Backward-compatible alias for the OpenAI API key."""

        return self.OPENAI_KEY

    @property
    def vlm_api_url(self) -> AnyHttpUrl | None:
        """Backward-compatible alias for the OpenAI API URL."""

        return self.openai_api_url

    @property
    def vlm_model(self) -> str:
        """Backward-compatible alias for the OpenAI model."""

        return self.openai_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()


settings = get_settings()
