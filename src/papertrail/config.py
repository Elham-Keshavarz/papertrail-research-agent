"""Validated configuration. Secrets are never serialized by the CLI."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PAPERTRAIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_papers: int = Field(default=5, ge=1, le=20)
    max_retries: int = Field(default=2, ge=0, le=5)
    request_timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0)
    max_pdf_pages: int = Field(default=100, ge=1, le=200)
    data_dir: Path = Field(default=Path("data"))
    model_name: str = Field(default="gpt-4.1-mini", min_length=1)
    openai_base_url: str = Field(default="https://api.openai.com/v1", min_length=1)
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
