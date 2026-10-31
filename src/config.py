"""
Configuration module for Multimodal Heart Disease Framework.
Uses Pydantic Settings to load configurations from environment variables or .env file.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_ENV: str = "development"
    APP_NAME: str = "Multimodal Heart Disease Research Platform"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    JWT_SECRET: str = "super-secret-key-heart-disease-multimodal-research-2026-secure"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_MINUTES: int = 60

    MODEL_PATH: str = str(BASE_DIR / "artifacts" / "best_multimodal_model.pt")
    MODEL_VERSION: str = "v1.0.0-multimodal-gated"
    MAX_ECG_SAMPLES: int = 20000

    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'heart_disease_research.db'}"
    LOG_LEVEL: str = "INFO"

    # Default physiological sampling params
    DEFAULT_FS: float = 500.0
    MIN_ECG_SAMPLES: int = 100
    NOMINAL_LEADS: int = 12

settings = Settings()
