from pydantic_settings import BaseSettings
from pydantic import Field
import os
import logging


logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    google_maps_api_key: str = Field(..., env='GOOGLE_MAPS_API_KEY')
    google_cloud_secret: str = Field(..., env='GOOGLE_CLOUD_SECRET')
    openai_api_key: str = Field(..., env='OPENAI_API_KEY')
    database_url: str = Field(..., env='DATABASE_URL')
    api_ninjas_key: str = Field(..., env='API_NINJAS_KEY')
    exploding_topics_api_key: str = Field(..., env='EXPLODING_TOPICS_API_KEY')
    vite_openai_api_key: str = Field(None, env='VITE_OPENAI_API_KEY')
    secret_key: str = Field(None, env='SECRET_KEY')
    debug: bool = Field(False, env='DEBUG')

    allowed_hosts: list[str] = ["*"]
    middleware: list[str] = [
        'django.middleware.security.SecurityMiddleware',
        'whitenoise.middleware.WhiteNoiseMiddleware',
    ]
    staticfiles_storage: str = "whitenoise.storage.CompressedManifestStaticFilesStorage"
    static_url: str = "/static/"
    static_root: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "staticfiles")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

try:
    settings = Settings()
    logger.info("Configuration settings loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load settings: {e}")
    raise
