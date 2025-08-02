"""
Configuration settings for ClerkAI application.
Uses Pydantic Settings for environment variable management.
"""

from typing import Optional, List
from pydantic import BaseSettings, validator
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "ClerkAI"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # Database
    database_url: str = "postgresql://clerkuser:clerkpass@localhost:5432/clerkai"
    database_echo: bool = False
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Security
    secret_key: str = "your-secret-key-here-change-in-production"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"
    
    # CORS
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    allowed_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allowed_headers: List[str] = ["*"]
    
    # File Upload
    upload_dir: str = "uploads"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".docx"]
    
    # OCR Settings
    tesseract_cmd: Optional[str] = None
    ocr_language: str = "eng"
    
    # LLM Settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    max_tokens: int = 2000
    temperature: float = 0.7
    
    # Model Context Protocol
    mcp_config_file: str = "model_context.json"
    mcp_auto_reload: bool = True
    
    # CrewAI Settings
    crew_max_workers: int = 4
    crew_timeout: int = 300
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/clerkai.log"
    log_max_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Email (for notifications)
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        if v not in ["development", "testing", "production"]:
            raise ValueError("Environment must be development, testing, or production")
        return v
    
    @validator("upload_dir")
    def create_upload_dir(cls, v):
        """Create upload directory if it doesn't exist."""
        os.makedirs(v, exist_ok=True)
        return v
    
    @validator("log_file")
    def create_log_dir(cls, v):
        """Create log directory if it doesn't exist."""
        log_dir = os.path.dirname(v)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create global settings instance
settings = Settings()