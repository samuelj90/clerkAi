"""
Logging configuration for ClerkAI application.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any
import structlog
from rich.logging import RichHandler

from .settings import settings


def setup_logging() -> None:
    """Configure logging for the application."""
    
    # Create logs directory if it doesn't exist
    log_file_path = Path(settings.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with Rich formatting for development
    if settings.environment == "development":
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=True,
        )
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    else:
        # Simple console handler for production
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, settings.log_level.upper()))
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file,
        maxBytes=settings.log_max_size,
        backupCount=settings.log_backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    configure_loggers()


def configure_loggers() -> None:
    """Configure specific loggers with appropriate levels."""
    
    logger_configs = {
        "uvicorn": logging.INFO,
        "uvicorn.error": logging.INFO,
        "uvicorn.access": logging.WARNING if settings.environment == "production" else logging.INFO,
        "sqlalchemy.engine": logging.WARNING,
        "sqlalchemy.pool": logging.WARNING,
        "httpx": logging.WARNING,
        "openai": logging.WARNING,
        "transformers": logging.WARNING,
        "celery": logging.INFO,
        "redis": logging.WARNING,
    }
    
    for logger_name, level in logger_configs.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        BoundLogger: Structured logger instance
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capability to any class."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Return a logger instance for this class."""
        return get_logger(self.__class__.__name__)


# Performance monitoring decorator
def log_performance(func_name: str = None):
    """
    Decorator to log function execution time.
    
    Args:
        func_name: Optional custom function name for logging
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = get_logger(func_name or func.__name__)
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    "Function executed successfully",
                    function=func.__name__,
                    execution_time=f"{execution_time:.3f}s"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    "Function execution failed",
                    function=func.__name__,
                    execution_time=f"{execution_time:.3f}s",
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator