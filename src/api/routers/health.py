"""
Health check router for ClerkAI application.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from datetime import datetime
import psutil
import os

from config.database import get_db, DatabaseManager
from config.settings import settings
from src.api.schemas.base import BaseSchema

router = APIRouter()


class HealthResponse(BaseSchema):
    """Health check response schema."""
    
    status: str
    timestamp: datetime
    version: str
    environment: str
    uptime: float
    database: dict
    system: dict


class DetailedHealthResponse(BaseSchema):
    """Detailed health check response schema."""
    
    status: str
    timestamp: datetime
    version: str
    environment: str
    uptime: float
    database: dict
    system: dict
    services: dict
    features: dict


def get_system_info():
    """Get system information."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
    }


def get_database_health():
    """Get database health information."""
    try:
        is_healthy = DatabaseManager.health_check()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "connection": "ok" if is_healthy else "failed"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connection": "failed",
            "error": str(e)
        }


def get_uptime():
    """Get application uptime in seconds."""
    try:
        # This is a simple implementation; in production, you might want to track this differently
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
        return uptime_seconds
    except:
        return 0.0


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        HealthResponse: Basic health information
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        environment=settings.environment,
        uptime=get_uptime(),
        database=get_database_health(),
        system=get_system_info()
    )


@router.get("/live")
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint.
    
    Returns:
        Simple OK response for liveness checks
    """
    return {"status": "alive", "timestamp": datetime.utcnow()}


@router.get("/ready")
async def readiness_probe(db: Session = Depends(get_db)):
    """
    Kubernetes readiness probe endpoint.
    
    Args:
        db: Database session for health check
        
    Returns:
        Readiness status
    """
    # Check database connectivity
    db_healthy = DatabaseManager.health_check()
    
    if not db_healthy:
        return {
            "status": "not_ready",
            "reason": "database_unhealthy",
            "timestamp": datetime.utcnow()
        }
    
    return {
        "status": "ready",
        "timestamp": datetime.utcnow()
    }


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(db: Session = Depends(get_db)):
    """
    Detailed health check endpoint.
    
    Args:
        db: Database session
        
    Returns:
        DetailedHealthResponse: Comprehensive health information
    """
    # Check services
    services = {
        "database": get_database_health(),
        "redis": {"status": "unknown"},  # TODO: Implement Redis health check
        "ocr": {"status": "available" if settings.tesseract_cmd else "not_configured"},
        "llm": {"status": "configured" if settings.openai_api_key else "not_configured"}
    }
    
    # Check features
    features = {
        "document_processing": True,
        "ocr_enabled": bool(settings.tesseract_cmd),
        "nlp_enabled": True,
        "llm_integration": bool(settings.openai_api_key),
        "mcp_enabled": True,
        "crewai_enabled": True,
        "metrics_enabled": settings.enable_metrics
    }
    
    # Determine overall status
    overall_status = "healthy"
    if services["database"]["status"] != "healthy":
        overall_status = "unhealthy"
    
    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        environment=settings.environment,
        uptime=get_uptime(),
        database=services["database"],
        system=get_system_info(),
        services=services,
        features=features
    )