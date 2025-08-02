"""
ClerkAI - Main FastAPI application entry point.

This module sets up the FastAPI application with all necessary middleware,
routers, and lifecycle events.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import time
import logging

from config.settings import settings
from config.database import init_db, close_db
from config.logging import setup_logging, get_logger

# Import routers
from src.api.routers import documents, workflows, reports, health

# Setup logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting ClerkAI application", version=settings.app_version)
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized successfully")
        
        # Initialize other services here
        # - Load MCP configurations
        # - Initialize CrewAI workflows
        # - Setup background tasks
        
        yield
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down ClerkAI application")
        await close_db()
        logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="AI-Powered Office Assistant with Model Context Protocol (MCP)",
    version=settings.app_version,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# Add trusted host middleware
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.yourdomain.com", "yourdomain.com"]
    )


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log request details
    logger.info(
        "Request processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=f"{process_time:.3f}s"
    )
    
    return response


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        url=str(request.url),
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(
        "Validation error occurred",
        errors=exc.errors(),
        url=str(request.url),
        method=request.method
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation failed",
            "details": exc.errors(),
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(
        "Unexpected error occurred",
        error=str(exc),
        error_type=type(exc).__name__,
        url=str(request.url),
        method=request.method
    )
    
    if settings.environment == "development":
        import traceback
        error_detail = {
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc()
        }
    else:
        error_detail = {
            "error": "Internal server error",
            "type": "InternalServerError"
        }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            **error_detail,
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "timestamp": time.time()
        }
    )


# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["Workflows"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with application information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "AI-Powered Office Assistant with Model Context Protocol",
        "environment": settings.environment,
        "docs_url": "/docs" if settings.environment != "production" else None,
        "health_check": "/health"
    }


# Application metadata
@app.get("/info")
async def app_info():
    """Get application information and status."""
    return {
        "application": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "debug": settings.debug
        },
        "features": {
            "document_processing": True,
            "ocr_enabled": True,
            "nlp_enabled": True,
            "llm_integration": bool(settings.openai_api_key),
            "mcp_enabled": True,
            "crewai_enabled": True
        },
        "endpoints": {
            "documents": "/api/v1/documents",
            "workflows": "/api/v1/workflows",
            "reports": "/api/v1/reports",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting ClerkAI development server")
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_config=None  # We handle logging ourselves
    )