"""
Base schemas for ClerkAI API.
"""

from typing import Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from uuid import UUID


class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
        arbitrary_types_allowed=True
    )


class TimestampSchema(BaseSchema):
    """Schema with timestamp fields."""
    
    created_at: datetime
    updated_at: datetime


class UUIDSchema(BaseSchema):
    """Schema with UUID field."""
    
    id: UUID


class SoftDeleteSchema(BaseSchema):
    """Schema with soft delete field."""
    
    deleted_at: Optional[datetime] = None


class PaginationSchema(BaseSchema):
    """Schema for pagination parameters."""
    
    skip: int = 0
    limit: int = 100


class PaginatedResponse(BaseSchema):
    """Schema for paginated responses."""
    
    items: list[Any]
    total: int
    skip: int
    limit: int
    has_next: bool
    has_prev: bool


class ErrorSchema(BaseSchema):
    """Schema for error responses."""
    
    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: datetime


class SuccessResponse(BaseSchema):
    """Schema for success responses."""
    
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None