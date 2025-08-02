"""
Base model classes and mixins for ClerkAI database models.
"""

from datetime import datetime
from typing import Any, Dict
from sqlalchemy import Column, Integer, DateTime, String
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import declarative_mixin
from sqlalchemy.dialects.postgresql import UUID
import uuid

from config.database import Base


@declarative_mixin
class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps."""
    
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )
    
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        index=True
    )


@declarative_mixin
class UUIDMixin:
    """Mixin to add UUID primary key."""
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )


@declarative_mixin
class SoftDeleteMixin:
    """Mixin to add soft delete functionality."""
    
    deleted_at = Column(DateTime, nullable=True, index=True)
    
    @property
    def is_deleted(self) -> bool:
        """Check if the record is soft deleted."""
        return self.deleted_at is not None
    
    def soft_delete(self) -> None:
        """Mark the record as deleted."""
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore a soft deleted record."""
        self.deleted_at = None


class BaseModel(Base, TimestampMixin, UUIDMixin, SoftDeleteMixin):
    """Base model class with common functionality."""
    
    __abstract__ = True
    
    def to_dict(self, exclude: set = None) -> Dict[str, Any]:
        """
        Convert model instance to dictionary.
        
        Args:
            exclude: Set of column names to exclude
            
        Returns:
            Dict representation of the model
        """
        exclude = exclude or set()
        result = {}
        
        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, uuid.UUID):
                    value = str(value)
                result[column.name] = value
        
        return result
    
    def update_from_dict(self, data: Dict[str, Any], exclude: set = None) -> None:
        """
        Update model instance from dictionary.
        
        Args:
            data: Dictionary with new values
            exclude: Set of column names to exclude from update
        """
        exclude = exclude or {'id', 'created_at'}
        
        for key, value in data.items():
            if key not in exclude and hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def get_table_name(cls) -> str:
        """Get the table name for this model."""
        return cls.__tablename__
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"<{self.__class__.__name__}(id={self.id})>"


class AuditMixin:
    """Mixin to add audit fields for tracking changes."""
    
    @declared_attr
    def created_by_id(cls):
        return Column(UUID(as_uuid=True), nullable=True, index=True)
    
    @declared_attr
    def updated_by_id(cls):
        return Column(UUID(as_uuid=True), nullable=True, index=True)
    
    @declared_attr
    def version(cls):
        return Column(Integer, default=1, nullable=False)


class MetadataMixin:
    """Mixin to add JSON metadata field."""
    
    @declared_attr
    def metadata_(cls):
        from sqlalchemy.dialects.postgresql import JSONB
        from sqlalchemy.dialects.sqlite import JSON
        from sqlalchemy import JSON as GenericJSON
        
        # Use JSONB for PostgreSQL, JSON for SQLite, generic JSON for others
        return Column(
            JSONB if 'postgresql' in str(cls.__table__.bind) 
            else JSON if 'sqlite' in str(cls.__table__.bind)
            else GenericJSON,
            default=dict,
            nullable=False
        )