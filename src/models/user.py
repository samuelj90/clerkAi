"""
User model for ClerkAI application.
"""

from sqlalchemy import Column, String, Boolean, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from typing import Optional

from .base import BaseModel, AuditMixin, MetadataMixin


class User(BaseModel, AuditMixin, MetadataMixin):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    # Basic information
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Profile information
    avatar_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    department = Column(String(100), nullable=True)
    job_title = Column(String(100), nullable=True)
    
    # Account settings
    email_verified = Column(Boolean, default=False, nullable=False)
    email_verified_at = Column(DateTime, nullable=True)
    last_login = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)
    
    # Preferences
    preferences = Column(JSONB, default=dict, nullable=False)
    
    # API settings
    api_key = Column(String(255), nullable=True, unique=True, index=True)
    api_key_created_at = Column(DateTime, nullable=True)
    
    # Relationships
    documents = relationship("Document", back_populates="owner", lazy="dynamic")
    workflows = relationship("Workflow", back_populates="created_by", lazy="dynamic")
    workflow_executions = relationship("WorkflowExecution", back_populates="user", lazy="dynamic")
    
    def __repr__(self):
        return f"<User(username={self.username}, email={self.email})>"
    
    def check_password(self, password: str) -> bool:
        """
        Check if provided password matches the hashed password.
        
        Args:
            password: Plain text password to check
            
        Returns:
            bool: True if password matches, False otherwise
        """
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.verify(password, self.hashed_password)
    
    def set_password(self, password: str) -> None:
        """
        Set password for the user.
        
        Args:
            password: Plain text password to hash and store
        """
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.hashed_password = pwd_context.hash(password)
    
    def verify_email(self) -> None:
        """Mark email as verified."""
        self.email_verified = True
        self.email_verified_at = datetime.utcnow()
    
    def record_login(self) -> None:
        """Record successful login."""
        self.last_login = datetime.utcnow()
        self.failed_login_attempts = 0
        self.locked_until = None
    
    def record_failed_login(self) -> None:
        """Record failed login attempt."""
        self.failed_login_attempts += 1
        
        # Lock account after 5 failed attempts for 30 minutes
        if self.failed_login_attempts >= 5:
            from datetime import timedelta
            self.locked_until = datetime.utcnow() + timedelta(minutes=30)
    
    @property
    def is_locked(self) -> bool:
        """Check if account is locked."""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until
    
    def generate_api_key(self) -> str:
        """
        Generate a new API key for the user.
        
        Returns:
            str: The generated API key
        """
        import secrets
        api_key = f"clerkai_{''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(32))}"
        self.api_key = api_key
        self.api_key_created_at = datetime.utcnow()
        return api_key
    
    def revoke_api_key(self) -> None:
        """Revoke the current API key."""
        self.api_key = None
        self.api_key_created_at = None
    
    def get_preference(self, key: str, default=None):
        """
        Get a user preference value.
        
        Args:
            key: Preference key
            default: Default value if key not found
            
        Returns:
            Preference value or default
        """
        return self.preferences.get(key, default)
    
    def set_preference(self, key: str, value) -> None:
        """
        Set a user preference value.
        
        Args:
            key: Preference key
            value: Preference value
        """
        if self.preferences is None:
            self.preferences = {}
        self.preferences[key] = value
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        """
        Convert user to dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive information
            
        Returns:
            dict: User dictionary representation
        """
        exclude = {'hashed_password'}
        if not include_sensitive:
            exclude.update({'api_key', 'failed_login_attempts', 'locked_until'})
        
        return super().to_dict(exclude=exclude)