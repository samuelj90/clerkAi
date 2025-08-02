"""
FastAPI dependencies for ClerkAI application.
"""

from typing import Optional, Generator
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt
import logging

from config.database import get_db
from config.settings import settings
from src.models.user import User

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()


def get_current_user(
    db: Session = Depends(get_db),
    token: HTTPAuthorizationCredentials = Security(security)
) -> User:
    """
    Get the current authenticated user from JWT token.
    
    Args:
        db: Database session
        token: Bearer token from authorization header
        
    Returns:
        User: Current authenticated user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token.credentials, 
            settings.secret_key, 
            algorithms=[settings.algorithm]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    
    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get the current active user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Current active user
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return current_user


def get_current_superuser(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get the current superuser.
    
    Args:
        current_user: Current active user
        
    Returns:
        User: Current superuser
        
    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    return current_user


def get_api_key_user(
    api_key: str,
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get user by API key.
    
    Args:
        api_key: API key from header
        db: Database session
        
    Returns:
        User or None if API key is invalid
    """
    if not api_key:
        return None
    
    user = db.query(User).filter(User.api_key == api_key).first()
    if user and user.is_active:
        return user
    
    return None


class OptionalAuth:
    """Dependency that provides optional authentication."""
    
    def __call__(
        self,
        db: Session = Depends(get_db),
        token: Optional[HTTPAuthorizationCredentials] = Security(security, auto_error=False),
        api_key: Optional[str] = None
    ) -> Optional[User]:
        """
        Get current user if authenticated, otherwise return None.
        
        Args:
            db: Database session
            token: Optional bearer token
            api_key: Optional API key
            
        Returns:
            User or None if not authenticated
        """
        # Try API key first
        if api_key:
            user = get_api_key_user(api_key, db)
            if user:
                return user
        
        # Try JWT token
        if token:
            try:
                payload = jwt.decode(
                    token.credentials,
                    settings.secret_key,
                    algorithms=[settings.algorithm]
                )
                username: str = payload.get("sub")
                if username:
                    user = db.query(User).filter(User.username == username).first()
                    if user and user.is_active:
                        return user
            except JWTError:
                pass
        
        return None


# Create instance of optional auth dependency
optional_auth = OptionalAuth()


def validate_pagination(
    skip: int = 0,
    limit: int = 100
) -> tuple[int, int]:
    """
    Validate pagination parameters.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        Tuple of validated skip and limit values
        
    Raises:
        HTTPException: If parameters are invalid
    """
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skip parameter must be non-negative"
        )
    
    if limit <= 0 or limit > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit parameter must be between 1 and 1000"
        )
    
    return skip, limit


def check_document_access(
    document_id: str,
    current_user: User,
    db: Session
) -> bool:
    """
    Check if user has access to a document.
    
    Args:
        document_id: Document ID to check
        current_user: Current user
        db: Database session
        
    Returns:
        True if user has access, False otherwise
    """
    from src.models.document import Document
    
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        return False
    
    # Owner or superuser can access
    return document.owner_id == current_user.id or current_user.is_superuser


def check_workflow_access(
    workflow_id: str,
    current_user: User,
    db: Session
) -> bool:
    """
    Check if user has access to a workflow.
    
    Args:
        workflow_id: Workflow ID to check
        current_user: Current user
        db: Database session
        
    Returns:
        True if user has access, False otherwise
    """
    from src.models.workflow import Workflow
    
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        return False
    
    # Creator or superuser can access
    return workflow.created_by_id == current_user.id or current_user.is_superuser