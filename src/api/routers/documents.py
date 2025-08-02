"""
Document management API router for ClerkAI.
"""

from typing import List, Optional
from fastapi import (
    APIRouter, Depends, HTTPException, status, 
    UploadFile, File, Form, Query
)
from sqlalchemy.orm import Session
from uuid import UUID
import os
import hashlib
import shutil
from datetime import datetime

from config.database import get_db
from config.settings import settings
from src.api.dependencies import (
    get_current_active_user, validate_pagination,
    check_document_access
)
from src.api.schemas.base import BaseSchema, PaginatedResponse, SuccessResponse
from src.models.user import User
from src.models.document import Document, DocumentStatus, DocumentType
from config.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class DocumentResponse(BaseSchema):
    """Document response schema."""
    
    id: UUID
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    document_type: DocumentType
    status: DocumentStatus
    confidence_score: Optional[float] = None
    title: Optional[str] = None
    description: Optional[str] = None
    extracted_text: Optional[str] = None
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    processing_duration: Optional[float] = None
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    entities: List[dict] = []
    keywords: List[str] = []
    summary: Optional[str] = None
    sentiment: Optional[str] = None
    extracted_data: dict = {}
    tags: List[str] = []
    created_at: datetime
    updated_at: datetime
    owner_id: UUID


class DocumentCreate(BaseSchema):
    """Document creation schema."""
    
    document_type: Optional[DocumentType] = DocumentType.OTHER
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = []


class DocumentUpdate(BaseSchema):
    """Document update schema."""
    
    title: Optional[str] = None
    description: Optional[str] = None
    document_type: Optional[DocumentType] = None
    tags: Optional[List[str]] = None


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_unique_filename(original_filename: str, upload_dir: str) -> str:
    """Generate a unique filename to avoid conflicts."""
    name, ext = os.path.splitext(original_filename)
    counter = 1
    filename = original_filename
    
    while os.path.exists(os.path.join(upload_dir, filename)):
        filename = f"{name}_{counter}{ext}"
        counter += 1
    
    return filename


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    document_type: DocumentType = Form(DocumentType.OTHER),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # Comma-separated tags
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Upload a new document.
    
    Args:
        file: File to upload
        document_type: Type of document
        title: Optional document title
        description: Optional document description
        tags: Optional comma-separated tags
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        DocumentResponse: Created document information
        
    Raises:
        HTTPException: If file is invalid or upload fails
    """
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.allowed_file_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not allowed. Allowed types: {settings.allowed_file_types}"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_file_size / (1024*1024):.1f}MB"
        )
    
    try:
        # Generate unique filename
        unique_filename = get_unique_filename(file.filename, settings.upload_dir)
        file_path = os.path.join(settings.upload_dir, unique_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Calculate file hash
        file_hash = calculate_file_hash(file_path)
        
        # Check for duplicate files
        existing_doc = db.query(Document).filter(Document.checksum == file_hash).first()
        if existing_doc:
            # Remove the uploaded file since it's a duplicate
            os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Document with same content already exists: {existing_doc.id}"
            )
        
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Create document record
        document = Document(
            filename=unique_filename,
            original_filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=file.content_type or "application/octet-stream",
            checksum=file_hash,
            document_type=document_type,
            title=title,
            description=description,
            tags=tag_list,
            owner_id=current_user.id
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        logger.info(
            "Document uploaded successfully",
            document_id=str(document.id),
            filename=file.filename,
            user_id=str(current_user.id)
        )
        
        # TODO: Trigger document processing workflow
        
        return DocumentResponse.model_validate(document)
        
    except Exception as e:
        # Clean up file if database operation fails
        if os.path.exists(file_path):
            os.remove(file_path)
        
        logger.error(
            "Document upload failed",
            filename=file.filename,
            user_id=str(current_user.id),
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document"
        )


@router.get("/", response_model=PaginatedResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    document_type: Optional[DocumentType] = Query(None),
    status: Optional[DocumentStatus] = Query(None),
    search: Optional[str] = Query(None),
    tags: Optional[str] = Query(None),  # Comma-separated tags
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List documents for the current user.
    
    Args:
        skip: Number of documents to skip
        limit: Maximum number of documents to return
        document_type: Filter by document type
        status: Filter by document status
        search: Search in title, description, and extracted text
        tags: Filter by comma-separated tags
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        PaginatedResponse: List of documents with pagination info
    """
    skip, limit = validate_pagination(skip, limit)
    
    # Build query
    query = db.query(Document).filter(Document.owner_id == current_user.id)
    
    # Apply filters
    if document_type:
        query = query.filter(Document.document_type == document_type)
    
    if status:
        query = query.filter(Document.status == status)
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Document.title.ilike(search_term)) |
            (Document.description.ilike(search_term)) |
            (Document.extracted_text.ilike(search_term)) |
            (Document.ocr_text.ilike(search_term))
        )
    
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        for tag in tag_list:
            query = query.filter(Document.tags.contains([tag]))
    
    # Get total count
    total = query.count()
    
    # Apply pagination and ordering
    documents = query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
    
    return PaginatedResponse(
        items=[DocumentResponse.model_validate(doc) for doc in documents],
        total=total,
        skip=skip,
        limit=limit,
        has_next=skip + limit < total,
        has_prev=skip > 0
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific document.
    
    Args:
        document_id: Document UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        DocumentResponse: Document information
        
    Raises:
        HTTPException: If document not found or access denied
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check access permissions
    if not check_document_access(str(document_id), current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return DocumentResponse.model_validate(document)


@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: UUID,
    document_update: DocumentUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update a document.
    
    Args:
        document_id: Document UUID
        document_update: Document update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        DocumentResponse: Updated document information
        
    Raises:
        HTTPException: If document not found or access denied
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check access permissions
    if not check_document_access(str(document_id), current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Update document
    update_data = document_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(document, field, value)
    
    db.commit()
    db.refresh(document)
    
    logger.info(
        "Document updated",
        document_id=str(document_id),
        user_id=str(current_user.id)
    )
    
    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", response_model=SuccessResponse)
async def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete a document.
    
    Args:
        document_id: Document UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        SuccessResponse: Deletion confirmation
        
    Raises:
        HTTPException: If document not found or access denied
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check access permissions
    if not check_document_access(str(document_id), current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    try:
        # Remove file from filesystem
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Soft delete document
        document.soft_delete()
        db.commit()
        
        logger.info(
            "Document deleted",
            document_id=str(document_id),
            user_id=str(current_user.id)
        )
        
        return SuccessResponse(
            message="Document deleted successfully"
        )
        
    except Exception as e:
        logger.error(
            "Document deletion failed",
            document_id=str(document_id),
            user_id=str(current_user.id),
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )


@router.post("/{document_id}/reprocess", response_model=SuccessResponse)
async def reprocess_document(
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Trigger document reprocessing.
    
    Args:
        document_id: Document UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        SuccessResponse: Reprocessing confirmation
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check access permissions
    if not check_document_access(str(document_id), current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Reset document status for reprocessing
    document.status = DocumentStatus.UPLOADED
    document.processing_started_at = None
    document.processing_completed_at = None
    document.processing_duration = None
    document.processing_steps = []
    
    db.commit()
    
    # TODO: Trigger document processing workflow
    
    logger.info(
        "Document reprocessing triggered",
        document_id=str(document_id),
        user_id=str(current_user.id)
    )
    
    return SuccessResponse(
        message="Document reprocessing started"
    )