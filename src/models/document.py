"""
Document model for ClerkAI application.
"""

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, 
    DateTime, ForeignKey, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

from .base import BaseModel, AuditMixin, MetadataMixin


class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DocumentType(str, Enum):
    """Document type classification."""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    REPORT = "report"
    EMAIL = "email"
    FORM = "form"
    IMAGE = "image"
    OTHER = "other"


class ProcessingStatus(str, Enum):
    """Processing step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Document(BaseModel, AuditMixin, MetadataMixin):
    """Document model for file storage and processing."""
    
    __tablename__ = "documents"
    
    # Basic information
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    mime_type = Column(String(100), nullable=False)
    checksum = Column(String(64), nullable=False, index=True)  # SHA-256 hash
    
    # Classification
    document_type = Column(SQLEnum(DocumentType), default=DocumentType.OTHER, nullable=False)
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.UPLOADED, nullable=False)
    confidence_score = Column(Float, nullable=True)  # Classification confidence
    
    # Content
    title = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    extracted_text = Column(Text, nullable=True)
    
    # Processing information
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    processing_duration = Column(Float, nullable=True)  # Duration in seconds
    processing_steps = Column(JSONB, default=list, nullable=False)
    
    # OCR results
    ocr_text = Column(Text, nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    ocr_language = Column(String(10), nullable=True)
    
    # NLP results
    entities = Column(JSONB, default=list, nullable=False)
    keywords = Column(JSONB, default=list, nullable=False)
    summary = Column(Text, nullable=True)
    sentiment = Column(String(20), nullable=True)
    
    # Structured data extraction
    extracted_data = Column(JSONB, default=dict, nullable=False)
    validation_results = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    owner = relationship("User", back_populates="documents")
    
    workflow_executions = relationship(
        "WorkflowExecution", 
        back_populates="document",
        cascade="all, delete-orphan"
    )
    
    # Indexing and search
    search_vector = Column(Text, nullable=True)  # For full-text search
    tags = Column(JSONB, default=list, nullable=False)
    
    def __repr__(self):
        return f"<Document(filename={self.filename}, type={self.document_type})>"
    
    def start_processing(self) -> None:
        """Mark document as processing started."""
        self.status = DocumentStatus.PROCESSING
        self.processing_started_at = datetime.utcnow()
    
    def complete_processing(self, duration: Optional[float] = None) -> None:
        """Mark document as processing completed."""
        self.status = DocumentStatus.PROCESSED
        self.processing_completed_at = datetime.utcnow()
        
        if duration is None and self.processing_started_at:
            duration = (
                self.processing_completed_at - self.processing_started_at
            ).total_seconds()
        
        self.processing_duration = duration
    
    def fail_processing(self, error: str) -> None:
        """Mark document processing as failed."""
        self.status = DocumentStatus.FAILED
        self.processing_completed_at = datetime.utcnow()
        
        if self.processing_started_at:
            self.processing_duration = (
                self.processing_completed_at - self.processing_started_at
            ).total_seconds()
        
        # Add error to processing steps
        error_step = {
            "step": "error",
            "status": ProcessingStatus.FAILED.value,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.processing_steps is None:
            self.processing_steps = []
        self.processing_steps.append(error_step)
    
    def add_processing_step(self, step_name: str, status: ProcessingStatus, 
                          details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a processing step to the document.
        
        Args:
            step_name: Name of the processing step
            status: Status of the step
            details: Additional details about the step
        """
        step = {
            "step": step_name,
            "status": status.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            step.update(details)
        
        if self.processing_steps is None:
            self.processing_steps = []
        
        self.processing_steps.append(step)
    
    def get_processing_step(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific processing step.
        
        Args:
            step_name: Name of the step to retrieve
            
        Returns:
            Dict with step information or None if not found
        """
        if not self.processing_steps:
            return None
        
        for step in reversed(self.processing_steps):
            if step.get("step") == step_name:
                return step
        
        return None
    
    def add_entity(self, entity_type: str, entity_text: str, 
                   confidence: float, start_pos: int = None, 
                   end_pos: int = None) -> None:
        """
        Add an extracted entity to the document.
        
        Args:
            entity_type: Type of entity (e.g., 'PERSON', 'ORG', 'DATE')
            entity_text: The actual entity text
            confidence: Confidence score for the entity
            start_pos: Start position in the text
            end_pos: End position in the text
        """
        entity = {
            "type": entity_type,
            "text": entity_text,
            "confidence": confidence
        }
        
        if start_pos is not None and end_pos is not None:
            entity.update({"start": start_pos, "end": end_pos})
        
        if self.entities is None:
            self.entities = []
        
        self.entities.append(entity)
    
    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the document.
        
        Args:
            tag: Tag to add
        """
        if self.tags is None:
            self.tags = []
        
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """
        Remove a tag from the document.
        
        Args:
            tag: Tag to remove
        """
        if self.tags and tag in self.tags:
            self.tags.remove(tag)
    
    def set_extracted_data(self, key: str, value: Any) -> None:
        """
        Set extracted data field.
        
        Args:
            key: Data field key
            value: Data field value
        """
        if self.extracted_data is None:
            self.extracted_data = {}
        
        self.extracted_data[key] = value
    
    def get_extracted_data(self, key: str, default=None) -> Any:
        """
        Get extracted data field.
        
        Args:
            key: Data field key
            default: Default value if key not found
            
        Returns:
            Data field value or default
        """
        if not self.extracted_data:
            return default
        
        return self.extracted_data.get(key, default)
    
    @property
    def is_processed(self) -> bool:
        """Check if document processing is completed."""
        return self.status == DocumentStatus.PROCESSED
    
    @property
    def is_processing(self) -> bool:
        """Check if document is currently being processed."""
        return self.status == DocumentStatus.PROCESSING
    
    @property
    def has_failed(self) -> bool:
        """Check if document processing has failed."""
        return self.status == DocumentStatus.FAILED
    
    @property
    def file_extension(self) -> str:
        """Get file extension."""
        return self.original_filename.split('.')[-1].lower() if '.' in self.original_filename else ''
    
    def update_search_vector(self) -> None:
        """Update search vector for full-text search."""
        search_content = []
        
        if self.title:
            search_content.append(self.title)
        if self.description:
            search_content.append(self.description)
        if self.extracted_text:
            search_content.append(self.extracted_text)
        if self.ocr_text:
            search_content.append(self.ocr_text)
        if self.tags:
            search_content.extend(self.tags)
        
        self.search_vector = ' '.join(search_content)