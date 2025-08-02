"""
Workflow model for ClerkAI application.
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


class WorkflowStatus(str, Enum):
    """Workflow status."""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class ExecutionStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StepStatus(str, Enum):
    """Workflow step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TriggerType(str, Enum):
    """Workflow trigger types."""
    MANUAL = "manual"
    DOCUMENT_UPLOAD = "document_upload"
    SCHEDULE = "schedule"
    API = "api"
    EMAIL = "email"


class Workflow(BaseModel, AuditMixin, MetadataMixin):
    """Workflow definition model."""
    
    __tablename__ = "workflows"
    
    # Basic information
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(20), default="1.0.0", nullable=False)
    status = Column(SQLEnum(WorkflowStatus), default=WorkflowStatus.DRAFT, nullable=False)
    
    # Configuration
    definition = Column(JSONB, nullable=False)  # Workflow definition (steps, conditions, etc.)
    input_schema = Column(JSONB, nullable=True)  # Expected input schema
    output_schema = Column(JSONB, nullable=True)  # Expected output schema
    
    # Triggers
    triggers = Column(JSONB, default=list, nullable=False)  # Trigger configurations
    
    # Settings
    timeout_seconds = Column(Integer, default=3600, nullable=False)  # 1 hour default
    max_retries = Column(Integer, default=3, nullable=False)
    retry_delay_seconds = Column(Integer, default=60, nullable=False)
    
    # Statistics
    execution_count = Column(Integer, default=0, nullable=False)
    success_count = Column(Integer, default=0, nullable=False)
    failure_count = Column(Integer, default=0, nullable=False)
    average_duration = Column(Float, nullable=True)
    
    # Relationships
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    created_by = relationship("User", back_populates="workflows")
    
    executions = relationship(
        "WorkflowExecution",
        back_populates="workflow",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    def __repr__(self):
        return f"<Workflow(name={self.name}, version={self.version})>"
    
    def activate(self) -> None:
        """Activate the workflow."""
        self.status = WorkflowStatus.ACTIVE
    
    def deactivate(self) -> None:
        """Deactivate the workflow."""
        self.status = WorkflowStatus.INACTIVE
    
    def archive(self) -> None:
        """Archive the workflow."""
        self.status = WorkflowStatus.ARCHIVED
    
    def add_trigger(self, trigger_type: TriggerType, config: Dict[str, Any]) -> None:
        """
        Add a trigger to the workflow.
        
        Args:
            trigger_type: Type of trigger
            config: Trigger configuration
        """
        trigger = {
            "type": trigger_type.value,
            "config": config,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if self.triggers is None:
            self.triggers = []
        
        self.triggers.append(trigger)
    
    def remove_trigger(self, trigger_type: TriggerType) -> bool:
        """
        Remove a trigger from the workflow.
        
        Args:
            trigger_type: Type of trigger to remove
            
        Returns:
            bool: True if trigger was removed, False if not found
        """
        if not self.triggers:
            return False
        
        original_length = len(self.triggers)
        self.triggers = [
            t for t in self.triggers 
            if t.get("type") != trigger_type.value
        ]
        
        return len(self.triggers) < original_length
    
    def update_statistics(self, execution: 'WorkflowExecution') -> None:
        """
        Update workflow statistics based on execution.
        
        Args:
            execution: Completed workflow execution
        """
        self.execution_count += 1
        
        if execution.status == ExecutionStatus.COMPLETED:
            self.success_count += 1
        elif execution.status in [ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT]:
            self.failure_count += 1
        
        # Update average duration
        if execution.duration:
            if self.average_duration is None:
                self.average_duration = execution.duration
            else:
                # Moving average
                self.average_duration = (
                    (self.average_duration * (self.execution_count - 1) + execution.duration) 
                    / self.execution_count
                )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.execution_count == 0:
            return 0.0
        return (self.success_count / self.execution_count) * 100
    
    @property
    def is_active(self) -> bool:
        """Check if workflow is active."""
        return self.status == WorkflowStatus.ACTIVE


class WorkflowExecution(BaseModel, AuditMixin):
    """Workflow execution instance."""
    
    __tablename__ = "workflow_executions"
    
    # Basic information
    execution_id = Column(String(50), unique=True, nullable=False, index=True)
    status = Column(SQLEnum(ExecutionStatus), default=ExecutionStatus.PENDING, nullable=False)
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)  # Duration in seconds
    
    # Input/Output
    input_data = Column(JSONB, nullable=True)
    output_data = Column(JSONB, nullable=True)
    error_details = Column(JSONB, nullable=True)
    
    # Progress tracking
    current_step = Column(Integer, default=0, nullable=False)
    total_steps = Column(Integer, default=0, nullable=False)
    steps = Column(JSONB, default=list, nullable=False)
    
    # Configuration
    retry_count = Column(Integer, default=0, nullable=False)
    timeout_at = Column(DateTime, nullable=True)
    
    # Context
    context = Column(JSONB, default=dict, nullable=False)  # Execution context
    
    # Relationships
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflows.id"), nullable=False, index=True)
    workflow = relationship("Workflow", back_populates="executions")
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    user = relationship("User", back_populates="workflow_executions")
    
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True, index=True)
    document = relationship("Document", back_populates="workflow_executions")
    
    def __repr__(self):
        return f"<WorkflowExecution(id={self.execution_id}, status={self.status})>"
    
    def start(self) -> None:
        """Start the workflow execution."""
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()
        
        # Set timeout
        if self.workflow and self.workflow.timeout_seconds:
            from datetime import timedelta
            self.timeout_at = self.started_at + timedelta(seconds=self.workflow.timeout_seconds)
    
    def complete(self, output_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Complete the workflow execution.
        
        Args:
            output_data: Output data from the execution
        """
        self.status = ExecutionStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        
        if output_data:
            self.output_data = output_data
        
        # Calculate duration
        if self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()
        
        # Update workflow statistics
        if self.workflow:
            self.workflow.update_statistics(self)
    
    def fail(self, error: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark the workflow execution as failed.
        
        Args:
            error: Error message
            error_details: Additional error details
        """
        self.status = ExecutionStatus.FAILED
        self.completed_at = datetime.utcnow()
        
        if self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()
        
        # Store error details
        self.error_details = {
            "error": error,
            "timestamp": self.completed_at.isoformat()
        }
        
        if error_details:
            self.error_details.update(error_details)
        
        # Update workflow statistics
        if self.workflow:
            self.workflow.update_statistics(self)
    
    def cancel(self) -> None:
        """Cancel the workflow execution."""
        self.status = ExecutionStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        
        if self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()
    
    def timeout(self) -> None:
        """Mark the workflow execution as timed out."""
        self.status = ExecutionStatus.TIMEOUT
        self.completed_at = datetime.utcnow()
        
        if self.started_at:
            self.duration = (self.completed_at - self.started_at).total_seconds()
        
        # Store timeout details
        self.error_details = {
            "error": "Workflow execution timed out",
            "timeout_at": self.timeout_at.isoformat() if self.timeout_at else None,
            "timestamp": self.completed_at.isoformat()
        }
        
        # Update workflow statistics
        if self.workflow:
            self.workflow.update_statistics(self)
    
    def add_step(self, step_name: str, status: StepStatus = StepStatus.PENDING,
                 details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a step to the execution.
        
        Args:
            step_name: Name of the step
            status: Status of the step
            details: Additional step details
        """
        step = {
            "name": step_name,
            "status": status.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            step.update(details)
        
        if self.steps is None:
            self.steps = []
        
        self.steps.append(step)
        self.total_steps = len(self.steps)
    
    def update_step(self, step_name: str, status: StepStatus,
                   details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a step status.
        
        Args:
            step_name: Name of the step to update
            status: New status
            details: Additional details to add
            
        Returns:
            bool: True if step was found and updated, False otherwise
        """
        if not self.steps:
            return False
        
        for i, step in enumerate(self.steps):
            if step.get("name") == step_name:
                step["status"] = status.value
                step["updated_at"] = datetime.utcnow().isoformat()
                
                if details:
                    step.update(details)
                
                # Update current step if this step is completed
                if status == StepStatus.COMPLETED:
                    self.current_step = i + 1
                
                return True
        
        return False
    
    def get_step(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Dict with step information or None if not found
        """
        if not self.steps:
            return None
        
        for step in self.steps:
            if step.get("name") == step_name:
                return step
        
        return None
    
    @property
    def is_running(self) -> bool:
        """Check if execution is running."""
        return self.status == ExecutionStatus.RUNNING
    
    @property
    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return self.status in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.TIMEOUT
        ]
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100
    
    def set_context(self, key: str, value: Any) -> None:
        """
        Set context value.
        
        Args:
            key: Context key
            value: Context value
        """
        if self.context is None:
            self.context = {}
        
        self.context[key] = value
    
    def get_context(self, key: str, default=None) -> Any:
        """
        Get context value.
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Context value or default
        """
        if not self.context:
            return default
        
        return self.context.get(key, default)