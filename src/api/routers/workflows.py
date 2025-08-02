"""
Workflow management API router for ClerkAI.
"""

from typing import List, Optional, Dict, Any
from fastapi import (
    APIRouter, Depends, HTTPException, status, Query
)
from sqlalchemy.orm import Session
from uuid import UUID
from datetime import datetime

from config.database import get_db
from src.api.dependencies import (
    get_current_active_user, validate_pagination,
    check_workflow_access
)
from src.api.schemas.base import BaseSchema, PaginatedResponse, SuccessResponse
from src.models.user import User
from src.models.workflow import (
    Workflow, WorkflowStatus, WorkflowExecution, 
    ExecutionStatus, TriggerType
)
from config.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class WorkflowResponse(BaseSchema):
    """Workflow response schema."""
    
    id: UUID
    name: str
    description: Optional[str] = None
    version: str
    status: WorkflowStatus
    definition: Dict[str, Any]
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    triggers: List[Dict[str, Any]] = []
    timeout_seconds: int
    max_retries: int
    retry_delay_seconds: int
    execution_count: int
    success_count: int
    failure_count: int
    average_duration: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    created_by_id: UUID


class WorkflowCreate(BaseSchema):
    """Workflow creation schema."""
    
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    definition: Dict[str, Any]
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 3600
    max_retries: int = 3
    retry_delay_seconds: int = 60


class WorkflowUpdate(BaseSchema):
    """Workflow update schema."""
    
    name: Optional[str] = None
    description: Optional[str] = None
    definition: Optional[Dict[str, Any]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    retry_delay_seconds: Optional[int] = None


class WorkflowExecutionResponse(BaseSchema):
    """Workflow execution response schema."""
    
    id: UUID
    execution_id: str
    status: ExecutionStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None
    current_step: int
    total_steps: int
    steps: List[Dict[str, Any]] = []
    retry_count: int
    context: Dict[str, Any] = {}
    workflow_id: UUID
    user_id: Optional[UUID] = None
    document_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime


class WorkflowExecutionCreate(BaseSchema):
    """Workflow execution creation schema."""
    
    input_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    document_id: Optional[UUID] = None


class TriggerCreate(BaseSchema):
    """Trigger creation schema."""
    
    trigger_type: TriggerType
    config: Dict[str, Any]


@router.post("/", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(
    workflow_data: WorkflowCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new workflow.
    
    Args:
        workflow_data: Workflow creation data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        WorkflowResponse: Created workflow information
    """
    # Check if workflow name already exists for this user
    existing_workflow = db.query(Workflow).filter(
        Workflow.name == workflow_data.name,
        Workflow.created_by_id == current_user.id
    ).first()
    
    if existing_workflow:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Workflow with this name already exists"
        )
    
    # Create workflow
    workflow = Workflow(
        **workflow_data.model_dump(),
        created_by_id=current_user.id
    )
    
    db.add(workflow)
    db.commit()
    db.refresh(workflow)
    
    logger.info(
        "Workflow created",
        workflow_id=str(workflow.id),
        name=workflow.name,
        user_id=str(current_user.id)
    )
    
    return WorkflowResponse.model_validate(workflow)


@router.get("/", response_model=PaginatedResponse)
async def list_workflows(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[WorkflowStatus] = Query(None),
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List workflows for the current user.
    
    Args:
        skip: Number of workflows to skip
        limit: Maximum number of workflows to return
        status: Filter by workflow status
        search: Search in name and description
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        PaginatedResponse: List of workflows with pagination info
    """
    skip, limit = validate_pagination(skip, limit)
    
    # Build query
    query = db.query(Workflow).filter(Workflow.created_by_id == current_user.id)
    
    # Apply filters
    if status:
        query = query.filter(Workflow.status == status)
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Workflow.name.ilike(search_term)) |
            (Workflow.description.ilike(search_term))
        )
    
    # Get total count
    total = query.count()
    
    # Apply pagination and ordering
    workflows = query.order_by(Workflow.created_at.desc()).offset(skip).limit(limit).all()
    
    return PaginatedResponse(
        items=[WorkflowResponse.model_validate(workflow) for workflow in workflows],
        total=total,
        skip=skip,
        limit=limit,
        has_next=skip + limit < total,
        has_prev=skip > 0
    )


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific workflow.
    
    Args:
        workflow_id: Workflow UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        WorkflowResponse: Workflow information
    """
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    # Check access permissions
    if not check_workflow_access(str(workflow_id), current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return WorkflowResponse.model_validate(workflow)


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: UUID,
    workflow_update: WorkflowUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update a workflow.
    
    Args:
        workflow_id: Workflow UUID
        workflow_update: Workflow update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        WorkflowResponse: Updated workflow information
    """
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    # Check access permissions
    if not check_workflow_access(str(workflow_id), current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Update workflow
    update_data = workflow_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(workflow, field, value)
    
    db.commit()
    db.refresh(workflow)
    
    logger.info(
        "Workflow updated",
        workflow_id=str(workflow_id),
        user_id=str(current_user.id)
    )
    
    return WorkflowResponse.model_validate(workflow)


@router.delete("/{workflow_id}", response_model=SuccessResponse)
async def delete_workflow(
    workflow_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete a workflow.
    
    Args:
        workflow_id: Workflow UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        SuccessResponse: Deletion confirmation
    """
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    # Check access permissions
    if not check_workflow_access(str(workflow_id), current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Soft delete workflow
    workflow.soft_delete()
    db.commit()
    
    logger.info(
        "Workflow deleted",
        workflow_id=str(workflow_id),
        user_id=str(current_user.id)
    )
    
    return SuccessResponse(
        message="Workflow deleted successfully"
    )


@router.post("/{workflow_id}/activate", response_model=SuccessResponse)
async def activate_workflow(
    workflow_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Activate a workflow.
    
    Args:
        workflow_id: Workflow UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        SuccessResponse: Activation confirmation
    """
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    # Check access permissions
    if not check_workflow_access(str(workflow_id), current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    workflow.activate()
    db.commit()
    
    logger.info(
        "Workflow activated",
        workflow_id=str(workflow_id),
        user_id=str(current_user.id)
    )
    
    return SuccessResponse(
        message="Workflow activated successfully"
    )


@router.post("/{workflow_id}/deactivate", response_model=SuccessResponse)
async def deactivate_workflow(
    workflow_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Deactivate a workflow.
    
    Args:
        workflow_id: Workflow UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        SuccessResponse: Deactivation confirmation
    """
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    # Check access permissions
    if not check_workflow_access(str(workflow_id), current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    workflow.deactivate()
    db.commit()
    
    logger.info(
        "Workflow deactivated",
        workflow_id=str(workflow_id),
        user_id=str(current_user.id)
    )
    
    return SuccessResponse(
        message="Workflow deactivated successfully"
    )


@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse, status_code=status.HTTP_201_CREATED)
async def execute_workflow(
    workflow_id: UUID,
    execution_data: WorkflowExecutionCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Execute a workflow.
    
    Args:
        workflow_id: Workflow UUID
        execution_data: Execution input data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        WorkflowExecutionResponse: Execution information
    """
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    # Check access permissions
    if not check_workflow_access(str(workflow_id), current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Check if workflow is active
    if not workflow.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow is not active"
        )
    
    # Generate execution ID
    import uuid
    execution_id = str(uuid.uuid4())[:8]
    
    # Create workflow execution
    execution = WorkflowExecution(
        execution_id=execution_id,
        workflow_id=workflow_id,
        user_id=current_user.id,
        input_data=execution_data.input_data,
        context=execution_data.context or {},
        document_id=execution_data.document_id
    )
    
    db.add(execution)
    db.commit()
    db.refresh(execution)
    
    # TODO: Start workflow execution in background
    
    logger.info(
        "Workflow execution started",
        workflow_id=str(workflow_id),
        execution_id=execution_id,
        user_id=str(current_user.id)
    )
    
    return WorkflowExecutionResponse.model_validate(execution)


@router.get("/{workflow_id}/executions", response_model=PaginatedResponse)
async def list_workflow_executions(
    workflow_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[ExecutionStatus] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List executions for a workflow.
    
    Args:
        workflow_id: Workflow UUID
        skip: Number of executions to skip
        limit: Maximum number of executions to return
        status: Filter by execution status
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        PaginatedResponse: List of executions with pagination info
    """
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    # Check access permissions
    if not check_workflow_access(str(workflow_id), current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    skip, limit = validate_pagination(skip, limit)
    
    # Build query
    query = db.query(WorkflowExecution).filter(WorkflowExecution.workflow_id == workflow_id)
    
    # Apply filters
    if status:
        query = query.filter(WorkflowExecution.status == status)
    
    # Get total count
    total = query.count()
    
    # Apply pagination and ordering
    executions = query.order_by(WorkflowExecution.created_at.desc()).offset(skip).limit(limit).all()
    
    return PaginatedResponse(
        items=[WorkflowExecutionResponse.model_validate(execution) for execution in executions],
        total=total,
        skip=skip,
        limit=limit,
        has_next=skip + limit < total,
        has_prev=skip > 0
    )


@router.get("/{workflow_id}/executions/{execution_id}", response_model=WorkflowExecutionResponse)
async def get_workflow_execution(
    workflow_id: UUID,
    execution_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific workflow execution.
    
    Args:
        workflow_id: Workflow UUID
        execution_id: Execution ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        WorkflowExecutionResponse: Execution information
    """
    execution = db.query(WorkflowExecution).filter(
        WorkflowExecution.workflow_id == workflow_id,
        WorkflowExecution.execution_id == execution_id
    ).first()
    
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found"
        )
    
    # Check access permissions
    if not check_workflow_access(str(workflow_id), current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return WorkflowExecutionResponse.model_validate(execution)


@router.post("/{workflow_id}/executions/{execution_id}/cancel", response_model=SuccessResponse)
async def cancel_workflow_execution(
    workflow_id: UUID,
    execution_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Cancel a workflow execution.
    
    Args:
        workflow_id: Workflow UUID
        execution_id: Execution ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        SuccessResponse: Cancellation confirmation
    """
    execution = db.query(WorkflowExecution).filter(
        WorkflowExecution.workflow_id == workflow_id,
        WorkflowExecution.execution_id == execution_id
    ).first()
    
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found"
        )
    
    # Check access permissions
    if not check_workflow_access(str(workflow_id), current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Check if execution can be cancelled
    if execution.is_completed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Execution is already completed"
        )
    
    execution.cancel()
    db.commit()
    
    # TODO: Signal background task to stop
    
    logger.info(
        "Workflow execution cancelled",
        workflow_id=str(workflow_id),
        execution_id=execution_id,
        user_id=str(current_user.id)
    )
    
    return SuccessResponse(
        message="Execution cancelled successfully"
    )