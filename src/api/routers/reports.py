"""
Reports API router for ClerkAI.
"""

from typing import List, Optional, Dict, Any
from fastapi import (
    APIRouter, Depends, HTTPException, status, Query
)
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from datetime import datetime, timedelta
from uuid import UUID

from config.database import get_db
from src.api.dependencies import get_current_active_user, validate_pagination
from src.api.schemas.base import BaseSchema, PaginatedResponse
from src.models.user import User
from src.models.document import Document, DocumentStatus, DocumentType
from src.models.workflow import Workflow, WorkflowExecution, ExecutionStatus
from config.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class DocumentStatsResponse(BaseSchema):
    """Document statistics response schema."""
    
    total_documents: int
    processed_documents: int
    processing_documents: int
    failed_documents: int
    total_size_mb: float
    documents_by_type: Dict[str, int]
    documents_by_month: Dict[str, int]
    processing_success_rate: float


class WorkflowStatsResponse(BaseSchema):
    """Workflow statistics response schema."""
    
    total_workflows: int
    active_workflows: int
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_execution_time: Optional[float] = None
    executions_by_status: Dict[str, int]
    executions_by_month: Dict[str, int]
    success_rate: float


class SystemStatsResponse(BaseSchema):
    """System statistics response schema."""
    
    total_users: int
    active_users: int
    total_storage_mb: float
    average_processing_time: Optional[float] = None
    most_active_users: List[Dict[str, Any]]
    most_processed_document_types: List[Dict[str, Any]]


class DashboardResponse(BaseSchema):
    """Dashboard data response schema."""
    
    document_stats: DocumentStatsResponse
    workflow_stats: WorkflowStatsResponse
    system_stats: SystemStatsResponse
    recent_activity: List[Dict[str, Any]]


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard_data(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get dashboard data for the current user.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        DashboardResponse: Comprehensive dashboard data
    """
    # Document statistics
    doc_stats = get_document_statistics(current_user, db)
    
    # Workflow statistics
    workflow_stats = get_workflow_statistics(current_user, db)
    
    # System statistics (for superusers only)
    if current_user.is_superuser:
        system_stats = get_system_statistics(db)
    else:
        system_stats = SystemStatsResponse(
            total_users=0,
            active_users=0,
            total_storage_mb=0.0,
            most_active_users=[],
            most_processed_document_types=[]
        )
    
    # Recent activity
    recent_activity = get_recent_activity(current_user, db)
    
    return DashboardResponse(
        document_stats=doc_stats,
        workflow_stats=workflow_stats,
        system_stats=system_stats,
        recent_activity=recent_activity
    )


@router.get("/documents/stats", response_model=DocumentStatsResponse)
async def get_document_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get document statistics for the current user.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        DocumentStatsResponse: Document statistics
    """
    return get_document_statistics(current_user, db)


@router.get("/workflows/stats", response_model=WorkflowStatsResponse)
async def get_workflow_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = get_db
):
    """
    Get workflow statistics for the current user.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        WorkflowStatsResponse: Workflow statistics
    """
    return get_workflow_statistics(current_user, db)


@router.get("/system/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get system statistics (superuser only).
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        SystemStatsResponse: System statistics
        
    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser access required"
        )
    
    return get_system_statistics(db)


@router.get("/dashboard/html", response_class=HTMLResponse)
async def get_dashboard_html(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get HTML dashboard page.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        HTMLResponse: HTML dashboard page
    """
    dashboard_data = await get_dashboard_data(current_user, db)
    
    # Generate HTML dashboard
    html_content = generate_dashboard_html(dashboard_data, current_user)
    
    return HTMLResponse(content=html_content)


def get_document_statistics(user: User, db: Session) -> DocumentStatsResponse:
    """
    Get document statistics for a user.
    
    Args:
        user: User to get statistics for
        db: Database session
        
    Returns:
        DocumentStatsResponse: Document statistics
    """
    # Base query for user's documents
    base_query = db.query(Document).filter(Document.owner_id == user.id)
    
    # Total documents
    total_documents = base_query.count()
    
    # Documents by status
    processed_documents = base_query.filter(Document.status == DocumentStatus.PROCESSED).count()
    processing_documents = base_query.filter(Document.status == DocumentStatus.PROCESSING).count()
    failed_documents = base_query.filter(Document.status == DocumentStatus.FAILED).count()
    
    # Total size
    total_size_bytes = db.query(func.sum(Document.file_size)).filter(
        Document.owner_id == user.id
    ).scalar() or 0
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    # Documents by type
    type_counts = db.query(
        Document.document_type, func.count(Document.id)
    ).filter(
        Document.owner_id == user.id
    ).group_by(Document.document_type).all()
    
    documents_by_type = {doc_type.value: count for doc_type, count in type_counts}
    
    # Documents by month (last 12 months)
    twelve_months_ago = datetime.utcnow() - timedelta(days=365)
    monthly_counts = db.query(
        func.date_trunc('month', Document.created_at).label('month'),
        func.count(Document.id).label('count')
    ).filter(
        Document.owner_id == user.id,
        Document.created_at >= twelve_months_ago
    ).group_by('month').order_by('month').all()
    
    documents_by_month = {
        month.strftime('%Y-%m'): count for month, count in monthly_counts
    }
    
    # Success rate
    processing_success_rate = 0.0
    if total_documents > 0:
        processing_success_rate = (processed_documents / total_documents) * 100
    
    return DocumentStatsResponse(
        total_documents=total_documents,
        processed_documents=processed_documents,
        processing_documents=processing_documents,
        failed_documents=failed_documents,
        total_size_mb=round(total_size_mb, 2),
        documents_by_type=documents_by_type,
        documents_by_month=documents_by_month,
        processing_success_rate=round(processing_success_rate, 2)
    )


def get_workflow_statistics(user: User, db: Session) -> WorkflowStatsResponse:
    """
    Get workflow statistics for a user.
    
    Args:
        user: User to get statistics for
        db: Database session
        
    Returns:
        WorkflowStatsResponse: Workflow statistics
    """
    # Base queries
    workflows_query = db.query(Workflow).filter(Workflow.created_by_id == user.id)
    executions_query = db.query(WorkflowExecution).join(Workflow).filter(
        Workflow.created_by_id == user.id
    )
    
    # Workflow counts
    total_workflows = workflows_query.count()
    active_workflows = workflows_query.filter(Workflow.status == 'ACTIVE').count()
    
    # Execution counts
    total_executions = executions_query.count()
    successful_executions = executions_query.filter(
        WorkflowExecution.status == ExecutionStatus.COMPLETED
    ).count()
    failed_executions = executions_query.filter(
        WorkflowExecution.status == ExecutionStatus.FAILED
    ).count()
    
    # Average execution time
    avg_duration = db.query(func.avg(WorkflowExecution.duration)).join(Workflow).filter(
        Workflow.created_by_id == user.id,
        WorkflowExecution.duration.isnot(None)
    ).scalar()
    
    # Executions by status
    status_counts = db.query(
        WorkflowExecution.status, func.count(WorkflowExecution.id)
    ).join(Workflow).filter(
        Workflow.created_by_id == user.id
    ).group_by(WorkflowExecution.status).all()
    
    executions_by_status = {status.value: count for status, count in status_counts}
    
    # Executions by month
    twelve_months_ago = datetime.utcnow() - timedelta(days=365)
    monthly_exec_counts = db.query(
        func.date_trunc('month', WorkflowExecution.created_at).label('month'),
        func.count(WorkflowExecution.id).label('count')
    ).join(Workflow).filter(
        Workflow.created_by_id == user.id,
        WorkflowExecution.created_at >= twelve_months_ago
    ).group_by('month').order_by('month').all()
    
    executions_by_month = {
        month.strftime('%Y-%m'): count for month, count in monthly_exec_counts
    }
    
    # Success rate
    success_rate = 0.0
    if total_executions > 0:
        success_rate = (successful_executions / total_executions) * 100
    
    return WorkflowStatsResponse(
        total_workflows=total_workflows,
        active_workflows=active_workflows,
        total_executions=total_executions,
        successful_executions=successful_executions,
        failed_executions=failed_executions,
        average_execution_time=round(avg_duration, 2) if avg_duration else None,
        executions_by_status=executions_by_status,
        executions_by_month=executions_by_month,
        success_rate=round(success_rate, 2)
    )


def get_system_statistics(db: Session) -> SystemStatsResponse:
    """
    Get system-wide statistics (superuser only).
    
    Args:
        db: Database session
        
    Returns:
        SystemStatsResponse: System statistics
    """
    # User counts
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    
    # Total storage
    total_storage_bytes = db.query(func.sum(Document.file_size)).scalar() or 0
    total_storage_mb = total_storage_bytes / (1024 * 1024)
    
    # Average processing time
    avg_processing_time = db.query(func.avg(Document.processing_duration)).filter(
        Document.processing_duration.isnot(None)
    ).scalar()
    
    # Most active users (by document count)
    most_active_users_query = db.query(
        User.username,
        User.full_name,
        func.count(Document.id).label('document_count')
    ).join(Document).group_by(
        User.id, User.username, User.full_name
    ).order_by(func.count(Document.id).desc()).limit(10).all()
    
    most_active_users = [
        {
            "username": username,
            "full_name": full_name,
            "document_count": document_count
        }
        for username, full_name, document_count in most_active_users_query
    ]
    
    # Most processed document types
    most_processed_types_query = db.query(
        Document.document_type,
        func.count(Document.id).label('count')
    ).group_by(Document.document_type).order_by(
        func.count(Document.id).desc()
    ).limit(10).all()
    
    most_processed_document_types = [
        {
            "document_type": doc_type.value,
            "count": count
        }
        for doc_type, count in most_processed_types_query
    ]
    
    return SystemStatsResponse(
        total_users=total_users,
        active_users=active_users,
        total_storage_mb=round(total_storage_mb, 2),
        average_processing_time=round(avg_processing_time, 2) if avg_processing_time else None,
        most_active_users=most_active_users,
        most_processed_document_types=most_processed_document_types
    )


def get_recent_activity(user: User, db: Session, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get recent activity for a user.
    
    Args:
        user: User to get activity for
        db: Database session
        limit: Maximum number of activities to return
        
    Returns:
        List of recent activities
    """
    activities = []
    
    # Recent document uploads
    recent_docs = db.query(Document).filter(
        Document.owner_id == user.id
    ).order_by(Document.created_at.desc()).limit(limit // 2).all()
    
    for doc in recent_docs:
        activities.append({
            "type": "document_upload",
            "timestamp": doc.created_at,
            "description": f"Uploaded document: {doc.original_filename}",
            "resource_id": str(doc.id),
            "status": doc.status.value
        })
    
    # Recent workflow executions
    recent_executions = db.query(WorkflowExecution).join(Workflow).filter(
        Workflow.created_by_id == user.id
    ).order_by(WorkflowExecution.created_at.desc()).limit(limit // 2).all()
    
    for execution in recent_executions:
        activities.append({
            "type": "workflow_execution",
            "timestamp": execution.created_at,
            "description": f"Executed workflow: {execution.workflow.name}",
            "resource_id": str(execution.id),
            "status": execution.status.value
        })
    
    # Sort by timestamp and limit
    activities.sort(key=lambda x: x["timestamp"], reverse=True)
    return activities[:limit]


def generate_dashboard_html(dashboard_data: DashboardResponse, user: User) -> str:
    """
    Generate HTML dashboard from data.
    
    Args:
        dashboard_data: Dashboard data
        user: Current user
        
    Returns:
        HTML string for dashboard
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ClerkAI Dashboard - {user.full_name}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }}
            .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .stat-card h3 {{ margin-top: 0; color: #2c3e50; }}
            .stat-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
            .recent-activity {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .activity-item {{ padding: 10px; border-bottom: 1px solid #eee; }}
            .activity-item:last-child {{ border-bottom: none; }}
            .status-success {{ color: #27ae60; }}
            .status-failed {{ color: #e74c3c; }}
            .status-processing {{ color: #f39c12; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>ClerkAI Dashboard</h1>
                <p>Welcome back, {user.full_name}!</p>
                <p>Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Documents</h3>
                    <div class="stat-value">{dashboard_data.document_stats.total_documents}</div>
                    <p>Total Documents</p>
                    <p>Processed: {dashboard_data.document_stats.processed_documents}</p>
                    <p>Processing: {dashboard_data.document_stats.processing_documents}</p>
                    <p>Failed: {dashboard_data.document_stats.failed_documents}</p>
                    <p>Success Rate: {dashboard_data.document_stats.processing_success_rate}%</p>
                </div>
                
                <div class="stat-card">
                    <h3>Workflows</h3>
                    <div class="stat-value">{dashboard_data.workflow_stats.total_workflows}</div>
                    <p>Total Workflows</p>
                    <p>Active: {dashboard_data.workflow_stats.active_workflows}</p>
                    <p>Executions: {dashboard_data.workflow_stats.total_executions}</p>
                    <p>Success Rate: {dashboard_data.workflow_stats.success_rate}%</p>
                </div>
                
                <div class="stat-card">
                    <h3>Storage</h3>
                    <div class="stat-value">{dashboard_data.document_stats.total_size_mb:.1f} MB</div>
                    <p>Total Storage Used</p>
                </div>
            </div>
            
            <div class="recent-activity">
                <h3>Recent Activity</h3>
    """
    
    for activity in dashboard_data.recent_activity:
        status_class = f"status-{activity['status'].replace('_', '-')}"
        html += f"""
                <div class="activity-item">
                    <strong>{activity['description']}</strong>
                    <span class="{status_class}">({activity['status']})</span>
                    <br>
                    <small>{activity['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
                </div>
        """
    
    html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    return html