"""
CrewAI Workflow Orchestrator for ClerkAI.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import asyncio
from dataclasses import dataclass
from enum import Enum
import uuid
import logging

from config.settings import settings
from config.logging import LoggerMixin, log_performance

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CrewTask:
    """Individual task in a crew workflow."""
    id: str
    name: str
    description: str
    agent_type: str
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    status: TaskStatus = TaskStatus.PENDING
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = None


@dataclass
class CrewAgent:
    """CrewAI agent definition."""
    name: str
    role: str
    goal: str
    backstory: str
    tools: List[str]
    capabilities: List[str]
    llm_config: Optional[Dict[str, Any]] = None


class WorkflowOrchestrator(LoggerMixin):
    """Orchestrator for CrewAI workflows in ClerkAI."""
    
    def __init__(self):
        """Initialize workflow orchestrator."""
        self.active_workflows = {}
        self.agents = {}
        self.tools = {}
        self._setup_default_agents()
        self._setup_default_tools()
        
        self.logger.info("CrewAI Workflow Orchestrator initialized")
    
    def _setup_default_agents(self):
        """Setup default agents for document processing."""
        self.agents = {
            'document_analyst': CrewAgent(
                name='Document Analyst',
                role='Document Analysis Specialist',
                goal='Analyze and extract information from documents accurately',
                backstory='Expert in document analysis with deep knowledge of various document types',
                tools=['ocr_service', 'nlp_service'],
                capabilities=['text_extraction', 'entity_recognition', 'classification']
            ),
            'data_extractor': CrewAgent(
                name='Data Extractor',
                role='Structured Data Extraction Specialist',
                goal='Extract structured data from documents based on schemas',
                backstory='Specialized in converting unstructured text into structured data',
                tools=['llm_service', 'nlp_service'],
                capabilities=['data_extraction', 'schema_validation', 'data_transformation']
            ),
            'quality_reviewer': CrewAgent(
                name='Quality Reviewer',
                role='Quality Assurance Specialist',
                goal='Review and validate extracted data for accuracy',
                backstory='Meticulous reviewer ensuring high-quality data extraction',
                tools=['validation_service'],
                capabilities=['data_validation', 'quality_scoring', 'error_detection']
            )
        }
    
    def _setup_default_tools(self):
        """Setup default tools for agents."""
        from src.services.ocr_service import ocr_service
        from src.services.nlp_service import nlp_service
        from src.services.llm_service import llm_service
        
        self.tools = {
            'ocr_service': ocr_service,
            'nlp_service': nlp_service,
            'llm_service': llm_service
        }
    
    async def execute_workflow(
        self,
        workflow_id: str,
        tasks: List[CrewTask],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a crew workflow.
        
        Args:
            workflow_id: Unique workflow identifier
            tasks: List of tasks to execute
            context: Workflow context
            
        Returns:
            Workflow execution results
        """
        self.logger.info(f"Starting workflow execution: {workflow_id}")
        
        workflow_context = {
            'id': workflow_id,
            'started_at': datetime.utcnow(),
            'tasks': {task.id: task for task in tasks},
            'status': 'running',
            'context': context or {},
            'results': {}
        }
        
        self.active_workflows[workflow_id] = workflow_context
        
        try:
            # Execute tasks based on dependencies
            execution_order = self._resolve_task_dependencies(tasks)
            
            for task_batch in execution_order:
                # Execute tasks in parallel within each batch
                batch_results = await self._execute_task_batch(task_batch, workflow_context)
                workflow_context['results'].update(batch_results)
            
            workflow_context['status'] = 'completed'
            workflow_context['completed_at'] = datetime.utcnow()
            
            self.logger.info(f"Workflow execution completed: {workflow_id}")
            
            return {
                'workflow_id': workflow_id,
                'status': 'completed',
                'results': workflow_context['results'],
                'duration': (workflow_context['completed_at'] - workflow_context['started_at']).total_seconds()
            }
            
        except Exception as e:
            workflow_context['status'] = 'failed'
            workflow_context['error'] = str(e)
            workflow_context['completed_at'] = datetime.utcnow()
            
            self.logger.error(f"Workflow execution failed: {workflow_id} - {str(e)}")
            
            return {
                'workflow_id': workflow_id,
                'status': 'failed',
                'error': str(e),
                'results': workflow_context.get('results', {})
            }
        
        finally:
            # Clean up completed workflow after some time
            asyncio.create_task(self._cleanup_workflow(workflow_id, delay=3600))
    
    def _resolve_task_dependencies(self, tasks: List[CrewTask]) -> List[List[CrewTask]]:
        """
        Resolve task dependencies and return execution order.
        
        Args:
            tasks: List of tasks
            
        Returns:
            List of task batches in execution order
        """
        task_map = {task.id: task for task in tasks}
        execution_order = []
        remaining_tasks = set(task.id for task in tasks)
        
        while remaining_tasks:
            # Find tasks with no remaining dependencies
            ready_tasks = []
            for task_id in remaining_tasks:
                task = task_map[task_id]
                dependencies = task.dependencies or []
                
                if all(dep not in remaining_tasks for dep in dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Circular dependency or invalid dependency
                raise ValueError("Circular dependency or invalid task dependencies detected")
            
            execution_order.append(ready_tasks)
            
            # Remove ready tasks from remaining
            for task in ready_tasks:
                remaining_tasks.remove(task.id)
        
        return execution_order
    
    async def _execute_task_batch(
        self,
        tasks: List[CrewTask],
        workflow_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a batch of tasks in parallel.
        
        Args:
            tasks: Tasks to execute
            workflow_context: Workflow context
            
        Returns:
            Batch execution results
        """
        batch_results = {}
        
        # Create coroutines for all tasks in batch
        task_coroutines = [
            self._execute_single_task(task, workflow_context)
            for task in tasks
        ]
        
        # Execute tasks in parallel
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        # Process results
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                task.status = TaskStatus.FAILED
                task.error = str(result)
                task.completed_at = datetime.utcnow()
                batch_results[task.id] = {'error': str(result)}
            else:
                task.status = TaskStatus.COMPLETED
                task.outputs = result
                task.completed_at = datetime.utcnow()
                batch_results[task.id] = result
        
        return batch_results
    
    async def _execute_single_task(
        self,
        task: CrewTask,
        workflow_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
            workflow_context: Workflow context
            
        Returns:
            Task execution result
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        
        self.logger.info(f"Executing task: {task.name} ({task.id})")
        
        try:
            # Get agent for this task
            agent = self.agents.get(task.agent_type)
            if not agent:
                raise ValueError(f"Unknown agent type: {task.agent_type}")
            
            # Prepare task inputs with context
            task_inputs = dict(task.inputs)
            task_inputs['context'] = workflow_context['context']
            task_inputs['workflow_results'] = workflow_context['results']
            
            # Execute task based on agent type
            result = await self._execute_agent_task(agent, task, task_inputs)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {task.name} - {str(e)}")
            raise
    
    async def _execute_agent_task(
        self,
        agent: CrewAgent,
        task: CrewTask,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute task using specified agent.
        
        Args:
            agent: Agent to use
            task: Task to execute
            inputs: Task inputs
            
        Returns:
            Task result
        """
        # This is a simplified implementation
        # In a full CrewAI integration, this would involve more sophisticated agent orchestration
        
        result = {}
        
        if agent.name == 'Document Analyst':
            result = await self._execute_document_analysis(task, inputs)
        elif agent.name == 'Data Extractor':
            result = await self._execute_data_extraction(task, inputs)
        elif agent.name == 'Quality Reviewer':
            result = await self._execute_quality_review(task, inputs)
        else:
            raise ValueError(f"Unknown agent: {agent.name}")
        
        return result
    
    async def _execute_document_analysis(self, task: CrewTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document analysis task."""
        file_path = inputs.get('file_path')
        if not file_path:
            raise ValueError("file_path is required for document analysis")
        
        # Use OCR service
        ocr_result = self.tools['ocr_service'].extract_text(file_path)
        
        # Use NLP service
        nlp_result = self.tools['nlp_service'].process_text(
            ocr_result.text,
            extract_entities=True,
            extract_keywords=True
        )
        
        return {
            'extracted_text': ocr_result.text,
            'ocr_confidence': ocr_result.confidence,
            'entities': nlp_result.entities,
            'keywords': nlp_result.keywords,
            'sentiment': nlp_result.sentiment
        }
    
    async def _execute_data_extraction(self, task: CrewTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data extraction task."""
        text = inputs.get('text') or inputs.get('extracted_text')
        schema = inputs.get('schema')
        
        if not text:
            raise ValueError("text is required for data extraction")
        
        if not schema:
            raise ValueError("schema is required for data extraction")
        
        # Use LLM service for structured data extraction
        extracted_data = await self.tools['llm_service'].extract_structured_data(
            text=text,
            schema=schema
        )
        
        return {
            'extracted_data': extracted_data,
            'extraction_method': 'llm'
        }
    
    async def _execute_quality_review(self, task: CrewTask, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quality review task."""
        extracted_data = inputs.get('extracted_data')
        
        if not extracted_data:
            raise ValueError("extracted_data is required for quality review")
        
        # Simple quality scoring (in production, this would be more sophisticated)
        quality_score = 0.8  # Placeholder
        issues = []
        
        # Check for missing required fields
        schema = inputs.get('schema', {})
        for field_name, field_def in schema.get('fields', {}).items():
            if field_def.get('required', False) and field_name not in extracted_data:
                issues.append(f"Missing required field: {field_name}")
                quality_score -= 0.1
        
        return {
            'quality_score': max(0.0, quality_score),
            'issues': issues,
            'review_status': 'passed' if quality_score >= 0.7 else 'failed'
        }
    
    async def _cleanup_workflow(self, workflow_id: str, delay: int = 3600):
        """Clean up workflow after delay."""
        await asyncio.sleep(delay)
        if workflow_id in self.active_workflows:
            del self.active_workflows[workflow_id]
            self.logger.info(f"Cleaned up workflow: {workflow_id}")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow."""
        return self.active_workflows.get(workflow_id)
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow = self.active_workflows[workflow_id]
        workflow['status'] = 'cancelled'
        workflow['completed_at'] = datetime.utcnow()
        
        # Cancel running tasks
        for task in workflow['tasks'].values():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.utcnow()
        
        self.logger.info(f"Cancelled workflow: {workflow_id}")
        return True
    
    def create_document_processing_workflow(
        self,
        file_path: str,
        schema: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None
    ) -> List[CrewTask]:
        """
        Create a standard document processing workflow.
        
        Args:
            file_path: Path to document to process
            schema: Optional schema for data extraction
            workflow_id: Optional workflow identifier
            
        Returns:
            List of tasks for the workflow
        """
        workflow_id = workflow_id or str(uuid.uuid4())
        
        tasks = [
            CrewTask(
                id=f"{workflow_id}_analysis",
                name="Document Analysis",
                description="Extract text and analyze document content",
                agent_type="document_analyst",
                inputs={"file_path": file_path}
            )
        ]
        
        if schema:
            tasks.extend([
                CrewTask(
                    id=f"{workflow_id}_extraction",
                    name="Data Extraction",
                    description="Extract structured data based on schema",
                    agent_type="data_extractor",
                    inputs={"schema": schema},
                    dependencies=[f"{workflow_id}_analysis"]
                ),
                CrewTask(
                    id=f"{workflow_id}_review",
                    name="Quality Review",
                    description="Review extracted data for quality",
                    agent_type="quality_reviewer",
                    inputs={"schema": schema},
                    dependencies=[f"{workflow_id}_extraction"]
                )
            ])
        
        return tasks
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform workflow orchestrator health check.
        
        Returns:
            Health check results
        """
        return {
            "service": "CrewAI Workflow Orchestrator",
            "status": "healthy",
            "active_workflows": len(self.active_workflows),
            "available_agents": list(self.agents.keys()),
            "available_tools": list(self.tools.keys()),
            "max_concurrent_workflows": settings.crew_max_workers,
            "errors": []
        }


# Global workflow orchestrator instance
workflow_orchestrator = WorkflowOrchestrator()