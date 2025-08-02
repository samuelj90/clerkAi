"""
Model Context Protocol (MCP) Context Manager for ClerkAI.
"""

import json
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict
import logging

from config.settings import settings
from config.logging import LoggerMixin, log_performance
from .schema_validator import MCPSchemaValidator

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """MCP Tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    category: str = "general"
    version: str = "1.0.0"
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MCPWorkflow:
    """MCP Workflow definition."""
    name: str
    description: str
    steps: List[Dict[str, Any]]
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    triggers: List[Dict[str, str]] = None
    category: str = "general"
    version: str = "1.0.0"
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MCPSchema:
    """MCP Schema definition."""
    name: str
    description: str
    fields: Dict[str, Any]
    category: str = "general"
    version: str = "1.0.0"
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MCPContext:
    """MCP Context container."""
    version: str
    tools: Dict[str, MCPTool]
    workflows: Dict[str, MCPWorkflow] 
    schemas: Dict[str, MCPSchema]
    global_config: Dict[str, Any]
    last_updated: datetime
    metadata: Optional[Dict[str, Any]] = None


class MCPContextManager(LoggerMixin):
    """Manager for Model Context Protocol contexts and configurations."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize MCP Context Manager.
        
        Args:
            config_file: Path to MCP configuration file
        """
        self.config_file = config_file or settings.mcp_config_file
        self.auto_reload = settings.mcp_auto_reload
        self.context: Optional[MCPContext] = None
        self.validator = MCPSchemaValidator()
        self._file_watcher_task = None
        
        # Load initial context
        self.load_context()
        
        # Start file watcher if auto-reload is enabled
        if self.auto_reload:
            self._start_file_watcher()
        
        self.logger.info("MCP Context Manager initialized", config_file=self.config_file)
    
    @log_performance("mcp_load_context")
    def load_context(self) -> None:
        """Load MCP context from configuration file."""
        try:
            if not os.path.exists(self.config_file):
                self.logger.warning(f"MCP config file not found: {self.config_file}")
                self._create_default_context()
                return
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Validate configuration structure
            self.validator.validate_config(config_data)
            
            # Parse configuration into MCP objects
            tools = {}
            for tool_name, tool_data in config_data.get('tools', {}).items():
                tools[tool_name] = MCPTool(
                    name=tool_name,
                    **tool_data
                )
            
            workflows = {}
            for workflow_name, workflow_data in config_data.get('workflows', {}).items():
                workflows[workflow_name] = MCPWorkflow(
                    name=workflow_name,
                    **workflow_data
                )
            
            schemas = {}
            for schema_name, schema_data in config_data.get('schemas', {}).items():
                schemas[schema_name] = MCPSchema(
                    name=schema_name,
                    **schema_data
                )
            
            self.context = MCPContext(
                version=config_data.get('version', '1.0.0'),
                tools=tools,
                workflows=workflows,
                schemas=schemas,
                global_config=config_data.get('global_config', {}),
                last_updated=datetime.utcnow(),
                metadata=config_data.get('metadata', {})
            )
            
            self.logger.info(
                "MCP context loaded successfully",
                tools_count=len(tools),
                workflows_count=len(workflows),
                schemas_count=len(schemas)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load MCP context: {e}")
            self._create_default_context()
    
    def save_context(self) -> None:
        """Save current MCP context to configuration file."""
        if not self.context:
            raise ValueError("No context to save")
        
        try:
            config_data = {
                'version': self.context.version,
                'tools': {
                    name: asdict(tool) for name, tool in self.context.tools.items()
                },
                'workflows': {
                    name: asdict(workflow) for name, workflow in self.context.workflows.items()
                },
                'schemas': {
                    name: asdict(schema) for name, schema in self.context.schemas.items()
                },
                'global_config': self.context.global_config,
                'metadata': self.context.metadata or {},
                'last_updated': self.context.last_updated.isoformat()
            }
            
            # Validate before saving
            self.validator.validate_config(config_data)
            
            # Create backup of existing file
            if os.path.exists(self.config_file):
                backup_file = f"{self.config_file}.backup.{int(datetime.utcnow().timestamp())}"
                os.rename(self.config_file, backup_file)
            
            # Save new configuration
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.context.last_updated = datetime.utcnow()
            
            self.logger.info("MCP context saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save MCP context: {e}")
            raise
    
    def _create_default_context(self) -> None:
        """Create default MCP context."""
        default_tools = {
            'document_ocr': MCPTool(
                name='document_ocr',
                description='Extract text from documents using OCR',
                input_schema={
                    'type': 'object',
                    'properties': {
                        'file_path': {'type': 'string'},
                        'language': {'type': 'string', 'default': 'eng'},
                        'enhance_image': {'type': 'boolean', 'default': True}
                    },
                    'required': ['file_path']
                },
                output_schema={
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string'},
                        'confidence': {'type': 'number'},
                        'language': {'type': 'string'}
                    }
                },
                category='document_processing'
            ),
            'nlp_analysis': MCPTool(
                name='nlp_analysis',
                description='Analyze text using NLP techniques',
                input_schema={
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string'},
                        'extract_entities': {'type': 'boolean', 'default': True},
                        'extract_keywords': {'type': 'boolean', 'default': True}
                    },
                    'required': ['text']
                },
                output_schema={
                    'type': 'object',
                    'properties': {
                        'entities': {'type': 'array'},
                        'keywords': {'type': 'array'},
                        'sentiment': {'type': 'string'}
                    }
                },
                category='nlp'
            )
        }
        
        default_workflows = {
            'document_processing': MCPWorkflow(
                name='document_processing',
                description='Complete document processing workflow',
                steps=[
                    {
                        'name': 'ocr_extraction',
                        'tool': 'document_ocr',
                        'input_mapping': {'file_path': '${input.file_path}'}
                    },
                    {
                        'name': 'nlp_analysis',
                        'tool': 'nlp_analysis',
                        'input_mapping': {'text': '${steps.ocr_extraction.output.text}'}
                    }
                ],
                input_schema={
                    'type': 'object',
                    'properties': {
                        'file_path': {'type': 'string'}
                    },
                    'required': ['file_path']
                },
                triggers=[
                    {'type': 'document_upload', 'condition': 'auto_process=true'}
                ],
                category='document_processing'
            )
        }
        
        default_schemas = {
            'invoice': MCPSchema(
                name='invoice',
                description='Invoice document schema',
                fields={
                    'invoice_number': {'type': 'string', 'required': True},
                    'date': {'type': 'string', 'format': 'date'},
                    'vendor': {'type': 'string', 'required': True},
                    'total_amount': {'type': 'number', 'required': True},
                    'currency': {'type': 'string', 'default': 'USD'},
                    'line_items': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'description': {'type': 'string'},
                                'quantity': {'type': 'number'},
                                'unit_price': {'type': 'number'},
                                'total': {'type': 'number'}
                            }
                        }
                    }
                },
                category='finance'
            )
        }
        
        self.context = MCPContext(
            version='1.0.0',
            tools=default_tools,
            workflows=default_workflows,
            schemas=default_schemas,
            global_config={
                'auto_process_documents': True,
                'default_language': 'en',
                'max_file_size_mb': 10
            },
            last_updated=datetime.utcnow(),
            metadata={'created_by': 'system', 'created_at': datetime.utcnow().isoformat()}
        )
        
        # Save default context
        self.save_context()
        
        self.logger.info("Created default MCP context")
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """
        Get tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            MCPTool or None if not found
        """
        if not self.context:
            return None
        
        return self.context.tools.get(tool_name)
    
    def get_workflow(self, workflow_name: str) -> Optional[MCPWorkflow]:
        """
        Get workflow by name.
        
        Args:
            workflow_name: Name of the workflow
            
        Returns:
            MCPWorkflow or None if not found
        """
        if not self.context:
            return None
        
        return self.context.workflows.get(workflow_name)
    
    def get_schema(self, schema_name: str) -> Optional[MCPSchema]:
        """
        Get schema by name.
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            MCPSchema or None if not found
        """
        if not self.context:
            return None
        
        return self.context.schemas.get(schema_name)
    
    def add_tool(self, tool: MCPTool) -> None:
        """
        Add or update a tool.
        
        Args:
            tool: MCPTool to add
        """
        if not self.context:
            raise ValueError("Context not initialized")
        
        # Validate tool
        self.validator.validate_tool(asdict(tool))
        
        self.context.tools[tool.name] = tool
        self.context.last_updated = datetime.utcnow()
        
        self.logger.info(f"Added/updated tool: {tool.name}")
    
    def add_workflow(self, workflow: MCPWorkflow) -> None:
        """
        Add or update a workflow.
        
        Args:
            workflow: MCPWorkflow to add
        """
        if not self.context:
            raise ValueError("Context not initialized")
        
        # Validate workflow
        self.validator.validate_workflow(asdict(workflow))
        
        self.context.workflows[workflow.name] = workflow
        self.context.last_updated = datetime.utcnow()
        
        self.logger.info(f"Added/updated workflow: {workflow.name}")
    
    def add_schema(self, schema: MCPSchema) -> None:
        """
        Add or update a schema.
        
        Args:
            schema: MCPSchema to add
        """
        if not self.context:
            raise ValueError("Context not initialized")
        
        # Validate schema
        self.validator.validate_schema(asdict(schema))
        
        self.context.schemas[schema.name] = schema
        self.context.last_updated = datetime.utcnow()
        
        self.logger.info(f"Added/updated schema: {schema.name}")
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool.
        
        Args:
            tool_name: Name of tool to remove
            
        Returns:
            True if removed, False if not found
        """
        if not self.context or tool_name not in self.context.tools:
            return False
        
        del self.context.tools[tool_name]
        self.context.last_updated = datetime.utcnow()
        
        self.logger.info(f"Removed tool: {tool_name}")
        return True
    
    def remove_workflow(self, workflow_name: str) -> bool:
        """
        Remove a workflow.
        
        Args:
            workflow_name: Name of workflow to remove
            
        Returns:
            True if removed, False if not found
        """
        if not self.context or workflow_name not in self.context.workflows:
            return False
        
        del self.context.workflows[workflow_name]
        self.context.last_updated = datetime.utcnow()
        
        self.logger.info(f"Removed workflow: {workflow_name}")
        return True
    
    def remove_schema(self, schema_name: str) -> bool:
        """
        Remove a schema.
        
        Args:
            schema_name: Name of schema to remove
            
        Returns:
            True if removed, False if not found
        """
        if not self.context or schema_name not in self.context.schemas:
            return False
        
        del self.context.schemas[schema_name]
        self.context.last_updated = datetime.utcnow()
        
        self.logger.info(f"Removed schema: {schema_name}")
        return True
    
    def list_tools(self, category: Optional[str] = None, enabled_only: bool = False) -> List[MCPTool]:
        """
        List tools with optional filtering.
        
        Args:
            category: Filter by category
            enabled_only: Only return enabled tools
            
        Returns:
            List of MCPTool objects
        """
        if not self.context:
            return []
        
        tools = list(self.context.tools.values())
        
        if category:
            tools = [tool for tool in tools if tool.category == category]
        
        if enabled_only:
            tools = [tool for tool in tools if tool.enabled]
        
        return tools
    
    def list_workflows(self, category: Optional[str] = None, enabled_only: bool = False) -> List[MCPWorkflow]:
        """
        List workflows with optional filtering.
        
        Args:
            category: Filter by category
            enabled_only: Only return enabled workflows
            
        Returns:
            List of MCPWorkflow objects
        """
        if not self.context:
            return []
        
        workflows = list(self.context.workflows.values())
        
        if category:
            workflows = [workflow for workflow in workflows if workflow.category == category]
        
        if enabled_only:
            workflows = [workflow for workflow in workflows if workflow.enabled]
        
        return workflows
    
    def list_schemas(self, category: Optional[str] = None, enabled_only: bool = False) -> List[MCPSchema]:
        """
        List schemas with optional filtering.
        
        Args:
            category: Filter by category
            enabled_only: Only return enabled schemas
            
        Returns:
            List of MCPSchema objects
        """
        if not self.context:
            return []
        
        schemas = list(self.context.schemas.values())
        
        if category:
            schemas = [schema for schema in schemas if schema.category == category]
        
        if enabled_only:
            schemas = [schema for schema in schemas if schema.enabled]
        
        return schemas
    
    def get_global_config(self, key: str, default: Any = None) -> Any:
        """
        Get global configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if not self.context:
            return default
        
        return self.context.global_config.get(key, default)
    
    def set_global_config(self, key: str, value: Any) -> None:
        """
        Set global configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        if not self.context:
            raise ValueError("Context not initialized")
        
        self.context.global_config[key] = value
        self.context.last_updated = datetime.utcnow()
        
        self.logger.info(f"Updated global config: {key}")
    
    def _start_file_watcher(self) -> None:
        """Start file watcher for auto-reload."""
        if self._file_watcher_task:
            return
        
        async def watch_file():
            """Watch configuration file for changes."""
            last_modified = None
            
            while True:
                try:
                    if os.path.exists(self.config_file):
                        current_modified = os.path.getmtime(self.config_file)
                        
                        if last_modified is not None and current_modified > last_modified:
                            self.logger.info("MCP config file changed, reloading...")
                            self.load_context()
                        
                        last_modified = current_modified
                    
                    await asyncio.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"File watcher error: {e}")
                    await asyncio.sleep(10)  # Wait longer on errors
        
        self._file_watcher_task = asyncio.create_task(watch_file())
        self.logger.info("Started MCP config file watcher")
    
    def stop_file_watcher(self) -> None:
        """Stop file watcher."""
        if self._file_watcher_task:
            self._file_watcher_task.cancel()
            self._file_watcher_task = None
            self.logger.info("Stopped MCP config file watcher")
    
    def export_context(self, export_path: str) -> None:
        """
        Export context to a file.
        
        Args:
            export_path: Path to export file
        """
        if not self.context:
            raise ValueError("No context to export")
        
        export_data = {
            'version': self.context.version,
            'exported_at': datetime.utcnow().isoformat(),
            'tools': {name: asdict(tool) for name, tool in self.context.tools.items()},
            'workflows': {name: asdict(workflow) for name, workflow in self.context.workflows.items()},
            'schemas': {name: asdict(schema) for name, schema in self.context.schemas.items()},
            'global_config': self.context.global_config,
            'metadata': self.context.metadata
        }
        
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Context exported to: {export_path}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform MCP context manager health check.
        
        Returns:
            Health check results
        """
        health = {
            "service": "MCP Context Manager",
            "status": "healthy",
            "context_loaded": self.context is not None,
            "config_file": self.config_file,
            "config_file_exists": os.path.exists(self.config_file),
            "auto_reload": self.auto_reload,
            "file_watcher_running": self._file_watcher_task is not None,
            "last_updated": self.context.last_updated.isoformat() if self.context else None,
            "counts": {},
            "errors": []
        }
        
        if self.context:
            health["counts"] = {
                "tools": len(self.context.tools),
                "workflows": len(self.context.workflows),
                "schemas": len(self.context.schemas)
            }
        else:
            health["status"] = "unhealthy"
            health["errors"].append("MCP context not loaded")
        
        if not os.path.exists(self.config_file):
            health["errors"].append(f"Config file not found: {self.config_file}")
        
        return health
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_file_watcher()


# Global MCP context manager instance
mcp_context = MCPContextManager()