"""
Schema validator for Model Context Protocol (MCP) in ClerkAI.
"""

from typing import Dict, List, Any, Optional, Union
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator
import logging

from config.logging import LoggerMixin

logger = logging.getLogger(__name__)


class MCPSchemaValidator(LoggerMixin):
    """Validator for MCP schemas, tools, and workflows."""
    
    def __init__(self):
        """Initialize schema validator with MCP schemas."""
        self.logger.info("MCP Schema Validator initialized")
        
        # Base JSON schema for common field types
        self.base_field_schema = {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["string", "number", "integer", "boolean", "array", "object"]},
                "description": {"type": "string"},
                "required": {"type": "boolean", "default": False},
                "default": {},
                "enum": {"type": "array"},
                "format": {"type": "string"},
                "pattern": {"type": "string"},
                "minimum": {"type": "number"},
                "maximum": {"type": "number"},
                "minLength": {"type": "integer"},
                "maxLength": {"type": "integer"},
                "items": {"type": "object"},
                "properties": {"type": "object"},
                "additionalProperties": {"type": "boolean"}
            },
            "required": ["type"],
            "additionalProperties": False
        }
        
        # Schema for MCP Tool definition
        self.tool_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "pattern": "^[a-zA-Z][a-zA-Z0-9_]*$"},
                "description": {"type": "string", "minLength": 10},
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "category": {"type": "string", "default": "general"},
                "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
                "enabled": {"type": "boolean", "default": True},
                "metadata": {"type": "object"}
            },
            "required": ["name", "description", "input_schema"],
            "additionalProperties": False
        }
        
        # Schema for MCP Workflow definition
        self.workflow_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "pattern": "^[a-zA-Z][a-zA-Z0-9_]*$"},
                "description": {"type": "string", "minLength": 10},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "tool": {"type": "string"},
                            "input_mapping": {"type": "object"},
                            "output_mapping": {"type": "object"},
                            "condition": {"type": "string"},
                            "error_handling": {"type": "string", "enum": ["stop", "continue", "retry"]},
                            "retry_count": {"type": "integer", "minimum": 0, "maximum": 10},
                            "timeout": {"type": "integer", "minimum": 1}
                        },
                        "required": ["name", "tool"],
                        "additionalProperties": False
                    },
                    "minItems": 1
                },
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "triggers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["manual", "document_upload", "schedule", "api", "email"]},
                            "condition": {"type": "string"},
                            "config": {"type": "object"}
                        },
                        "required": ["type"],
                        "additionalProperties": False
                    }
                },
                "category": {"type": "string", "default": "general"},
                "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
                "enabled": {"type": "boolean", "default": True},
                "metadata": {"type": "object"}
            },
            "required": ["name", "description", "steps", "input_schema"],
            "additionalProperties": False
        }
        
        # Schema for MCP Schema definition
        self.schema_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "pattern": "^[a-zA-Z][a-zA-Z0-9_]*$"},
                "description": {"type": "string", "minLength": 10},
                "fields": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z][a-zA-Z0-9_]*$": self.base_field_schema
                    },
                    "additionalProperties": False
                },
                "category": {"type": "string", "default": "general"},
                "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
                "enabled": {"type": "boolean", "default": True},
                "metadata": {"type": "object"}
            },
            "required": ["name", "description", "fields"],
            "additionalProperties": False
        }
        
        # Schema for complete MCP configuration
        self.config_schema = {
            "type": "object",
            "properties": {
                "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
                "tools": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z][a-zA-Z0-9_]*$": self.tool_schema
                    },
                    "additionalProperties": False
                },
                "workflows": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z][a-zA-Z0-9_]*$": self.workflow_schema
                    },
                    "additionalProperties": False
                },
                "schemas": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z][a-zA-Z0-9_]*$": self.schema_schema
                    },
                    "additionalProperties": False
                },
                "global_config": {"type": "object"},
                "metadata": {"type": "object"},
                "last_updated": {"type": "string", "format": "date-time"}
            },
            "required": ["version"],
            "additionalProperties": False
        }
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate complete MCP configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            validate(instance=config, schema=self.config_schema)
            
            # Additional semantic validation
            self._validate_workflow_dependencies(config)
            
            self.logger.debug("MCP configuration validation passed")
            
        except ValidationError as e:
            self.logger.error(f"MCP configuration validation failed: {e.message}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected validation error: {e}")
            raise ValidationError(f"Validation failed: {str(e)}")
    
    def validate_tool(self, tool: Dict[str, Any]) -> None:
        """
        Validate MCP tool definition.
        
        Args:
            tool: Tool dictionary to validate
            
        Raises:
            ValidationError: If tool is invalid
        """
        try:
            validate(instance=tool, schema=self.tool_schema)
            
            # Validate input schema is valid JSON Schema
            if "input_schema" in tool:
                self._validate_json_schema(tool["input_schema"])
            
            # Validate output schema if present
            if "output_schema" in tool:
                self._validate_json_schema(tool["output_schema"])
            
            self.logger.debug(f"Tool validation passed: {tool.get('name', 'unknown')}")
            
        except ValidationError as e:
            self.logger.error(f"Tool validation failed: {e.message}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected tool validation error: {e}")
            raise ValidationError(f"Tool validation failed: {str(e)}")
    
    def validate_workflow(self, workflow: Dict[str, Any]) -> None:
        """
        Validate MCP workflow definition.
        
        Args:
            workflow: Workflow dictionary to validate
            
        Raises:
            ValidationError: If workflow is invalid
        """
        try:
            validate(instance=workflow, schema=self.workflow_schema)
            
            # Validate input schema is valid JSON Schema
            if "input_schema" in workflow:
                self._validate_json_schema(workflow["input_schema"])
            
            # Validate output schema if present
            if "output_schema" in workflow:
                self._validate_json_schema(workflow["output_schema"])
            
            # Validate step dependencies
            self._validate_workflow_steps(workflow)
            
            self.logger.debug(f"Workflow validation passed: {workflow.get('name', 'unknown')}")
            
        except ValidationError as e:
            self.logger.error(f"Workflow validation failed: {e.message}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected workflow validation error: {e}")
            raise ValidationError(f"Workflow validation failed: {str(e)}")
    
    def validate_schema(self, schema: Dict[str, Any]) -> None:
        """
        Validate MCP schema definition.
        
        Args:
            schema: Schema dictionary to validate
            
        Raises:
            ValidationError: If schema is invalid
        """
        try:
            validate(instance=schema, schema=self.schema_schema)
            
            # Validate field definitions
            if "fields" in schema:
                for field_name, field_def in schema["fields"].items():
                    self._validate_field_definition(field_name, field_def)
            
            self.logger.debug(f"Schema validation passed: {schema.get('name', 'unknown')}")
            
        except ValidationError as e:
            self.logger.error(f"Schema validation failed: {e.message}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected schema validation error: {e}")
            raise ValidationError(f"Schema validation failed: {str(e)}")
    
    def validate_data_against_schema(self, data: Any, schema: Dict[str, Any]) -> None:
        """
        Validate data against a schema definition.
        
        Args:
            data: Data to validate
            schema: Schema to validate against
            
        Raises:
            ValidationError: If data doesn't match schema
        """
        try:
            # Convert MCP schema to JSON Schema format
            json_schema = self._convert_mcp_schema_to_json_schema(schema)
            
            # Validate data
            validate(instance=data, schema=json_schema)
            
            self.logger.debug("Data validation against schema passed")
            
        except ValidationError as e:
            self.logger.error(f"Data validation failed: {e.message}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected data validation error: {e}")
            raise ValidationError(f"Data validation failed: {str(e)}")
    
    def _validate_json_schema(self, schema: Dict[str, Any]) -> None:
        """
        Validate that a dictionary is a valid JSON Schema.
        
        Args:
            schema: Schema dictionary to validate
            
        Raises:
            ValidationError: If schema is not valid JSON Schema
        """
        try:
            # Try to create a validator with the schema
            Draft7Validator.check_schema(schema)
        except Exception as e:
            raise ValidationError(f"Invalid JSON Schema: {str(e)}")
    
    def _validate_workflow_dependencies(self, config: Dict[str, Any]) -> None:
        """
        Validate workflow dependencies (tools exist, no circular references).
        
        Args:
            config: Complete configuration to validate
            
        Raises:
            ValidationError: If dependencies are invalid
        """
        workflows = config.get("workflows", {})
        tools = config.get("tools", {})
        
        for workflow_name, workflow in workflows.items():
            # Check that all referenced tools exist
            for step in workflow.get("steps", []):
                tool_name = step.get("tool")
                if tool_name and tool_name not in tools:
                    raise ValidationError(
                        f"Workflow '{workflow_name}' references unknown tool '{tool_name}'"
                    )
            
            # Check for circular dependencies (simplified check)
            self._check_workflow_cycles(workflow_name, workflow, workflows)
    
    def _check_workflow_cycles(
        self, 
        workflow_name: str, 
        workflow: Dict[str, Any], 
        all_workflows: Dict[str, Any],
        visited: Optional[set] = None
    ) -> None:
        """
        Check for circular dependencies in workflow definitions.
        
        Args:
            workflow_name: Name of current workflow
            workflow: Workflow definition
            all_workflows: All workflow definitions
            visited: Set of visited workflows (for recursion)
            
        Raises:
            ValidationError: If circular dependency found
        """
        if visited is None:
            visited = set()
        
        if workflow_name in visited:
            raise ValidationError(f"Circular dependency detected involving workflow '{workflow_name}'")
        
        visited.add(workflow_name)
        
        # Check if any steps reference other workflows (simplified check)
        # This would need to be expanded based on actual workflow execution semantics
        
        visited.remove(workflow_name)
    
    def _validate_workflow_steps(self, workflow: Dict[str, Any]) -> None:
        """
        Validate workflow steps for consistency.
        
        Args:
            workflow: Workflow definition to validate
            
        Raises:
            ValidationError: If steps are invalid
        """
        steps = workflow.get("steps", [])
        step_names = set()
        
        for i, step in enumerate(steps):
            step_name = step.get("name")
            
            # Check for duplicate step names
            if step_name in step_names:
                raise ValidationError(f"Duplicate step name '{step_name}' in workflow")
            step_names.add(step_name)
            
            # Validate input mapping references
            input_mapping = step.get("input_mapping", {})
            self._validate_mapping_references(input_mapping, step_names, f"step {i+1}")
    
    def _validate_mapping_references(
        self, 
        mapping: Dict[str, Any], 
        available_steps: set, 
        context: str
    ) -> None:
        """
        Validate variable references in mappings.
        
        Args:
            mapping: Input/output mapping to validate
            available_steps: Set of available step names
            context: Context for error messages
            
        Raises:
            ValidationError: If references are invalid
        """
        for key, value in mapping.items():
            if isinstance(value, str) and value.startswith("${"):
                # Parse variable reference
                if value.startswith("${steps."):
                    # Extract step name
                    ref_part = value[8:-1]  # Remove ${steps. and }
                    step_ref = ref_part.split(".")[0]
                    
                    if step_ref not in available_steps:
                        raise ValidationError(
                            f"Invalid step reference '{step_ref}' in {context}"
                        )
    
    def _validate_field_definition(self, field_name: str, field_def: Dict[str, Any]) -> None:
        """
        Validate individual field definition.
        
        Args:
            field_name: Name of the field
            field_def: Field definition to validate
            
        Raises:
            ValidationError: If field definition is invalid
        """
        field_type = field_def.get("type")
        
        # Type-specific validation
        if field_type == "array" and "items" not in field_def:
            raise ValidationError(f"Array field '{field_name}' must specify 'items'")
        
        if field_type == "object" and "properties" not in field_def:
            self.logger.warning(f"Object field '{field_name}' should specify 'properties'")
        
        # Validate enum values match type
        if "enum" in field_def:
            enum_values = field_def["enum"]
            for value in enum_values:
                if not self._value_matches_type(value, field_type):
                    raise ValidationError(
                        f"Enum value '{value}' doesn't match type '{field_type}' for field '{field_name}'"
                    )
        
        # Validate default value matches type
        if "default" in field_def:
            default_value = field_def["default"]
            if not self._value_matches_type(default_value, field_type):
                raise ValidationError(
                    f"Default value '{default_value}' doesn't match type '{field_type}' for field '{field_name}'"
                )
    
    def _value_matches_type(self, value: Any, expected_type: str) -> bool:
        """
        Check if a value matches an expected type.
        
        Args:
            value: Value to check
            expected_type: Expected type string
            
        Returns:
            True if value matches type, False otherwise
        """
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return False
        
        return isinstance(value, expected_python_type)
    
    def _convert_mcp_schema_to_json_schema(self, mcp_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert MCP schema format to standard JSON Schema.
        
        Args:
            mcp_schema: MCP schema definition
            
        Returns:
            JSON Schema dictionary
        """
        if "fields" not in mcp_schema:
            raise ValidationError("MCP schema must have 'fields' property")
        
        json_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for field_name, field_def in mcp_schema["fields"].items():
            # Convert field definition to JSON Schema property
            prop_schema = dict(field_def)
            
            # Handle required fields
            if prop_schema.pop("required", False):
                json_schema["required"].append(field_name)
            
            json_schema["properties"][field_name] = prop_schema
        
        return json_schema
    
    def get_validation_errors(self, data: Any, schema: Dict[str, Any]) -> List[str]:
        """
        Get list of validation errors without raising exception.
        
        Args:
            data: Data to validate
            schema: Schema to validate against
            
        Returns:
            List of error messages
        """
        errors = []
        
        try:
            self.validate_data_against_schema(data, schema)
        except ValidationError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform schema validator health check.
        
        Returns:
            Health check results
        """
        health = {
            "service": "MCP Schema Validator",
            "status": "healthy",
            "schemas_loaded": {
                "tool_schema": bool(self.tool_schema),
                "workflow_schema": bool(self.workflow_schema),
                "schema_schema": bool(self.schema_schema),
                "config_schema": bool(self.config_schema)
            },
            "errors": []
        }
        
        # Test basic validation functionality
        try:
            test_tool = {
                "name": "test_tool",
                "description": "Test tool description for health check",
                "input_schema": {"type": "object", "properties": {"test": {"type": "string"}}},
                "category": "test",
                "version": "1.0.0"
            }
            
            self.validate_tool(test_tool)
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["errors"].append(f"Basic validation test failed: {str(e)}")
        
        return health