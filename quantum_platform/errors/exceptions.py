"""
Quantum Platform Exception Hierarchy

This module defines all custom exceptions used throughout the quantum platform,
with clear categorization between user errors and system errors.
"""

import traceback
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for errors."""
    component: str = ""
    operation: str = ""
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    user_input: Optional[str] = None
    system_state: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


class QuantumPlatformError(Exception):
    """
    Base exception for all quantum platform errors.
    
    This provides a consistent interface for error handling across the platform.
    """
    
    def __init__(
        self,
        message: str,
        user_message: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.user_message = user_message or self._generate_user_message()
        self.error_code = error_code or self._generate_error_code()
        self.context = context or ErrorContext()
        self.severity = severity
        self.suggestions = suggestions or []
        self.traceback_str = traceback.format_exc()
    
    def _generate_user_message(self) -> str:
        """Generate a user-friendly message if not provided."""
        return "A quantum platform error occurred. Please check your input and try again."
    
    def _generate_error_code(self) -> str:
        """Generate an error code for reference."""
        return f"QP{hash(self.message) % 10000:04d}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for reporting."""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'user_message': self.user_message,
            'error_code': self.error_code,
            'severity': self.severity.value,
            'context': {
                'component': self.context.component,
                'operation': self.context.operation,
                'line_number': self.context.line_number,
                'file_path': self.context.file_path,
                'user_input': self.context.user_input,
                'system_state': self.context.system_state
            },
            'suggestions': self.suggestions,
            'traceback': self.traceback_str
        }


class UserError(QuantumPlatformError):
    """
    Base class for errors caused by user input or actions.
    
    These errors should have clear, actionable messages for users.
    """
    
    def _generate_user_message(self) -> str:
        return "There's an issue with your quantum program. Please review the details below."


class SystemError(QuantumPlatformError):
    """
    Base class for internal system errors.
    
    These are typically bugs or unexpected conditions in the platform.
    """
    
    def _generate_user_message(self) -> str:
        return "An internal error occurred. This may be a platform issue."


# =============================================================================
# Compiler Errors
# =============================================================================

class CompilationError(UserError):
    """Errors during quantum program compilation."""
    
    def _generate_user_message(self) -> str:
        return "Failed to compile your quantum program. Please check the syntax and structure."


class ParseError(CompilationError):
    """Errors during parsing of quantum programs."""
    
    def _generate_user_message(self) -> str:
        return "Syntax error in your quantum program. Please check for typos or invalid syntax."


class OptimizationError(CompilationError):
    """Errors during circuit optimization."""
    
    def _generate_user_message(self) -> str:
        return "Failed to optimize your quantum circuit. The circuit may be too complex or contain unsupported patterns."


class TranspilationError(CompilationError):
    """Errors during transpilation to hardware."""
    
    def _generate_user_message(self) -> str:
        return "Failed to transpile your circuit for the target hardware. Check hardware compatibility."


# =============================================================================
# Circuit Errors
# =============================================================================

class CircuitError(UserError):
    """Base class for quantum circuit errors."""
    
    def _generate_user_message(self) -> str:
        return "Error in your quantum circuit. Please check the circuit structure and operations."


class QubitError(CircuitError):
    """Errors related to qubit operations."""
    
    def _generate_user_message(self) -> str:
        return "Qubit error: Check qubit allocation, indices, and usage."


class GateError(CircuitError):
    """Errors related to quantum gate operations."""
    
    def _generate_user_message(self) -> str:
        return "Gate error: Check gate parameters, target qubits, and operation validity."


class MeasurementError(CircuitError):
    """Errors related to measurement operations."""
    
    def _generate_user_message(self) -> str:
        return "Measurement error: Check measurement targets and classical register allocation."


# =============================================================================
# Execution Errors
# =============================================================================

class ExecutionError(QuantumPlatformError):
    """Base class for execution errors."""
    
    def _generate_user_message(self) -> str:
        return "Error during quantum program execution."


class SimulationError(ExecutionError):
    """Errors during quantum simulation."""
    
    def _generate_user_message(self) -> str:
        return "Simulation failed. Check circuit size, complexity, and available system resources."


class HardwareError(ExecutionError):
    """Errors during hardware execution."""
    
    def _generate_user_message(self) -> str:
        return "Hardware execution failed. Check device availability and circuit compatibility."


# =============================================================================
# Compliance Errors
# =============================================================================

class ComplianceError(UserError):
    """Base class for compliance-related errors."""
    
    def _generate_user_message(self) -> str:
        return "Compliance violation detected. Please review the requirements and constraints."


class ResourceLimitError(ComplianceError):
    """Errors when resource limits are exceeded."""
    
    def _generate_user_message(self) -> str:
        return "Resource limit exceeded. Reduce circuit size or complexity, or select a different target."


class SecurityError(ComplianceError):
    """Security-related errors."""
    
    def _generate_user_message(self) -> str:
        return "Security policy violation. Please check permissions and authentication."


# =============================================================================
# Serialization Errors
# =============================================================================

class SerializationError(QuantumPlatformError):
    """Base class for serialization errors."""
    
    def _generate_user_message(self) -> str:
        return "Error during circuit serialization or deserialization."


class ImportError(SerializationError):
    """Errors during circuit import."""
    
    def _generate_user_message(self) -> str:
        return "Failed to import circuit. Check file format and content validity."


class ExportError(SerializationError):
    """Errors during circuit export."""
    
    def _generate_user_message(self) -> str:
        return "Failed to export circuit. Check target format compatibility."


# =============================================================================
# Infrastructure Errors
# =============================================================================

class ConfigurationError(SystemError):
    """Configuration-related errors."""
    
    def _generate_user_message(self) -> str:
        return "Configuration error. Check platform settings and configuration files."


class PluginError(SystemError):
    """Plugin-related errors."""
    
    def _generate_user_message(self) -> str:
        return "Plugin error. A plugin may be incompatible or corrupted."


class NetworkError(SystemError):
    """Network-related errors."""
    
    def _generate_user_message(self) -> str:
        return "Network error. Check internet connection and service availability."


# =============================================================================
# Error Creation Utilities
# =============================================================================

def create_parse_error(
    message: str,
    line_number: Optional[int] = None,
    file_path: Optional[str] = None,
    user_input: Optional[str] = None,
    suggestions: Optional[List[str]] = None
) -> ParseError:
    """Create a parse error with context."""
    context = ErrorContext(
        component="Compiler",
        operation="Parse",
        line_number=line_number,
        file_path=file_path,
        user_input=user_input
    )
    
    return ParseError(
        message=message,
        context=context,
        suggestions=suggestions or []
    )


def create_qubit_error(
    message: str,
    qubit_id: Optional[int] = None,
    operation: Optional[str] = None,
    suggestions: Optional[List[str]] = None
) -> QubitError:
    """Create a qubit error with context."""
    context = ErrorContext(
        component="Circuit",
        operation=operation or "Qubit Operation",
        system_state={"qubit_id": qubit_id} if qubit_id is not None else {}
    )
    
    return QubitError(
        message=message,
        context=context,
        suggestions=suggestions or []
    )


def create_resource_limit_error(
    message: str,
    resource_type: str,
    limit: int,
    requested: int,
    suggestions: Optional[List[str]] = None
) -> ResourceLimitError:
    """Create a resource limit error with context."""
    context = ErrorContext(
        component="Compliance",
        operation="Resource Check",
        system_state={
            "resource_type": resource_type,
            "limit": limit,
            "requested": requested
        }
    )
    
    default_suggestions = [
        f"Reduce {resource_type} usage to {limit} or less",
        "Consider using a different target with higher limits",
        "Split your circuit into smaller parts"
    ]
    
    return ResourceLimitError(
        message=message,
        context=context,
        suggestions=suggestions or default_suggestions
    )


def create_simulation_error(
    message: str,
    num_qubits: Optional[int] = None,
    shots: Optional[int] = None,
    memory_required: Optional[float] = None,
    suggestions: Optional[List[str]] = None
) -> SimulationError:
    """Create a simulation error with context."""
    context = ErrorContext(
        component="Simulation",
        operation="Execute",
        system_state={
            "num_qubits": num_qubits,
            "shots": shots,
            "memory_required_gb": memory_required
        }
    )
    
    default_suggestions = []
    if num_qubits and num_qubits > 20:
        default_suggestions.append("Reduce the number of qubits (current: {})".format(num_qubits))
    if shots and shots > 100000:
        default_suggestions.append("Reduce the number of shots (current: {})".format(shots))
    if memory_required and memory_required > 8:
        default_suggestions.append("Circuit requires too much memory ({:.1f}GB)".format(memory_required))
    
    return SimulationError(
        message=message,
        context=context,
        suggestions=suggestions or default_suggestions
    ) 