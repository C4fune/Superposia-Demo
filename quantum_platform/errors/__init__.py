"""
Quantum Platform Error Handling System

This module provides comprehensive error handling, reporting, and alerting
capabilities for the quantum platform.
"""

from .exceptions import (
    # Base exceptions
    QuantumPlatformError,
    UserError,
    SystemError,
    
    # Compiler errors
    CompilationError,
    ParseError,
    OptimizationError,
    TranspilationError,
    
    # Circuit errors
    CircuitError,
    QubitError,
    GateError,
    MeasurementError,
    
    # Execution errors
    ExecutionError,
    SimulationError,
    HardwareError,
    
    # Compliance errors
    ComplianceError,
    ResourceLimitError,
    SecurityError,
    
    # Serialization errors
    SerializationError,
    ImportError,
    ExportError,
    
    # Infrastructure errors
    ConfigurationError,
    PluginError,
    NetworkError,
    
    # Helper functions
    create_parse_error,
    create_qubit_error,
    create_resource_limit_error,
    create_simulation_error
)

from .reporter import (
    ErrorReporter,
    ErrorReport,
    ErrorContext,
    get_error_reporter,
    report_error
)

from .handler import (
    ErrorHandler,
    ErrorLevel,
    ErrorCategory,
    format_error_message,
    create_user_friendly_message
)

from .alerts import (
    AlertManager,
    AlertType,
    AlertSeverity,
    get_alert_manager,
    create_error_alert
)

__all__ = [
    # Exceptions
    'QuantumPlatformError', 'UserError', 'SystemError',
    'CompilationError', 'ParseError', 'OptimizationError', 'TranspilationError',
    'CircuitError', 'QubitError', 'GateError', 'MeasurementError',
    'ExecutionError', 'SimulationError', 'HardwareError',
    'ComplianceError', 'ResourceLimitError', 'SecurityError',
    'SerializationError', 'ImportError', 'ExportError',
    'ConfigurationError', 'PluginError', 'NetworkError',
    
    # Helper functions
    'create_parse_error', 'create_qubit_error', 'create_resource_limit_error', 'create_simulation_error',
    
    # Reporting
    'ErrorReporter', 'ErrorReport', 'ErrorContext', 'get_error_reporter', 'report_error',
    
    # Handling
    'ErrorHandler', 'ErrorLevel', 'ErrorCategory',
    'format_error_message', 'create_user_friendly_message',
    
    # Alerts
    'AlertManager', 'AlertType', 'AlertSeverity', 'get_alert_manager', 'create_error_alert'
] 