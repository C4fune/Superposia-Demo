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
    ParameterError,
    MeasurementError,
    
    # Execution errors
    ExecutionError,
    SimulationError,
    HardwareError,
    MitigationError,
    
    # Compliance errors
    ComplianceError,
    ResourceLimitError,
    SecurityError,
    
    # Serialization errors
    SerializationError,
    ImportError as QuantumImportError,
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

from .decorator import (
    handle_errors,
    catch_and_report,
    user_friendly_errors,
    critical_operation,
    safe_execute,
    compiler_errors,
    simulation_errors,
    ui_errors,
    hardware_errors
)

__all__ = [
    # Exceptions
    'QuantumPlatformError', 'UserError', 'SystemError',
    'CompilationError', 'ParseError', 'OptimizationError', 'TranspilationError',
    'CircuitError', 'QubitError', 'GateError', 'ParameterError', 'MeasurementError',
    'ExecutionError', 'SimulationError', 'HardwareError', 'MitigationError',
    'ComplianceError', 'ResourceLimitError', 'SecurityError',
    'SerializationError', 'QuantumImportError', 'ExportError',
    'ConfigurationError', 'PluginError', 'NetworkError',
    
    # Helper functions
    'create_parse_error', 'create_qubit_error', 'create_resource_limit_error', 'create_simulation_error',
    
    # Reporting
    'ErrorReporter', 'ErrorReport', 'ErrorContext', 'get_error_reporter', 'report_error',
    
    # Handling
    'ErrorHandler', 'ErrorLevel', 'ErrorCategory',
    'format_error_message', 'create_user_friendly_message',
    
    # Alerts
    'AlertManager', 'AlertType', 'AlertSeverity', 'get_alert_manager', 'create_error_alert',
    
    # Decorators
    'handle_errors', 'catch_and_report', 'user_friendly_errors', 'critical_operation',
    'safe_execute', 'compiler_errors', 'simulation_errors', 'ui_errors', 'hardware_errors'
] 