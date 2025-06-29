"""
Error Handling Decorators

This module provides decorators for automatic error handling,
reporting, and user-friendly error display.
"""

import functools
import traceback
from typing import Callable, Optional, Any, Dict, List, Type, Union
from .exceptions import QuantumPlatformError, SystemError
from .reporter import get_error_reporter, ErrorContext
from .alerts import get_alert_manager, AlertType, AlertSeverity
from .handler import format_error_message

from quantum_platform.observability.logging import get_logger


def handle_errors(
    component: str = "",
    operation: str = "",
    user_message: Optional[str] = None,
    show_alert: bool = True,
    report_error: bool = True,
    reraise: bool = False,
    fallback_return: Any = None,
    expected_exceptions: Optional[List[Type[Exception]]] = None
):
    """
    Decorator for comprehensive error handling.
    
    Args:
        component: Component name for context
        operation: Operation name for context
        user_message: Custom user message for errors
        show_alert: Whether to show user alert
        report_error: Whether to report error
        reraise: Whether to reraise the exception
        fallback_return: Value to return on error (if not reraising)
        expected_exceptions: List of expected exception types to handle differently
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(component or func.__module__)
            
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                # Create error context
                context = ErrorContext(
                    component=component or func.__module__,
                    operation=operation or func.__name__,
                    user_action=f"Called {func.__name__}"
                )
                
                # Check if this is an expected exception
                is_expected = (expected_exceptions and 
                              any(isinstance(e, exc_type) for exc_type in expected_exceptions))
                
                # Log the error
                if is_expected:
                    logger.warning(f"Expected error in {func.__name__}: {e}")
                else:
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                
                # Report error if requested
                if report_error and not is_expected:
                    try:
                        reporter = get_error_reporter()
                        reporter.collect_error(e, context)
                    except Exception as report_err:
                        logger.error(f"Failed to report error: {report_err}")
                
                # Show alert if requested
                if show_alert:
                    try:
                        alert_manager = get_alert_manager()
                        
                        # Format error message
                        formatted = format_error_message(e)
                        
                        # Determine alert type
                        alert_type = AlertType.WARNING if is_expected else AlertType.ERROR
                        severity = AlertSeverity.MEDIUM if is_expected else AlertSeverity.HIGH
                        
                        alert_manager.create_alert(
                            title=formatted.title,
                            message=user_message or formatted.message,
                            alert_type=alert_type,
                            severity=severity,
                            component=component,
                            metadata={
                                'error_code': formatted.error_code,
                                'suggestions': formatted.suggestions,
                                'function': func.__name__
                            }
                        )
                    except Exception as alert_err:
                        logger.error(f"Failed to show alert: {alert_err}")
                
                # Reraise or return fallback
                if reraise or (isinstance(e, QuantumPlatformError) and not is_expected):
                    raise
                else:
                    return fallback_return
        
        return wrapper
    return decorator


def catch_and_report(
    component: str = "",
    operation: str = "",
    fallback_return: Any = None
):
    """
    Simple decorator for catching and reporting errors without user alerts.
    
    Suitable for background operations or API functions.
    """
    return handle_errors(
        component=component,
        operation=operation,
        show_alert=False,
        report_error=True,
        reraise=False,
        fallback_return=fallback_return
    )


def user_friendly_errors(
    component: str = "",
    operation: str = "",
    user_message: str = "An error occurred. Please try again."
):
    """
    Decorator for user-facing operations that should show friendly error messages.
    
    Always shows alerts but doesn't reraise exceptions.
    """
    return handle_errors(
        component=component,
        operation=operation,
        user_message=user_message,
        show_alert=True,
        report_error=True,
        reraise=False
    )


def critical_operation(
    component: str = "",
    operation: str = "",
    alert_title: str = "Critical Error"
):
    """
    Decorator for critical operations that should always reraise exceptions.
    
    Shows alerts and reports errors but still raises the exception.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Always show critical alert
                try:
                    alert_manager = get_alert_manager()
                    alert_manager.create_alert(
                        title=alert_title,
                        message=f"Critical error in {operation or func.__name__}: {e}",
                        alert_type=AlertType.ERROR,
                        severity=AlertSeverity.CRITICAL,
                        component=component,
                        persistent=True,
                        auto_dismiss=False
                    )
                except Exception:
                    pass  # Don't let alert failure prevent reraise
                
                # Always reraise critical errors
                raise
        
        return wrapper
    return decorator


def validate_input(
    validation_func: Callable,
    error_message: str = "Invalid input",
    user_message: Optional[str] = None
):
    """
    Decorator for input validation with automatic error handling.
    
    Args:
        validation_func: Function that validates input and raises ValueError if invalid
        error_message: Technical error message
        user_message: User-friendly error message
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Validate input
                validation_func(*args, **kwargs)
            except ValueError as e:
                # Create validation error
                from .exceptions import UserError
                validation_error = UserError(
                    message=f"{error_message}: {e}",
                    user_message=user_message or f"Invalid input: {e}"
                )
                
                # Show user alert
                alert_manager = get_alert_manager()
                alert_manager.warning_alert(
                    title="Input Validation Error",
                    message=validation_error.user_message,
                    component=func.__module__
                )
                
                raise validation_error
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def safe_execute(
    default_return: Any = None,
    log_errors: bool = True,
    component: str = ""
):
    """
    Decorator for operations that should never fail.
    
    Catches all exceptions and returns a default value.
    Useful for cleanup operations or non-critical features.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = get_logger(component or func.__module__)
                    logger.warning(f"Safe execution failed in {func.__name__}: {e}")
                
                return default_return
        
        return wrapper
    return decorator


def timeout_with_error(
    timeout_seconds: float,
    error_message: str = "Operation timed out",
    component: str = ""
):
    """
    Decorator that adds timeout functionality with proper error handling.
    
    Args:
        timeout_seconds: Timeout in seconds
        error_message: Error message for timeout
        component: Component name for context
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                from .exceptions import ExecutionError
                raise ExecutionError(
                    message=f"Function {func.__name__} timed out after {timeout_seconds}s",
                    user_message=error_message,
                    severity="high"
                )
            
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel timeout
                return result
            except Exception as e:
                signal.alarm(0)  # Cancel timeout
                
                # Handle timeout errors specially
                if isinstance(e, ExecutionError) and "timed out" in str(e):
                    alert_manager = get_alert_manager()
                    alert_manager.warning_alert(
                        title="Operation Timeout",
                        message=error_message,
                        component=component
                    )
                
                raise
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


# Convenience decorators for common use cases
def compiler_errors(operation: str = ""):
    """Decorator for compiler operations."""
    return handle_errors(
        component="Compiler",
        operation=operation,
        show_alert=True,
        report_error=True,
        reraise=True
    )


def simulation_errors(operation: str = ""):
    """Decorator for simulation operations."""
    return handle_errors(
        component="Simulation",
        operation=operation,
        show_alert=True,
        report_error=True,
        reraise=False,
        fallback_return=None
    )


def ui_errors(operation: str = ""):
    """Decorator for UI operations."""
    return handle_errors(
        component="UI",
        operation=operation,
        show_alert=True,
        report_error=False,
        reraise=False
    )


def hardware_errors(operation: str = ""):
    """Decorator for hardware operations."""
    return handle_errors(
        component="Hardware",
        operation=operation,
        show_alert=True,
        report_error=True,
        reraise=True,
        expected_exceptions=[ConnectionError, TimeoutError]
    ) 