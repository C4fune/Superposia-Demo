"""
Error Handler and Message Formatting

This module provides error handling utilities for creating user-friendly
error messages and managing error display.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from quantum_platform.observability.logging import get_logger


class ErrorLevel(Enum):
    """Error severity levels for display."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for user guidance."""
    SYNTAX = "syntax"
    LOGIC = "logic"
    RESOURCE = "resource"
    HARDWARE = "hardware"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    INTERNAL = "internal"


@dataclass
class FormattedError:
    """Formatted error message for display."""
    title: str
    message: str
    details: str
    suggestions: List[str]
    error_code: str
    level: ErrorLevel
    category: ErrorCategory
    show_details: bool = False


class ErrorHandler:
    """
    Handles error formatting and user message creation.
    
    Provides consistent, user-friendly error messages across the platform.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Error message templates
        self._templates = {
            ErrorCategory.SYNTAX: {
                'title': 'Syntax Error',
                'icon': '⚠️',
                'suggestions': [
                    'Check for typos in your quantum program',
                    'Verify proper syntax and indentation',
                    'Ensure all parentheses and brackets are balanced'
                ]
            },
            ErrorCategory.LOGIC: {
                'title': 'Logic Error',
                'icon': '🔍',
                'suggestions': [
                    'Review your quantum circuit logic',
                    'Check qubit indices and gate parameters',
                    'Verify measurement and classical register usage'
                ]
            },
            ErrorCategory.RESOURCE: {
                'title': 'Resource Limit',
                'icon': '📊',
                'suggestions': [
                    'Reduce circuit size or complexity',
                    'Use fewer qubits or shots',
                    'Choose a different target device'
                ]
            },
            ErrorCategory.HARDWARE: {
                'title': 'Hardware Error',
                'icon': '🔧',
                'suggestions': [
                    'Check device availability and status',
                    'Verify circuit compatibility with target hardware',
                    'Try again later or use a different device'
                ]
            },
            ErrorCategory.NETWORK: {
                'title': 'Network Error',
                'icon': '🌐',
                'suggestions': [
                    'Check your internet connection',
                    'Verify service availability',
                    'Try again in a few moments'
                ]
            },
            ErrorCategory.CONFIGURATION: {
                'title': 'Configuration Error',
                'icon': '⚙️',
                'suggestions': [
                    'Check platform settings and configuration',
                    'Verify credentials and permissions',
                    'Reset to default configuration if needed'
                ]
            },
            ErrorCategory.INTERNAL: {
                'title': 'Internal Error',
                'icon': '🐛',
                'suggestions': [
                    'This appears to be a platform issue',
                    'Try restarting the application',
                    'Report this error to the development team'
                ]
            }
        }
    
    def format_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> FormattedError:
        """
        Format an exception into a user-friendly error message.
        
        Args:
            exception: The exception to format
            context: Additional context information
            
        Returns:
            Formatted error message
        """
        # Get error attributes
        error_type = exception.__class__.__name__
        error_message = str(exception)
        user_message = getattr(exception, 'user_message', error_message)
        error_code = getattr(exception, 'error_code', f"QP{hash(error_message) % 10000:04d}")
        suggestions = getattr(exception, 'suggestions', [])
        
        # Determine category and level
        category = self._categorize_error(exception, error_type)
        level = self._determine_level(exception)
        
        # Get template
        template = self._templates.get(category, self._templates[ErrorCategory.INTERNAL])
        
        # Create formatted message
        title = template['title']
        if hasattr(exception, 'context') and exception.context.component:
            title += f" in {exception.context.component}"
        
        # Enhance message with context
        enhanced_message = self._enhance_message(user_message, context)
        
        # Combine suggestions
        all_suggestions = suggestions + template['suggestions']
        unique_suggestions = list(dict.fromkeys(all_suggestions))  # Remove duplicates
        
        return FormattedError(
            title=title,
            message=enhanced_message,
            details=error_message if error_message != user_message else "",
            suggestions=unique_suggestions[:5],  # Limit to 5 suggestions
            error_code=error_code,
            level=level,
            category=category
        )
    
    def _categorize_error(self, exception: Exception, error_type: str) -> ErrorCategory:
        """Categorize an error based on its type and context."""
        # Import here to avoid circular imports
        from .exceptions import (
            ParseError, CompilationError, QubitError, GateError,
            ResourceLimitError, HardwareError, NetworkError, 
            ConfigurationError, SystemError
        )
        
        if isinstance(exception, ParseError):
            return ErrorCategory.SYNTAX
        elif isinstance(exception, (QubitError, GateError)):
            return ErrorCategory.LOGIC
        elif isinstance(exception, ResourceLimitError):
            return ErrorCategory.RESOURCE
        elif isinstance(exception, HardwareError):
            return ErrorCategory.HARDWARE
        elif isinstance(exception, NetworkError):
            return ErrorCategory.NETWORK
        elif isinstance(exception, ConfigurationError):
            return ErrorCategory.CONFIGURATION
        elif isinstance(exception, SystemError):
            return ErrorCategory.INTERNAL
        else:
            # Fallback based on error message keywords
            message_lower = str(exception).lower()
            if any(word in message_lower for word in ['syntax', 'parse', 'invalid']):
                return ErrorCategory.SYNTAX
            elif any(word in message_lower for word in ['qubit', 'gate', 'circuit']):
                return ErrorCategory.LOGIC
            elif any(word in message_lower for word in ['memory', 'limit', 'resource']):
                return ErrorCategory.RESOURCE
            elif any(word in message_lower for word in ['network', 'connection', 'timeout']):
                return ErrorCategory.NETWORK
            else:
                return ErrorCategory.INTERNAL
    
    def _determine_level(self, exception: Exception) -> ErrorLevel:
        """Determine the severity level of an error."""
        if hasattr(exception, 'severity'):
            severity = exception.severity
            if hasattr(severity, 'value'):
                severity = severity.value
            
            severity_map = {
                'low': ErrorLevel.INFO,
                'medium': ErrorLevel.WARNING,
                'high': ErrorLevel.ERROR,
                'critical': ErrorLevel.CRITICAL
            }
            return severity_map.get(severity, ErrorLevel.ERROR)
        
        # Fallback based on exception type
        from .exceptions import SystemError, ResourceLimitError
        
        if isinstance(exception, SystemError):
            return ErrorLevel.CRITICAL
        elif isinstance(exception, ResourceLimitError):
            return ErrorLevel.WARNING
        else:
            return ErrorLevel.ERROR
    
    def _enhance_message(self, message: str, context: Optional[Dict[str, Any]]) -> str:
        """Enhance error message with context information."""
        if not context:
            return message
        
        enhanced = message
        
        # Add line number if available
        if 'line_number' in context:
            enhanced += f" (Line {context['line_number']})"
        
        # Add operation context
        if 'operation' in context:
            enhanced += f" during {context['operation']}"
        
        # Add specific details based on context
        if 'qubit_id' in context:
            enhanced += f" (Qubit {context['qubit_id']})"
        
        if 'gate_name' in context:
            enhanced += f" (Gate: {context['gate_name']})"
        
        return enhanced


def format_error_message(exception: Exception, context: Optional[Dict[str, Any]] = None) -> FormattedError:
    """Convenience function to format an error message."""
    handler = ErrorHandler()
    return handler.format_error(exception, context)


def create_user_friendly_message(
    error_type: str,
    message: str,
    suggestions: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a user-friendly error message from basic components.
    
    Args:
        error_type: Type of error
        message: Error message
        suggestions: List of suggestions
        context: Additional context
        
    Returns:
        Formatted user-friendly message
    """
    lines = [f"❌ {error_type}: {message}"]
    
    if context:
        if 'line_number' in context:
            lines.append(f"📍 Location: Line {context['line_number']}")
        if 'component' in context:
            lines.append(f"🔧 Component: {context['component']}")
    
    if suggestions:
        lines.append("\n💡 Suggestions:")
        for i, suggestion in enumerate(suggestions[:3], 1):
            lines.append(f"  {i}. {suggestion}")
    
    return "\n".join(lines)


def extract_line_number(traceback_str: str) -> Optional[int]:
    """Extract line number from traceback string."""
    # Look for line number patterns in traceback
    patterns = [
        r'line (\d+)',
        r', line (\d+),',
        r'File ".*", line (\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, traceback_str)
        if match:
            return int(match.group(1))
    
    return None


def extract_error_location(traceback_str: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract file path and line number from traceback."""
    # Look for file and line patterns
    pattern = r'File "(.*)", line (\d+)'
    match = re.search(pattern, traceback_str)
    
    if match:
        file_path = match.group(1)
        line_number = int(match.group(2))
        return file_path, line_number
    
    return None, None 