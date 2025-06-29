"""
Unified Logging System for Quantum Computing Platform

This module provides a comprehensive logging framework that all platform components
use to record information, warnings, and errors. It includes configurable log levels,
formatted output, file rotation, and performance-aware logging.
"""

import logging
import logging.handlers
import os
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from contextlib import contextmanager

class LogLevel(Enum):
    """Enumeration of available log levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO  
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class LogFormat(Enum):
    """Predefined log format templates."""
    STANDARD = "%(asctime)s [%(component)s] %(levelname)s: %(message)s"
    DETAILED = "%(asctime)s [%(component)s] %(levelname)s [%(funcName)s:%(lineno)d]: %(message)s"
    COMPACT = "%(asctime)s [%(component)s] %(levelname)-7s: %(message)s"
    DEBUG = "%(asctime)s [%(component)s] %(levelname)s [%(module)s.%(funcName)s:%(lineno)d] [Thread-%(thread)d]: %(message)s"

@dataclass
class LogConfig:
    """Configuration for the logging system."""
    level: LogLevel = LogLevel.INFO
    format_template: LogFormat = LogFormat.STANDARD
    log_to_console: bool = True
    log_to_file: bool = True
    log_file_path: str = "logs/quantum_platform.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    date_format: str = "%Y-%m-%d %H:%M:%S"
    component_name: Optional[str] = None
    enable_performance_logging: bool = False
    context_data: Optional[Dict[str, Any]] = None

class ComponentLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds component context to log records."""
    
    def __init__(self, logger: logging.Logger, component: str, extra: Optional[Dict[str, Any]] = None):
        self.component = component
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """Add component information to log records."""
        extra = kwargs.get('extra', {})
        extra['component'] = self.component
        
        # Add context data if available
        if hasattr(self, 'extra') and self.extra:
            extra.update(self.extra)
            
        kwargs['extra'] = extra
        return msg, kwargs

class QuantumLogger:
    """
    Main logging class that provides unified logging across the platform.
    
    This class manages logger configuration, provides component-specific loggers,
    and handles performance-aware logging with configurable output destinations.
    """
    
    _instance: Optional['QuantumLogger'] = None
    _lock = threading.Lock()
    _loggers: Dict[str, ComponentLoggerAdapter] = {}
    
    def __new__(cls, config: Optional[LogConfig] = None):
        """Singleton pattern to ensure unified logging configuration."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: Optional[LogConfig] = None):
        """Initialize the quantum logger with the given configuration."""
        if self._initialized:
            return
            
        self.config = config or LogConfig()
        self._setup_logging()
        self._initialized = True
        self._performance_cache = {}
        self._context_stack = threading.local()
    
    def _setup_logging(self):
        """Setup the logging configuration."""
        # Create logs directory if it doesn't exist
        log_dir = Path(self.config.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger('quantum_platform')
        root_logger.setLevel(self.config.level.value)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter(
            fmt=self.config.format_template.value,
            datefmt=self.config.date_format
        )
        
        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.config.level.value)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.log_to_file:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=self.config.log_file_path,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(self.config.level.value)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def get_logger(self, component: str, extra_context: Optional[Dict[str, Any]] = None) -> ComponentLoggerAdapter:
        """
        Get a component-specific logger.
        
        Args:
            component: Name of the component (e.g., 'Compiler', 'Simulator', 'Security')
            extra_context: Additional context data to include in logs
            
        Returns:
            ComponentLoggerAdapter configured for the component
        """
        logger_key = f"{component}:{hash(str(extra_context))}"
        
        if logger_key not in self._loggers:
            base_logger = logging.getLogger(f'quantum_platform.{component.lower()}')
            self._loggers[logger_key] = ComponentLoggerAdapter(
                base_logger, component, extra_context
            )
        
        return self._loggers[logger_key]
    
    def update_config(self, new_config: LogConfig):
        """Update the logging configuration."""
        self.config = new_config
        self._setup_logging()
        
        # Update existing loggers
        for logger_adapter in self._loggers.values():
            logger_adapter.logger.setLevel(new_config.level.value)
    
    def set_level(self, level: LogLevel):
        """Set the global log level."""
        self.config.level = level
        root_logger = logging.getLogger('quantum_platform')
        root_logger.setLevel(level.value)
        
        # Update all handlers
        for handler in root_logger.handlers:
            handler.setLevel(level.value)
    
    @contextmanager
    def performance_context(self, operation: str, component: str = "Platform"):
        """
        Context manager for performance-aware logging.
        
        Args:
            operation: Name of the operation being timed
            component: Component performing the operation
        """
        if not self.config.enable_performance_logging:
            yield
            return
            
        logger = self.get_logger(component)
        start_time = time.time()
        
        logger.debug(f"Starting {operation}")
        
        try:
            yield
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed {operation} after {duration:.3f}s: {e}")
            raise
        else:
            duration = time.time() - start_time
            if duration > 1.0:  # Log slow operations at INFO level
                logger.info(f"Completed {operation} in {duration:.3f}s")
            else:
                logger.debug(f"Completed {operation} in {duration:.3f}s")
    
    @contextmanager
    def user_context(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """
        Context manager for user/session-specific logging.
        
        Args:
            user_id: User identifier for multi-user scenarios
            session_id: Session identifier for tracking
        """
        if not hasattr(self._context_stack, 'contexts'):
            self._context_stack.contexts = []
        
        context = {}
        if user_id:
            context['user_id'] = user_id
        if session_id:
            context['session_id'] = session_id
        
        self._context_stack.contexts.append(context)
        
        try:
            yield
        finally:
            self._context_stack.contexts.pop()
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get the current logging context."""
        if not hasattr(self._context_stack, 'contexts'):
            return {}
        
        combined_context = {}
        for context in self._context_stack.contexts:
            combined_context.update(context)
        return combined_context
    
    def log_system_info(self):
        """Log system information at startup."""
        logger = self.get_logger("System")
        logger.info("Quantum Computing Platform starting up")
        logger.info(f"Log level: {self.config.level.name}")
        logger.info(f"Logging to file: {self.config.log_to_file}")
        if self.config.log_to_file:
            logger.info(f"Log file: {self.config.log_file_path}")
        logger.info(f"Performance logging: {self.config.enable_performance_logging}")

# Global logging instance
_global_logger: Optional[QuantumLogger] = None
_setup_lock = threading.Lock()

def setup_logging(config: Optional[LogConfig] = None) -> QuantumLogger:
    """
    Setup the global logging system.
    
    Args:
        config: Logging configuration. If None, uses default configuration.
        
    Returns:
        QuantumLogger instance
    """
    global _global_logger
    
    with _setup_lock:
        if _global_logger is None:
            _global_logger = QuantumLogger(config)
            _global_logger.log_system_info()
        elif config is not None:
            _global_logger.update_config(config)
    
    return _global_logger

def get_logger(component: str, extra_context: Optional[Dict[str, Any]] = None) -> ComponentLoggerAdapter:
    """
    Get a component-specific logger from the global logging system.
    
    Args:
        component: Name of the component
        extra_context: Additional context data
        
    Returns:
        ComponentLoggerAdapter for the component
    """
    if _global_logger is None:
        setup_logging()
    
    return _global_logger.get_logger(component, extra_context)

def configure_logging(
    level: Optional[LogLevel] = None,
    log_to_console: Optional[bool] = None,
    log_to_file: Optional[bool] = None,
    log_file_path: Optional[str] = None,
    enable_performance: Optional[bool] = None,
    format_template: Optional[LogFormat] = None
) -> LogConfig:
    """
    Configure logging with specific parameters.
    
    Args:
        level: Log level to set
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_file_path: Path to log file
        enable_performance: Whether to enable performance logging
        format_template: Log format template to use
        
    Returns:
        LogConfig with the specified configuration
    """
    config = LogConfig()
    
    if level is not None:
        config.level = level
    if log_to_console is not None:
        config.log_to_console = log_to_console
    if log_to_file is not None:
        config.log_to_file = log_to_file
    if log_file_path is not None:
        config.log_file_path = log_file_path
    if enable_performance is not None:
        config.enable_performance_logging = enable_performance
    if format_template is not None:
        config.format_template = format_template
    
    return config

# Convenience functions for common logging operations
def log_info(component: str, message: str, **kwargs):
    """Log an info message."""
    logger = get_logger(component)
    logger.info(message, **kwargs)

def log_warning(component: str, message: str, **kwargs):
    """Log a warning message."""
    logger = get_logger(component)
    logger.warning(message, **kwargs)

def log_error(component: str, message: str, **kwargs):
    """Log an error message."""
    logger = get_logger(component)
    logger.error(message, **kwargs)

def log_debug(component: str, message: str, **kwargs):
    """Log a debug message."""
    logger = get_logger(component)
    logger.debug(message, **kwargs) 