"""
Observability Integration with Platform Components

This module provides integration points to add logging, monitoring, and debugging
capabilities to existing platform components without requiring major code changes.
"""

import functools
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Type
from quantum_platform.observability.logging import get_logger, LogLevel, setup_logging, configure_logging
from quantum_platform.observability.monitor import get_monitor, measure_performance
from quantum_platform.observability.debug import get_debug_context, debug_operation

class ObservabilityMixin:
    """
    Mixin class to add observability capabilities to existing classes.
    
    Classes can inherit from this mixin to automatically get logging,
    monitoring, and debugging capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize observability capabilities."""
        super().__init__(*args, **kwargs)
        
        # Get component name from class name
        self._component_name = self.__class__.__name__
        
        # Initialize observability tools
        self._logger = get_logger(self._component_name)
        self._monitor = get_monitor()
        self._debug_context = get_debug_context()
        
        # Performance tracking
        self._operation_count = 0
        
    @property
    def logger(self):
        """Get the component logger."""
        return self._logger
    
    @property
    def monitor(self):
        """Get the system monitor."""
        return self._monitor
    
    @property
    def debug_context(self):
        """Get the debug context."""
        return self._debug_context
    
    def log_operation_start(self, operation: str, **kwargs):
        """Log the start of an operation."""
        self._operation_count += 1
        self.logger.info(f"Starting {operation} (operation #{self._operation_count})", extra=kwargs)
    
    def log_operation_success(self, operation: str, duration: Optional[float] = None, **kwargs):
        """Log successful completion of an operation."""
        if duration is not None:
            self.logger.info(f"Successfully completed {operation} in {duration:.3f}s", extra=kwargs)
        else:
            self.logger.info(f"Successfully completed {operation}", extra=kwargs)
    
    def log_operation_error(self, operation: str, error: Exception, duration: Optional[float] = None, **kwargs):
        """Log an operation error."""
        if duration is not None:
            self.logger.error(f"Failed {operation} after {duration:.3f}s: {error}", extra=kwargs)
        else:
            self.logger.error(f"Failed {operation}: {error}", extra=kwargs)
    
    @contextmanager
    def observe_operation(self, operation: str, **metadata):
        """Context manager for comprehensive operation observability."""
        with self.monitor.measure_operation(operation, self._component_name, metadata):
            with self.debug_context.operation_context(operation, self._component_name):
                self.log_operation_start(operation, **metadata)
                try:
                    yield
                    self.log_operation_success(operation)
                except Exception as e:
                    self.log_operation_error(operation, e)
                    raise

def add_observability(cls: Type) -> Type:
    """
    Class decorator to add observability capabilities to existing classes.
    
    Args:
        cls: Class to enhance with observability
        
    Returns:
        Enhanced class with observability capabilities
    """
    # Store original __init__
    original_init = cls.__init__
    
    def new_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        
        # Add observability
        component_name = cls.__name__
        self._logger = get_logger(component_name)
        self._monitor = get_monitor()
        self._debug_context = get_debug_context()
        self._operation_count = 0
        
        # Add convenience methods
        self.logger = self._logger
        self.monitor = self._monitor
        self.debug_context = self._debug_context
        
        self.logger.debug(f"Initialized {component_name} with observability")
    
    # Replace __init__
    cls.__init__ = new_init
    
    # Add observe_operation method
    def observe_operation(self, operation: str, **metadata):
        @contextmanager
        def context():
            with self.monitor.measure_operation(operation, cls.__name__, metadata):
                with self.debug_context.operation_context(operation, cls.__name__):
                    self.logger.info(f"Starting {operation}")
                    try:
                        yield
                        self.logger.info(f"Successfully completed {operation}")
                    except Exception as e:
                        self.logger.error(f"Failed {operation}: {e}")
                        raise
        return context()
    
    cls.observe_operation = observe_operation
    
    return cls

def log_method_calls(component_name: Optional[str] = None, 
                    log_args: bool = False, 
                    log_result: bool = False,
                    performance_tracking: bool = True):
    """
    Decorator to add logging to method calls.
    
    Args:
        component_name: Component name for logging
        log_args: Whether to log method arguments
        log_result: Whether to log method results
        performance_tracking: Whether to track performance
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine component name
            comp_name = component_name
            if not comp_name and args and hasattr(args[0], '__class__'):
                comp_name = args[0].__class__.__name__
            comp_name = comp_name or "Function"
            
            logger = get_logger(comp_name)
            monitor = get_monitor()
            
            # Prepare log message
            func_name = func.__name__
            log_msg = f"Calling {func_name}"
            
            if log_args:
                arg_strs = [str(arg) for arg in args[1:]]  # Skip self
                kwarg_strs = [f"{k}={v}" for k, v in kwargs.items()]
                all_args = arg_strs + kwarg_strs
                if all_args:
                    log_msg += f" with args: {', '.join(all_args[:5])}"  # Limit to first 5 args
                    if len(all_args) > 5:
                        log_msg += "..."
            
            logger.debug(log_msg)
            
            # Execute function with monitoring
            if performance_tracking:
                with monitor.measure_operation(func_name, comp_name):
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Log result if requested
            if log_result:
                result_str = str(result)
                if len(result_str) > 100:
                    result_str = result_str[:100] + "..."
                logger.debug(f"{func_name} returned: {result_str}")
            else:
                logger.debug(f"Completed {func_name}")
            
            return result
        
        return wrapper
    return decorator

class PlatformIntegration:
    """
    Main integration class for adding observability to the platform.
    
    This class provides methods to integrate observability features
    into existing platform components.
    """
    
    def __init__(self):
        """Initialize platform integration."""
        self.logger = get_logger("Integration")
        self.monitor = get_monitor()
        self._integration_registry = {}
    
    def setup_observability(self, 
                          log_level: LogLevel = LogLevel.INFO,
                          log_to_file: bool = True,
                          log_file_path: str = "logs/quantum_platform.log",
                          enable_performance_monitoring: bool = True,
                          enable_debug_mode: bool = False):
        """
        Setup comprehensive observability for the platform.
        
        Args:
            log_level: Global logging level
            log_to_file: Whether to log to file
            log_file_path: Path to log file
            enable_performance_monitoring: Whether to enable performance monitoring
            enable_debug_mode: Whether to enable debug mode
        """
        # Configure logging
        config = configure_logging(
            level=log_level,
            log_to_file=log_to_file,
            log_file_path=log_file_path,
            enable_performance=enable_performance_monitoring
        )
        
        setup_logging(config)
        
        # Setup monitoring
        if enable_performance_monitoring:
            self.monitor.start_continuous_monitoring()
        
        self.logger.info("Observability system initialized")
        self.logger.info(f"Log level: {log_level.name}")
        self.logger.info(f"Performance monitoring: {enable_performance_monitoring}")
        self.logger.info(f"Debug mode: {enable_debug_mode}")
    
    def integrate_component(self, component: Any, component_name: Optional[str] = None):
        """
        Integrate observability into an existing component.
        
        Args:
            component: Component instance to integrate
            component_name: Optional component name override
        """
        comp_name = component_name or component.__class__.__name__
        
        # Add observability attributes
        component._logger = get_logger(comp_name)
        component._monitor = get_monitor()
        component._debug_context = get_debug_context()
        
        # Add convenience properties
        component.logger = component._logger
        component.monitor = component._monitor
        component.debug_context = component._debug_context
        
        # Add observe_operation method
        def observe_operation(operation: str, **metadata):
            @contextmanager
            def context():
                with component.monitor.measure_operation(operation, comp_name, metadata):
                    with component.debug_context.operation_context(operation, comp_name):
                        component.logger.info(f"Starting {operation}")
                        try:
                            yield
                            component.logger.info(f"Successfully completed {operation}")
                        except Exception as e:
                            component.logger.error(f"Failed {operation}: {e}")
                            raise
            return context()
        
        component.observe_operation = observe_operation
        
        # Register integration
        self._integration_registry[comp_name] = {
            'component': component,
            'integration_time': threading.current_thread().ident
        }
        
        self.logger.info(f"Integrated observability into {comp_name}")
    
    def enhance_existing_components(self):
        """
        Automatically enhance existing platform components with observability.
        
        This method attempts to find and enhance key platform components.
        """
        try:
            # Try to enhance compiler components
            self._enhance_compiler_components()
            
            # Try to enhance simulation components
            self._enhance_simulation_components()
            
            # Try to enhance security components
            self._enhance_security_components()
            
            # Try to enhance plugin components
            self._enhance_plugin_components()
            
        except Exception as e:
            self.logger.error(f"Error enhancing existing components: {e}")
    
    def _enhance_compiler_components(self):
        """Enhance compiler components with observability."""
        try:
            from quantum_platform.compiler.ir.circuit import QuantumCircuit
            from quantum_platform.compiler.gates.factory import GateFactory
            
            # Add observability to QuantumCircuit
            original_add_operation = QuantumCircuit.add_operation
            
            @log_method_calls("Circuit", performance_tracking=True)
            def enhanced_add_operation(self, operation):
                return original_add_operation(self, operation)
            
            QuantumCircuit.add_operation = enhanced_add_operation
            
            self.logger.info("Enhanced compiler components")
            
        except ImportError:
            self.logger.debug("Compiler components not available for enhancement")
    
    def _enhance_simulation_components(self):
        """Enhance simulation components with observability."""
        try:
            from quantum_platform.simulation.statevector import StateVectorSimulator
            
            # Add observability to simulator
            original_run = StateVectorSimulator.run
            
            @log_method_calls("Simulator", performance_tracking=True)
            def enhanced_run(self, circuit, shots=1, initial_state=None):
                return original_run(self, circuit, shots, initial_state)
            
            StateVectorSimulator.run = enhanced_run
            
            self.logger.info("Enhanced simulation components")
            
        except ImportError:
            self.logger.debug("Simulation components not available for enhancement")
    
    def _enhance_security_components(self):
        """Enhance security components with observability."""
        try:
            from quantum_platform.security.enforcement import require_permission
            
            # Security components already have audit logging
            self.logger.info("Security components already have comprehensive logging")
            
        except ImportError:
            self.logger.debug("Security components not available for enhancement")
    
    def _enhance_plugin_components(self):
        """Enhance plugin components with observability."""
        try:
            from quantum_platform.plugins.manager import PluginManager
            
            # Add observability to plugin manager
            original_load_plugin = PluginManager.load_plugin
            
            @log_method_calls("PluginManager", performance_tracking=True)
            def enhanced_load_plugin(self, plugin_info):
                return original_load_plugin(self, plugin_info)
            
            PluginManager.load_plugin = enhanced_load_plugin
            
            self.logger.info("Enhanced plugin components")
            
        except ImportError:
            self.logger.debug("Plugin components not available for enhancement")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of observability integrations."""
        return {
            'integrated_components': list(self._integration_registry.keys()),
            'total_integrations': len(self._integration_registry),
            'integration_details': {
                name: {
                    'component_type': info['component'].__class__.__name__,
                    'has_logger': hasattr(info['component'], 'logger'),
                    'has_monitor': hasattr(info['component'], 'monitor'),
                    'has_debug_context': hasattr(info['component'], 'debug_context')
                }
                for name, info in self._integration_registry.items()
            }
        }

# Global integration instance
_global_integration: Optional[PlatformIntegration] = None
_integration_lock = threading.Lock()

def get_integration() -> PlatformIntegration:
    """Get the global platform integration instance."""
    global _global_integration
    
    with _integration_lock:
        if _global_integration is None:
            _global_integration = PlatformIntegration()
    
    return _global_integration

def initialize_observability(**kwargs):
    """
    Initialize observability for the entire platform.
    
    This is the main entry point for setting up observability.
    """
    integration = get_integration()
    integration.setup_observability(**kwargs)
    integration.enhance_existing_components()
    return integration 