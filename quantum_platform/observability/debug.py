"""
Debugging Tools and Context Managers

This module provides debugging aids for quantum circuits, simulations, and general
platform operations. It includes context managers for detailed debugging, circuit
analysis tools, and simulation state inspection capabilities.
"""

import threading
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union, Set
from datetime import datetime
from contextlib import contextmanager
from quantum_platform.observability.logging import get_logger
from quantum_platform.observability.monitor import get_monitor

@dataclass
class DebugEvent:
    """Container for debug event information."""
    timestamp: datetime
    event_type: str
    component: str
    operation: str
    details: Dict[str, Any]
    stack_trace: Optional[str] = None
    thread_id: int = field(default_factory=lambda: threading.get_ident())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert debug event to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'component': self.component,
            'operation': self.operation,
            'details': self.details,
            'stack_trace': self.stack_trace,
            'thread_id': self.thread_id
        }

class DebugContext:
    """
    Debug context manager for detailed operation tracking.
    
    This class provides comprehensive debugging capabilities including
    operation tracking, variable monitoring, and detailed error analysis.
    """
    
    def __init__(self, max_events: int = 1000):
        """
        Initialize debug context.
        
        Args:
            max_events: Maximum number of debug events to keep in memory
        """
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.active_contexts: Dict[int, Dict[str, Any]] = {}
        self.watched_variables: Dict[str, Any] = {}
        self.breakpoints: Set[str] = set()
        
        # Threading
        self._lock = threading.RLock()
        
        # Logger
        self.logger = get_logger("Debug")
        
        # Monitor integration
        self.monitor = get_monitor()
    
    @contextmanager
    def operation_context(self, operation: str, component: str = "Platform", 
                         watch_variables: Optional[Dict[str, Any]] = None,
                         enable_stack_trace: bool = False):
        """
        Context manager for debugging specific operations.
        
        Args:
            operation: Name of the operation being debugged
            component: Component performing the operation
            watch_variables: Variables to monitor during operation
            enable_stack_trace: Whether to capture stack traces
        """
        thread_id = threading.get_ident()
        start_time = datetime.now()
        
        # Setup context
        context_info = {
            'operation': operation,
            'component': component,
            'start_time': start_time,
            'watch_variables': watch_variables or {},
            'enable_stack_trace': enable_stack_trace
        }
        
        with self._lock:
            self.active_contexts[thread_id] = context_info
        
        # Record start event
        self._record_event('operation_start', component, operation, {
            'start_time': start_time.isoformat(),
            'watched_vars': list((watch_variables or {}).keys())
        }, enable_stack_trace)
        
        self.logger.debug(f"Started debugging {operation} in {component}")
        
        try:
            yield self
            
            # Success case
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self._record_event('operation_success', component, operation, {
                'end_time': end_time.isoformat(),
                'duration': duration,
                'final_variables': self._get_variable_snapshot(watch_variables)
            }, enable_stack_trace)
            
            self.logger.debug(f"Successfully completed debugging {operation} in {duration:.3f}s")
            
        except Exception as e:
            # Failure case
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self._record_event('operation_error', component, operation, {
                'end_time': end_time.isoformat(),
                'duration': duration,
                'error': str(e),
                'error_type': type(e).__name__,
                'final_variables': self._get_variable_snapshot(watch_variables)
            }, True)  # Always capture stack trace on errors
            
            self.logger.error(f"Error in debugging {operation} after {duration:.3f}s: {e}")
            raise
            
        finally:
            with self._lock:
                self.active_contexts.pop(thread_id, None)
    
    def _record_event(self, event_type: str, component: str, operation: str, 
                     details: Dict[str, Any], capture_stack: bool = False):
        """Record a debug event."""
        stack_trace = None
        if capture_stack:
            stack_trace = ''.join(traceback.format_stack())
        
        event = DebugEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            component=component,
            operation=operation,
            details=details,
            stack_trace=stack_trace
        )
        
        with self._lock:
            self.events.append(event)
    
    def _get_variable_snapshot(self, variables: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get a snapshot of current variable values."""
        if not variables:
            return {}
        
        snapshot = {}
        for name, var in variables.items():
            try:
                # Handle different types of variables
                if hasattr(var, '__dict__'):
                    snapshot[name] = str(var)  # Complex objects
                elif isinstance(var, (list, tuple, set)):
                    snapshot[name] = f"{type(var).__name__}(len={len(var)})"
                elif isinstance(var, dict):
                    snapshot[name] = f"dict(keys={len(var)})"
                else:
                    snapshot[name] = str(var)
            except Exception as e:
                snapshot[name] = f"<error getting value: {e}>"
        
        return snapshot
    
    def log_checkpoint(self, checkpoint_name: str, variables: Optional[Dict[str, Any]] = None,
                      component: str = "Platform"):
        """
        Log a debug checkpoint with current state.
        
        Args:
            checkpoint_name: Name of the checkpoint
            variables: Variables to log at this checkpoint
            component: Component creating the checkpoint
        """
        thread_id = threading.get_ident()
        current_context = self.active_contexts.get(thread_id, {})
        
        details = {
            'checkpoint_name': checkpoint_name,
            'variables': self._get_variable_snapshot(variables),
            'active_operation': current_context.get('operation', 'none')
        }
        
        self._record_event('checkpoint', component, checkpoint_name, details)
        self.logger.debug(f"Debug checkpoint: {checkpoint_name}")
    
    def set_breakpoint(self, breakpoint_id: str):
        """Set a debug breakpoint."""
        with self._lock:
            self.breakpoints.add(breakpoint_id)
        self.logger.info(f"Set breakpoint: {breakpoint_id}")
    
    def remove_breakpoint(self, breakpoint_id: str):
        """Remove a debug breakpoint."""
        with self._lock:
            self.breakpoints.discard(breakpoint_id)
        self.logger.info(f"Removed breakpoint: {breakpoint_id}")
    
    def check_breakpoint(self, breakpoint_id: str) -> bool:
        """Check if a breakpoint is set."""
        with self._lock:
            return breakpoint_id in self.breakpoints
    
    def get_events(self, event_type: Optional[str] = None, 
                  component: Optional[str] = None,
                  operation: Optional[str] = None) -> List[DebugEvent]:
        """
        Get debug events with optional filtering.
        
        Args:
            event_type: Filter by event type
            component: Filter by component
            operation: Filter by operation
            
        Returns:
            List of matching debug events
        """
        with self._lock:
            events = list(self.events)
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if component:
            events = [e for e in events if e.component == component]
        if operation:
            events = [e for e in events if e.operation == operation]
        
        return events
    
    def get_active_contexts(self) -> Dict[int, Dict[str, Any]]:
        """Get information about currently active debug contexts."""
        with self._lock:
            return dict(self.active_contexts)
    
    def export_debug_data(self) -> Dict[str, Any]:
        """Export all debug data for analysis."""
        with self._lock:
            return {
                'events': [e.to_dict() for e in self.events],
                'active_contexts': dict(self.active_contexts),
                'breakpoints': list(self.breakpoints),
                'export_timestamp': datetime.now().isoformat()
            }

class CircuitDebugger:
    """
    Specialized debugger for quantum circuits.
    
    Provides detailed analysis and debugging capabilities for quantum circuit
    operations, including gate tracking, qubit state monitoring, and operation validation.
    """
    
    def __init__(self, debug_context: Optional[DebugContext] = None):
        """
        Initialize circuit debugger.
        
        Args:
            debug_context: Debug context to use, or None to create new one
        """
        self.debug_context = debug_context or DebugContext()
        self.logger = get_logger("CircuitDebugger")
        self.gate_history: List[Dict[str, Any]] = []
        self.qubit_states: Dict[str, Any] = {}
        self.operation_count = defaultdict(int)
    
    @contextmanager
    def debug_circuit(self, circuit_name: str, enable_gate_tracking: bool = True):
        """
        Context manager for debugging quantum circuit operations.
        
        Args:
            circuit_name: Name of the circuit being debugged
            enable_gate_tracking: Whether to track individual gate operations
        """
        with self.debug_context.operation_context(
            f"circuit_{circuit_name}", 
            "CircuitDebugger",
            enable_stack_trace=True
        ):
            self.logger.info(f"Starting circuit debug: {circuit_name}")
            
            if enable_gate_tracking:
                self.gate_history.clear()
                self.operation_count.clear()
            
            try:
                yield self
                
                # Log circuit completion statistics
                self.logger.info(f"Circuit {circuit_name} completed successfully")
                self._log_circuit_statistics()
                
            except Exception as e:
                self.logger.error(f"Circuit {circuit_name} failed: {e}")
                self._log_circuit_statistics()
                raise
    
    def log_gate_operation(self, gate_name: str, qubits: List[str], 
                          parameters: Optional[Dict[str, Any]] = None):
        """
        Log a gate operation for debugging.
        
        Args:
            gate_name: Name of the gate
            qubits: List of qubit identifiers
            parameters: Gate parameters if any
        """
        operation_info = {
            'timestamp': datetime.now().isoformat(),
            'gate_name': gate_name,
            'qubits': qubits,
            'parameters': parameters or {},
            'operation_index': len(self.gate_history)
        }
        
        self.gate_history.append(operation_info)
        self.operation_count[gate_name] += 1
        
        self.debug_context.log_checkpoint(
            f"gate_{gate_name}_{len(self.gate_history)}",
            {'gate_info': operation_info},
            "CircuitDebugger"
        )
        
        self.logger.debug(f"Applied {gate_name} to qubits {qubits}")
    
    def update_qubit_state(self, qubit_id: str, state_info: Dict[str, Any]):
        """
        Update qubit state information for debugging.
        
        Args:
            qubit_id: Qubit identifier
            state_info: State information dictionary
        """
        self.qubit_states[qubit_id] = {
            'timestamp': datetime.now().isoformat(),
            **state_info
        }
        
        self.logger.debug(f"Updated state for qubit {qubit_id}")
    
    def _log_circuit_statistics(self):
        """Log circuit operation statistics."""
        total_operations = sum(self.operation_count.values())
        
        stats = {
            'total_operations': total_operations,
            'operation_breakdown': dict(self.operation_count),
            'unique_gates': len(self.operation_count),
            'qubits_involved': len(self.qubit_states)
        }
        
        self.logger.info(f"Circuit statistics: {stats}")
        
        self.debug_context.log_checkpoint(
            "circuit_completion_stats",
            stats,
            "CircuitDebugger"
        )
    
    def get_gate_history(self) -> List[Dict[str, Any]]:
        """Get the complete gate operation history."""
        return list(self.gate_history)
    
    def get_operation_statistics(self) -> Dict[str, int]:
        """Get operation count statistics."""
        return dict(self.operation_count)

class SimulationDebugger:
    """
    Specialized debugger for quantum simulations.
    
    Provides detailed monitoring and debugging of simulation execution,
    including state evolution tracking and measurement analysis.
    """
    
    def __init__(self, debug_context: Optional[DebugContext] = None):
        """
        Initialize simulation debugger.
        
        Args:
            debug_context: Debug context to use, or None to create new one
        """
        self.debug_context = debug_context or DebugContext()
        self.logger = get_logger("SimulationDebugger")
        self.state_evolution: List[Dict[str, Any]] = []
        self.measurement_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
    
    @contextmanager
    def debug_simulation(self, simulation_name: str, enable_state_tracking: bool = True):
        """
        Context manager for debugging quantum simulations.
        
        Args:
            simulation_name: Name of the simulation
            enable_state_tracking: Whether to track state evolution
        """
        with self.debug_context.operation_context(
            f"simulation_{simulation_name}",
            "SimulationDebugger",
            enable_stack_trace=True
        ):
            self.logger.info(f"Starting simulation debug: {simulation_name}")
            
            if enable_state_tracking:
                self.state_evolution.clear()
                self.measurement_history.clear()
                self.performance_metrics.clear()
            
            try:
                yield self
                
                self.logger.info(f"Simulation {simulation_name} completed successfully")
                self._log_simulation_statistics()
                
            except Exception as e:
                self.logger.error(f"Simulation {simulation_name} failed: {e}")
                self._log_simulation_statistics()
                raise
    
    def log_state_evolution(self, step: int, state_info: Dict[str, Any]):
        """
        Log state evolution step.
        
        Args:
            step: Simulation step number
            state_info: State information at this step
        """
        evolution_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'state_info': state_info
        }
        
        self.state_evolution.append(evolution_entry)
        
        self.debug_context.log_checkpoint(
            f"state_evolution_step_{step}",
            evolution_entry,
            "SimulationDebugger"
        )
        
        self.logger.debug(f"Logged state evolution step {step}")
    
    def log_measurement(self, qubits: List[str], results: Dict[str, Any]):
        """
        Log measurement operation and results.
        
        Args:
            qubits: List of measured qubits
            results: Measurement results
        """
        measurement_entry = {
            'timestamp': datetime.now().isoformat(),
            'qubits': qubits,
            'results': results,
            'measurement_index': len(self.measurement_history)
        }
        
        self.measurement_history.append(measurement_entry)
        
        self.debug_context.log_checkpoint(
            f"measurement_{len(self.measurement_history)}",
            measurement_entry,
            "SimulationDebugger"
        )
        
        self.logger.debug(f"Logged measurement of qubits {qubits}")
    
    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """
        Update simulation performance metrics.
        
        Args:
            metrics: Performance metrics dictionary
        """
        self.performance_metrics.update({
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
        
        self.logger.debug("Updated simulation performance metrics")
    
    def _log_simulation_statistics(self):
        """Log simulation statistics."""
        stats = {
            'total_evolution_steps': len(self.state_evolution),
            'total_measurements': len(self.measurement_history),
            'performance_metrics': self.performance_metrics
        }
        
        self.logger.info(f"Simulation statistics: {stats}")
        
        self.debug_context.log_checkpoint(
            "simulation_completion_stats",
            stats,
            "SimulationDebugger"
        )
    
    def get_state_evolution(self) -> List[Dict[str, Any]]:
        """Get the complete state evolution history."""
        return list(self.state_evolution)
    
    def get_measurement_history(self) -> List[Dict[str, Any]]:
        """Get the complete measurement history."""
        return list(self.measurement_history)

# Global debug context
_global_debug_context: Optional[DebugContext] = None
_debug_lock = threading.Lock()

def get_debug_context() -> DebugContext:
    """Get the global debug context instance."""
    global _global_debug_context
    
    with _debug_lock:
        if _global_debug_context is None:
            _global_debug_context = DebugContext()
    
    return _global_debug_context

def get_debugger() -> DebugContext:
    """Get the global debugger instance (alias for get_debug_context)."""
    return get_debug_context()

def debug_operation(operation: str, component: str = "Platform", 
                   watch_variables: Optional[Dict[str, Any]] = None):
    """
    Decorator for debugging function operations.
    
    Args:
        operation: Name of the operation
        component: Component performing the operation
        watch_variables: Variables to monitor
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            debug_ctx = get_debug_context()
            with debug_ctx.operation_context(operation, component, watch_variables):
                return func(*args, **kwargs)
        return wrapper
    return decorator 