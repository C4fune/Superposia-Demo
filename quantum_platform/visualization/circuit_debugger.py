"""Quantum Circuit Step-by-Step Debugger"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np

from quantum_platform.compiler.ir.circuit import QuantumCircuit
from quantum_platform.compiler.ir.operation import Operation
from quantum_platform.simulation.statevector import StateVectorSimulator
from quantum_platform.visualization.state_visualizer import StateVisualizer
from quantum_platform.observability import get_logger


class StepMode(Enum):
    """Step-by-step execution modes."""
    STEP_INTO = "step_into"
    RUN_TO_END = "run_to_end"
    RUN_TO_BREAKPOINT = "run_to_breakpoint"
    PAUSE = "pause"
    ABORT = "abort"


class DebuggerState(Enum):
    """Current state of the debugger."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    ABORTED = "aborted"


@dataclass
class DebugEvent:
    """Event that occurs during debugging."""
    event_type: str
    timestamp: float
    operation_index: int
    operation: Operation
    state_before: Optional[Any] = None
    state_after: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Breakpoint:
    """Represents a breakpoint in the circuit."""
    operation_index: int
    condition: Optional[str] = None
    enabled: bool = True
    hit_count: int = 0
    description: Optional[str] = None


class BreakpointManager:
    """Manages breakpoints for circuit debugging."""
    
    def __init__(self):
        self.breakpoints: Dict[int, Breakpoint] = {}
        self._logger = get_logger(__name__)
    
    def add_breakpoint(self, operation_index: int, condition: Optional[str] = None, description: Optional[str] = None) -> str:
        """Add a breakpoint at the specified operation index."""
        breakpoint = Breakpoint(
            operation_index=operation_index, 
            condition=condition,
            description=description
        )
        self.breakpoints[operation_index] = breakpoint
        self._logger.info(f"Added breakpoint at operation {operation_index}")
        return f"bp_{operation_index}"
    
    def remove_breakpoint(self, operation_index: int) -> bool:
        """Remove breakpoint at the specified operation index."""
        if operation_index in self.breakpoints:
            del self.breakpoints[operation_index]
            self._logger.info(f"Removed breakpoint at operation {operation_index}")
            return True
        return False
    
    def should_break_at(self, operation_index: int, debug_context: Dict[str, Any]) -> bool:
        """Check if execution should break at the given operation index."""
        if operation_index in self.breakpoints:
            bp = self.breakpoints[operation_index]
            if bp.enabled:
                bp.hit_count += 1
                self._logger.debug(f"Hit breakpoint at operation {operation_index} (count: {bp.hit_count})")
                return True
        return False
    
    def get_breakpoints(self) -> Dict[int, Breakpoint]:
        """Get all current breakpoints."""
        return self.breakpoints.copy()
    
    def clear_all_breakpoints(self) -> None:
        """Clear all breakpoints."""
        self.breakpoints.clear()
        self._logger.info("Cleared all breakpoints")
    
    def toggle_breakpoint(self, operation_index: int) -> bool:
        """Toggle a breakpoint on/off. Returns new enabled state."""
        if operation_index in self.breakpoints:
            bp = self.breakpoints[operation_index]
            bp.enabled = not bp.enabled
            self._logger.info(f"Toggled breakpoint at operation {operation_index} to {'enabled' if bp.enabled else 'disabled'}")
            return bp.enabled
        return False


@dataclass
class DebugSession:
    """Represents a debugging session for a quantum circuit."""
    circuit: QuantumCircuit
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_operation_index: int = 0
    state: DebuggerState = DebuggerState.IDLE
    current_quantum_state: Optional[Any] = None
    execution_trace: List[DebugEvent] = field(default_factory=list)
    breakpoint_manager: BreakpointManager = field(default_factory=BreakpointManager)
    classical_memory: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def is_at_end(self) -> bool:
        """Check if we're at the end of the circuit."""
        return self.current_operation_index >= len(self.circuit.operations)
    
    def get_progress(self) -> float:
        """Get execution progress as a percentage."""
        if not self.circuit.operations:
            return 100.0
        return (self.current_operation_index / len(self.circuit.operations)) * 100.0


class QuantumDebugger:
    """Main quantum circuit debugger."""
    
    def __init__(self, 
                 simulator: Optional[StateVectorSimulator] = None,
                 visualizer: Optional[StateVisualizer] = None):
        self.simulator = simulator or StateVectorSimulator()
        self.visualizer = visualizer or StateVisualizer()
        self._logger = get_logger(__name__)
        
        # Active debugging sessions
        self.active_sessions: Dict[str, DebugSession] = {}
        
        # Debug execution control
        self._execution_lock = threading.Lock()
        self._execution_control: Dict[str, Any] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'operation_start': [],
            'operation_complete': [],
            'breakpoint_hit': [],
            'state_changed': [],
            'session_complete': [],
            'error_occurred': []
        }
    
    def start_debug_session(self, circuit: QuantumCircuit, 
                          initial_state: Optional[Any] = None) -> str:
        """Start a new debugging session for a circuit."""
        session = DebugSession(circuit=circuit)
        session.start_time = time.time()
        session.state = DebuggerState.IDLE
        
        # Initialize simulator state
        self.simulator._initialize_state(circuit.num_qubits, initial_state)
        session.current_quantum_state = self.simulator.get_current_state().copy()
        
        # Store session
        self.active_sessions[session.session_id] = session
        
        self._logger.info(f"Started debug session {session.session_id}")
        return session.session_id
    
    def step_next(self, session_id: str) -> Dict[str, Any]:
        """Execute the next operation in the debugging session."""
        return self._execute_step(session_id, StepMode.STEP_INTO)
    
    def step_over(self, session_id: str) -> Dict[str, Any]:
        """Step over the current operation (same as step_next for now)."""
        return self._execute_step(session_id, StepMode.STEP_INTO)
    
    def run_to_end(self, session_id: str) -> Dict[str, Any]:
        """Run the circuit to completion."""
        return self._execute_step(session_id, StepMode.RUN_TO_END)
    
    def run_to_breakpoint(self, session_id: str) -> Dict[str, Any]:
        """Run until hitting a breakpoint."""
        return self._execute_step(session_id, StepMode.RUN_TO_BREAKPOINT)
    
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get the current state of a debugging session."""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        
        state = {
            'session_id': session_id,
            'current_operation_index': session.current_operation_index,
            'state': session.state.value,
            'progress': session.get_progress(),
            'circuit_name': session.circuit.name,
            'total_operations': len(session.circuit.operations),
            'breakpoints': [bp.operation_index for bp in session.breakpoint_manager.get_breakpoints().values() if bp.enabled],
            'is_at_end': session.is_at_end()
        }
        
        # Add current operation info if available
        if not session.is_at_end():
            current_op = session.circuit.operations[session.current_operation_index]
            state['current_operation'] = str(current_op)
        
        # Add quantum state info if available
        if session.current_quantum_state is not None:
            state['quantum_state_norm'] = float(np.linalg.norm(session.current_quantum_state))
            state['num_qubits'] = int(np.log2(len(session.current_quantum_state)))
        
        return state
    
    def end_session(self, session_id: str) -> bool:
        """End a debugging session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.end_time = time.time()
            del self.active_sessions[session_id]
            self._logger.info(f"Ended debug session {session_id}")
            return True
        return False
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_sessions.keys())
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add an event handler for debugging events."""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable) -> None:
        """Remove an event handler."""
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
    
    def _fire_event(self, event_type: str, **kwargs) -> None:
        """Fire a debugging event to all registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(**kwargs)
                except Exception as e:
                    self._logger.error(f"Event handler error: {e}")
    
    def restart_session(self, session_id: str) -> bool:
        """
        Restart a debugging session from the beginning.
        
        Args:
            session_id: ID of session to restart
            
        Returns:
            True if session was restarted successfully
        """
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Reset session state
        session.current_operation_index = 0
        session.state = DebuggerState.IDLE
        session.execution_trace.clear()
        
        # Reinitialize simulator state
        self.simulator._initialize_state(session.circuit.num_qubits)
        current_state = self.simulator.get_current_state()
        if current_state is not None:
            session.current_quantum_state = current_state.copy()
        
        self._logger.info(f"Restarted debug session {session_id}")
        return True
    
    def inspect_state_at_operation(self, session_id: str, operation_index: int) -> Dict[str, Any]:
        """
        Inspect the quantum state at a specific operation index.
        
        Args:
            session_id: ID of the debug session
            operation_index: Index of operation to inspect
            
        Returns:
            Dictionary containing state information at that point
        """
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        
        if operation_index < 0 or operation_index >= len(session.circuit.operations):
            return {'error': f'Operation index {operation_index} out of range'}
        
        # Save current session state
        saved_index = session.current_operation_index
        saved_state = session.state
        saved_quantum_state = session.current_quantum_state.copy() if session.current_quantum_state is not None else None
        
        try:
            # Execute up to the specified operation
            self.simulator._initialize_state(session.circuit.num_qubits)
            
            for i in range(operation_index):
                if i < len(session.circuit.operations):
                    operation = session.circuit.operations[i]
                    self.simulator._execute_single_operation(operation)
            
            # Get state at this point
            current_state = self.simulator.get_current_state()
            state_info = {
                'operation_index': operation_index,
                'quantum_state': current_state.copy() if current_state is not None else None,
                'state_norm': float(np.linalg.norm(current_state)) if current_state is not None else 0.0,
                'num_qubits': int(np.log2(len(current_state))) if current_state is not None else 0
            }
            
            # Add operation info
            if operation_index < len(session.circuit.operations):
                operation = session.circuit.operations[operation_index]
                state_info['next_operation'] = str(operation)
            
            return state_info
            
        except Exception as e:
            return {'error': f'Failed to inspect state: {str(e)}'}
            
        finally:
            # Restore session state
            session.current_operation_index = saved_index
            session.state = saved_state
            if saved_quantum_state is not None:
                session.current_quantum_state = saved_quantum_state
    
    def _execute_step(self, session_id: str, mode: StepMode) -> Dict[str, Any]:
        """Execute debugging step with specified mode."""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        
        if session.is_at_end():
            session.state = DebuggerState.COMPLETED
            return {'completed': True}
        
        try:
            if mode == StepMode.STEP_INTO:
                # Execute single operation
                operation = session.circuit.operations[session.current_operation_index]
                self.simulator._execute_single_operation(operation)
                
                # Update session state
                current_state = self.simulator.get_current_state()
                if current_state is not None:
                    session.current_quantum_state = current_state.copy()
                session.current_operation_index += 1
                session.state = DebuggerState.PAUSED
                
                return {
                    'success': True,
                    'operation_executed': str(operation),
                    'session_state': self.get_session_state(session_id)
                }
            
            elif mode == StepMode.RUN_TO_END:
                # Execute all remaining operations
                while not session.is_at_end():
                    operation = session.circuit.operations[session.current_operation_index]
                    self.simulator._execute_single_operation(operation)
                    session.current_operation_index += 1
                
                # Update final state
                current_state = self.simulator.get_current_state()
                if current_state is not None:
                    session.current_quantum_state = current_state.copy()
                session.state = DebuggerState.COMPLETED
                
                return {
                    'success': True,
                    'completed': True,
                    'session_state': self.get_session_state(session_id)
                }
            
            elif mode == StepMode.RUN_TO_BREAKPOINT:
                # Execute until breakpoint or end
                while not session.is_at_end():
                    operation_index = session.current_operation_index
                    
                    # Check for breakpoint
                    if session.breakpoint_manager.should_break_at(operation_index, {}):
                        session.state = DebuggerState.PAUSED
                        return {
                            'success': True,
                            'breakpoint_hit': True,
                            'operation_index': operation_index,
                            'session_state': self.get_session_state(session_id)
                        }
                    
                    # Execute operation
                    operation = session.circuit.operations[operation_index]
                    self.simulator._execute_single_operation(operation)
                    session.current_operation_index += 1
                
                # Reached end without breakpoint
                current_state = self.simulator.get_current_state()
                if current_state is not None:
                    session.current_quantum_state = current_state.copy()
                session.state = DebuggerState.COMPLETED
                
                return {
                    'success': True,
                    'completed': True,
                    'session_state': self.get_session_state(session_id)
                }
            
            else:
                return {'error': f'Unsupported step mode: {mode}'}
                
        except Exception as e:
            session.state = DebuggerState.ERROR
            self._logger.error(f"Debug step failed: {e}")
            return {'error': str(e)}
