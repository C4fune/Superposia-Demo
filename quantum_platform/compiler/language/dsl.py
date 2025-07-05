"""
High-Level Quantum Programming Language DSL

This module provides the main interface for the Python-based quantum DSL,
including the QuantumProgram context manager and related functionality.
"""

from typing import List, Optional, Dict, Any, Union, Callable, ContextManager
from contextlib import contextmanager
import threading
from dataclasses import dataclass

from quantum_platform.compiler.ir.qubit import Qubit, QubitRegister
from quantum_platform.compiler.ir.circuit import QuantumCircuit, ClassicalRegister
from quantum_platform.compiler.ir.operation import Operation, MeasurementOperation
from quantum_platform.compiler.ir.types import Parameter, ParameterDict
from quantum_platform.compiler.gates.factory import GateFactory


# Thread-local storage for the current quantum context
_context_stack = threading.local()


def _get_context_stack() -> List['QuantumContext']:
    """Get the context stack for the current thread."""
    if not hasattr(_context_stack, 'stack'):
        _context_stack.stack = []
    return _context_stack.stack


def get_current_context() -> Optional['QuantumContext']:
    """Get the current active quantum context."""
    stack = _get_context_stack()
    return stack[-1] if stack else None


class QuantumContext:
    """
    Context for quantum program execution.
    
    This provides the runtime environment for quantum programs, including
    qubit allocation, gate application, and measurement operations.
    """
    
    def __init__(self, circuit: Optional[QuantumCircuit] = None, name: Optional[str] = None):
        """
        Initialize quantum context.
        
        Args:
            circuit: Optional quantum circuit to work with
            name: Optional name for the context
        """
        self.circuit = circuit or QuantumCircuit(name=name)
        self.name = name or "quantum_context"
        self.gate_factory = GateFactory(self.circuit)
        
        # Track allocated resources
        self._allocated_qubits: List[Qubit] = []
        self._classical_registers: Dict[str, ClassicalRegister] = {}
        
        # Context state
        self._is_active = False
    
    def __enter__(self) -> 'QuantumContext':
        """Enter the quantum context."""
        stack = _get_context_stack()
        stack.append(self)
        self._is_active = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the quantum context."""
        stack = _get_context_stack()
        if stack and stack[-1] is self:
            stack.pop()
        self._is_active = False
    
    @property
    def is_active(self) -> bool:
        """Check if this context is currently active."""
        return self._is_active
    
    def allocate_qubit(self, name: Optional[str] = None) -> Qubit:
        """
        Allocate a single qubit.
        
        Args:
            name: Optional name for the qubit
            
        Returns:
            Allocated qubit
        """
        qubit = self.circuit.allocate_qubit(name)
        self._allocated_qubits.append(qubit)
        return qubit
    
    def allocate_qubits(self, count: int, names: Optional[List[str]] = None) -> List[Qubit]:
        """
        Allocate multiple qubits.
        
        Args:
            count: Number of qubits to allocate
            names: Optional names for the qubits
            
        Returns:
            List of allocated qubits
        """
        qubits = self.circuit.allocate_qubits(count, names)
        self._allocated_qubits.extend(qubits)
        return qubits
    
    def allocate(self, count: int = 1, names: Optional[List[str]] = None) -> List[Qubit]:
        """
        Allocate qubits in this context.
        
        Args:
            count: Number of qubits to allocate
            names: Optional names for the qubits
            
        Returns:
            List of allocated qubits (always a list for consistency)
        """
        if count == 1:
            qubit = self.circuit.allocate_qubit(names[0] if names else None)
            self._allocated_qubits.append(qubit)
            return [qubit]
        else:
            qubits = self.circuit.allocate_qubits(count, names)
            self._allocated_qubits.extend(qubits)
            return qubits
    
    def allocate_register(self, name: str, size: int) -> QubitRegister:
        """
        Allocate a quantum register.
        
        Args:
            name: Name of the register
            size: Size of the register
            
        Returns:
            Allocated quantum register
        """
        register = self.circuit.allocate_register(name, size)
        self._allocated_qubits.extend(register.qubits)
        return register
    
    def add_classical_register(self, name: str, size: int) -> ClassicalRegister:
        """
        Add a classical register for measurement results.
        
        Args:
            name: Name of the register
            size: Size of the register
            
        Returns:
            Created classical register
        """
        register = self.circuit.add_classical_register(name, size)
        self._classical_registers[name] = register
        return register
    
    def apply_gate(self, gate_name: str, targets: Union[Qubit, List[Qubit]],
                   controls: Optional[Union[Qubit, List[Qubit]]] = None,
                   **params) -> Operation:
        """
        Apply a gate to qubits.
        
        Args:
            gate_name: Name of the gate
            targets: Target qubit(s)
            controls: Control qubit(s) (optional)
            **params: Gate parameters
            
        Returns:
            The created operation
        """
        # Normalize inputs
        if isinstance(targets, Qubit):
            targets = [targets]
        if isinstance(controls, Qubit):
            controls = [controls]
        
        return self.gate_factory.apply_gate(gate_name, targets, controls, params)
    
    def measure(self, qubits: Union[Qubit, List[Qubit]], 
                classical_register: Optional[str] = None) -> MeasurementOperation:
        """
        Measure qubits and store results.
        
        Args:
            qubits: Qubit(s) to measure
            classical_register: Optional classical register to store results
            
        Returns:
            Measurement operation
        """
        if isinstance(qubits, Qubit):
            qubits = [qubits]
        
        return self.circuit.add_measurement(qubits, classical_register)
    
    def get_measurement_result(self, register_name: str) -> Optional[List[int]]:
        """
        Get measurement results from a classical register.
        
        Args:
            register_name: Name of the classical register
            
        Returns:
            List of measurement results or None if not found
        """
        if register_name in self._classical_registers:
            register = self._classical_registers[register_name]
            return [val for val in register.values if val is not None]
        return None
    
    def reset_qubit(self, qubit: Qubit) -> Operation:
        """Reset a qubit to |0> state."""
        return self.apply_gate("reset", [qubit])
    
    def barrier(self, qubits: Optional[Union[Qubit, List[Qubit]]] = None) -> Operation:
        """Add a barrier operation."""
        if qubits is None:
            qubits = self._allocated_qubits
        elif isinstance(qubits, Qubit):
            qubits = [qubits]
        
        return self.apply_gate("barrier", qubits)
    
    def get_circuit_info(self) -> Dict[str, Any]:
        """Get information about the current circuit."""
        return {
            "name": self.circuit.name,
            "num_qubits": self.circuit.num_qubits,
            "num_operations": self.circuit.num_operations,
            "depth": self.circuit.depth,
            "is_parameterized": self.circuit.is_parameterized
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export context circuit as dictionary."""
        if not self.circuit:
            return {"name": self.name, "circuit": None}
        
        return {
            "name": self.name,
            "circuit": self.circuit.to_dict()
        }
    
    def to_json(self) -> str:
        """Export context circuit as JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)


class QuantumProgram:
    """
    Main entry point for the quantum programming DSL.
    
    This provides a context manager for writing quantum programs with
    automatic resource management and circuit generation.
    """
    
    def __init__(self, name: Optional[str] = None, num_qubits: int = 0):
        """
        Initialize a quantum program.
        
        Args:
            name: Optional name for the program
            num_qubits: Number of qubits to pre-allocate
        """
        self.name = name or "quantum_program"
        self.num_qubits = num_qubits
        self._context: Optional[QuantumContext] = None
        self._circuit: Optional[QuantumCircuit] = None
    
    def __enter__(self) -> QuantumContext:
        """Enter the quantum program context."""
        circuit = QuantumCircuit(name=self.name, num_qubits=self.num_qubits)
        self._context = QuantumContext(circuit, self.name)
        self._circuit = circuit
        return self._context.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the quantum program context."""
        if self._context:
            self._context.__exit__(exc_type, exc_val, exc_tb)
    
    @property
    def circuit(self) -> Optional[QuantumCircuit]:
        """Get the generated circuit."""
        return self._circuit
    
    def to_dict(self) -> Dict[str, Any]:
        """Export program as dictionary."""
        if not self._circuit:
            return {"name": self.name, "circuit": None}
        
        return {
            "name": self.name,
            "circuit": self._circuit.to_dict()
        }
    
    def to_json(self) -> str:
        """Export program as JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Module-level convenience functions
# =============================================================================

def allocate(count: int = 1, names: Optional[List[str]] = None) -> List[Qubit]:
    """
    Allocate qubits in the current context.
    
    Args:
        count: Number of qubits to allocate
        names: Optional names for the qubits
        
    Returns:
        List of allocated qubits
    """
    context = get_current_context()
    if context is None:
        raise RuntimeError("No active quantum context. Use 'with QuantumProgram():' first.")
    return context.allocate(count, names)


def allocate_register(name: str, size: int) -> QubitRegister:
    """Allocate a quantum register in the current context."""
    context = get_current_context()
    if context is None:
        raise RuntimeError("No active quantum context. Use 'with QuantumProgram():' first.")
    return context.allocate_register(name, size)


def add_classical_register(name: str, size: int) -> ClassicalRegister:
    """Add a classical register in the current context."""
    context = get_current_context()
    if context is None:
        raise RuntimeError("No active quantum context. Use 'with QuantumProgram():' first.")
    return context.add_classical_register(name, size)


def measure(qubits: Union[Qubit, List[Qubit]], 
            classical_register: Optional[str] = None) -> MeasurementOperation:
    """Measure qubits in the current context."""
    context = get_current_context()
    if context is None:
        raise RuntimeError("No active quantum context. Use 'with QuantumProgram():' first.")
    return context.measure(qubits, classical_register)


def reset(qubit: Qubit) -> Operation:
    """Reset a qubit in the current context."""
    context = get_current_context()
    if context is None:
        raise RuntimeError("No active quantum context. Use 'with QuantumProgram():' first.")
    return context.reset_qubit(qubit)


def barrier(qubits: Optional[Union[Qubit, List[Qubit]]] = None) -> Operation:
    """Add a barrier in the current context."""
    context = get_current_context()
    if context is None:
        raise RuntimeError("No active quantum context. Use 'with QuantumProgram():' first.")
    return context.barrier(qubits)


def get_current_circuit() -> Optional[QuantumCircuit]:
    """Get the current circuit."""
    context = get_current_context()
    return context.circuit if context else None 