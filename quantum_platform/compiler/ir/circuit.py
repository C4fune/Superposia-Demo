"""
Quantum Circuit representation for the IR.

This module defines the QuantumCircuit class which serves as the main container
for quantum programs, holding qubits, operations, and classical registers.
"""

from typing import List, Dict, Optional, Set, Any, Union, Iterator, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import uuid
import json

from quantum_platform.compiler.ir.qubit import Qubit, QubitRegister, QubitState
from quantum_platform.compiler.ir.operation import (
    Operation, GateOperation, MeasurementOperation, IfOperation, LoopOperation,
    OperationType
)
from quantum_platform.compiler.ir.types import ParameterValue, Parameter, ParameterDict


@dataclass
class ClassicalRegister:
    """
    Represents a classical register for storing measurement results.
    
    Classical registers hold the outcomes of quantum measurements and
    intermediate classical computations.
    """
    name: str
    size: int
    values: List[Optional[int]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize the register values."""
        if len(self.values) != self.size:
            self.values = [None] * self.size
    
    def __getitem__(self, index: int) -> Optional[int]:
        """Get value at index."""
        if not 0 <= index < self.size:
            raise IndexError(f"Index {index} out of range for register of size {self.size}")
        return self.values[index]
    
    def __setitem__(self, index: int, value: int) -> None:
        """Set value at index."""
        if not 0 <= index < self.size:
            raise IndexError(f"Index {index} out of range for register of size {self.size}")
        self.values[index] = value
    
    def __len__(self) -> int:
        """Get size of register."""
        return self.size
    
    def __str__(self) -> str:
        return f"{self.name}[{self.size}]"
    
    def __repr__(self) -> str:
        return f"ClassicalRegister(name={self.name!r}, size={self.size}, values={self.values})"


class QuantumCircuit:
    """
    Main container for quantum programs in the IR.
    
    A QuantumCircuit holds all the information needed to represent a quantum
    program: qubits, operations, classical registers, and metadata.
    """
    
    def __init__(self, name: Optional[str] = None, num_qubits: int = 0):
        """
        Initialize a quantum circuit.
        
        Args:
            name: Optional name for the circuit
            num_qubits: Initial number of qubits to allocate (0 means no pre-allocation)
        """
        self.name = name or f"circuit_{uuid.uuid4().hex[:8]}"
        self.operations: List[Operation] = []
        self.qubits: List[Qubit] = []
        self.classical_registers: Dict[str, ClassicalRegister] = {}
        self.metadata: Dict[str, Any] = {}
        
        # Internal tracking
        self._qubit_counter = 0
        self._operation_counter = 0
        self._parameter_table: Dict[str, ParameterValue] = {}
        
        # NOTE: Don't pre-allocate qubits unless explicitly requested
        # Most quantum programs allocate qubits explicitly in their code
        # Pre-allocation can cause extra qubits that aren't used
    
    @property
    def num_qubits(self) -> int:
        """Get the number of qubits in the circuit."""
        return len(self.qubits)
    
    @property
    def num_operations(self) -> int:
        """Get the number of operations in the circuit."""
        return len(self.operations)
    
    @property
    def depth(self) -> int:
        """
        Calculate the circuit depth (number of time steps).
        
        This is a simplified calculation that assumes operations on different
        qubits can be performed in parallel.
        """
        if not self.operations:
            return 0
        
        # Track the time step for each qubit
        qubit_times = defaultdict(int)
        
        for op in self.operations:
            # Find the maximum time among all qubits involved
            max_time = 0
            for qubit in op.all_qubits:
                max_time = max(max_time, qubit_times[qubit.id])
            
            # All involved qubits advance to the next time step
            for qubit in op.all_qubits:
                qubit_times[qubit.id] = max_time + 1
        
        return max(qubit_times.values()) if qubit_times else 0
    
    @property
    def is_parameterized(self) -> bool:
        """Check if the circuit contains symbolic parameters."""
        return any(op.is_parameterized for op in self.operations)
    
    def allocate_qubit(self, name: Optional[str] = None) -> Qubit:
        """
        Allocate a single qubit.
        
        Args:
            name: Optional name for the qubit
            
        Returns:
            The allocated qubit
        """
        qubit = Qubit(id=self._qubit_counter, name=name)
        qubit._allocation_order = len(self.qubits)
        self.qubits.append(qubit)
        self._qubit_counter += 1
        return qubit
    
    def allocate_qubits(self, count: int, names: Optional[List[str]] = None) -> List[Qubit]:
        """
        Allocate multiple qubits.
        
        Args:
            count: Number of qubits to allocate
            names: Optional list of names (must match count if provided)
            
        Returns:
            List of allocated qubits
        """
        if names and len(names) != count:
            raise ValueError(f"Names list length {len(names)} doesn't match count {count}")
        
        qubits = []
        for i in range(count):
            name = names[i] if names else None
            qubits.append(self.allocate_qubit(name))
        
        return qubits
    
    def allocate_register(self, name: str, size: int) -> QubitRegister:
        """
        Allocate a quantum register.
        
        Args:
            name: Name of the register
            size: Number of qubits in the register
            
        Returns:
            The allocated register
        """
        start_id = self._qubit_counter
        register = QubitRegister(name, size, start_id)
        
        # Add qubits to the circuit
        for qubit in register.qubits:
            qubit._allocation_order = len(self.qubits)
            self.qubits.append(qubit)
            self._qubit_counter += 1
        
        return register
    
    def add_classical_register(self, name: str, size: int) -> ClassicalRegister:
        """
        Add a classical register for measurement results.
        
        Args:
            name: Name of the register
            size: Size of the register
            
        Returns:
            The created classical register
        """
        if name in self.classical_registers:
            raise ValueError(f"Classical register '{name}' already exists")
        
        register = ClassicalRegister(name, size)
        self.classical_registers[name] = register
        return register
    
    def add_operation(self, operation: Operation) -> None:
        """
        Add an operation to the circuit.
        
        Args:
            operation: The operation to add
        """
        # Validate operation
        operation.validate()
        
        # Check that all qubits are part of this circuit
        circuit_qubit_ids = {q.id for q in self.qubits}
        op_qubit_ids = {q.id for q in operation.all_qubits}
        
        if not op_qubit_ids.issubset(circuit_qubit_ids):
            missing = op_qubit_ids - circuit_qubit_ids
            raise ValueError(f"Operation uses qubits not in circuit: {missing}")
        
        # Mark qubits as in use
        for qubit in operation.all_qubits:
            qubit.mark_in_use()
        
        # Add to operations list
        operation._timestamp = self._operation_counter
        self.operations.append(operation)
        self._operation_counter += 1
    
    def add_gate(self, gate_name: str, targets: List[Qubit], 
                 controls: Optional[List[Qubit]] = None,
                 params: Optional[ParameterDict] = None) -> GateOperation:
        """
        Add a gate operation to the circuit.
        
        Args:
            gate_name: Name of the gate
            targets: Target qubits
            controls: Control qubits (optional)
            params: Gate parameters (optional)
            
        Returns:
            The created gate operation
        """
        gate_op = GateOperation(
            name=gate_name,
            targets=targets,
            controls=controls or [],
            params=params or {}
        )
        self.add_operation(gate_op)
        return gate_op
    
    def add_measurement(self, targets: List[Qubit], 
                       classical_register: Optional[str] = None) -> MeasurementOperation:
        """
        Add a measurement operation to the circuit.
        
        Args:
            targets: Qubits to measure
            classical_register: Name of classical register to store results
            
        Returns:
            The created measurement operation
        """
        measure_op = MeasurementOperation(
            name="measure",
            targets=targets,
            classical_target=classical_register
        )
        self.add_operation(measure_op)
        
        # Mark qubits as measured
        for qubit in targets:
            qubit.mark_measured()
        
        return measure_op
    
    def get_operations_on_qubit(self, qubit: Qubit) -> List[Operation]:
        """Get all operations that act on a specific qubit."""
        return [op for op in self.operations if qubit in op.all_qubits]
    
    def get_operations_by_type(self, op_type: OperationType) -> List[Operation]:
        """Get all operations of a specific type."""
        return [op for op in self.operations if op.operation_type == op_type]
    
    def get_gate_operations(self) -> List[GateOperation]:
        """Get all gate operations."""
        return [op for op in self.operations if isinstance(op, GateOperation)]
    
    def get_measurement_operations(self) -> List[MeasurementOperation]:
        """Get all measurement operations."""
        return [op for op in self.operations if isinstance(op, MeasurementOperation)]
    
    def substitute_parameters(self, substitutions: Dict[str, Parameter]) -> 'QuantumCircuit':
        """
        Create a new circuit with parameters substituted.
        
        Args:
            substitutions: Dictionary mapping parameter names to values
            
        Returns:
            New circuit with substituted parameters
        """
        new_circuit = QuantumCircuit(name=f"{self.name}_substituted")
        
        # Copy qubits (create new instances with same properties)
        qubit_map = {}
        for qubit in self.qubits:
            new_qubit = Qubit(
                id=qubit.id,
                name=qubit.name,
                state=qubit.state,
                register_name=qubit.register_name,
                metadata=qubit.metadata.copy()
            )
            new_circuit.qubits.append(new_qubit)
            qubit_map[qubit.id] = new_qubit
        
        # Copy classical registers
        for name, register in self.classical_registers.items():
            new_circuit.classical_registers[name] = ClassicalRegister(
                name=register.name,
                size=register.size,
                values=register.values.copy()
            )
        
        # Copy operations with parameter substitution
        for operation in self.operations:
            # Map qubits to new circuit's qubits
            new_targets = [qubit_map[q.id] for q in operation.targets]
            new_controls = [qubit_map[q.id] for q in operation.controls]
            
            # Substitute parameters
            new_operation = operation.substitute_parameters(substitutions)
            new_operation.targets = new_targets
            new_operation.controls = new_controls
            
            new_circuit.operations.append(new_operation)
        
        # Copy metadata
        new_circuit.metadata = self.metadata.copy()
        
        return new_circuit
    
    def copy(self) -> 'QuantumCircuit':
        """Create a deep copy of this circuit."""
        return self.substitute_parameters({})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert circuit to dictionary representation.
        
        This is useful for serialization and debugging.
        """
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_operations": self.num_operations,
            "depth": self.depth,
            "qubits": [
                {
                    "id": q.id,
                    "name": q.name,
                    "state": q.state.value,
                    "register_name": q.register_name
                }
                for q in self.qubits
            ],
            "classical_registers": {
                name: {
                    "size": reg.size,
                    "values": reg.values
                }
                for name, reg in self.classical_registers.items()
            },
            "operations": [
                {
                    "type": op.__class__.__name__,
                    "name": op.name,
                    "targets": [q.id for q in op.targets],
                    "controls": [q.id for q in op.controls],
                    "params": {k: str(v) for k, v in op.params.items()},
                    "metadata": op.metadata
                }
                for op in self.operations
            ],
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert circuit to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def __str__(self) -> str:
        """String representation showing basic circuit info."""
        return f"QuantumCircuit('{self.name}', qubits={self.num_qubits}, ops={self.num_operations}, depth={self.depth})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        lines = [f"QuantumCircuit: {self.name}"]
        lines.append(f"  Qubits: {self.num_qubits}")
        lines.append(f"  Operations: {self.num_operations}")
        lines.append(f"  Depth: {self.depth}")
        
        if self.classical_registers:
            lines.append("  Classical Registers:")
            for name, reg in self.classical_registers.items():
                lines.append(f"    {name}: {reg.size} bits")
        
        if self.operations:
            lines.append("  Operations:")
            for i, op in enumerate(self.operations[:5]):  # Show first 5
                lines.append(f"    {i}: {op}")
            if len(self.operations) > 5:
                lines.append(f"    ... and {len(self.operations) - 5} more")
        
        return "\n".join(lines) 