"""
Quantum Operations for the IR.

This module defines all types of operations that can appear in a quantum circuit,
including gates, measurements, and classical control flow constructs.
"""

from typing import List, Optional, Dict, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import uuid

from quantum_platform.compiler.ir.qubit import Qubit
from quantum_platform.compiler.ir.types import ParameterValue, Parameter, ParameterDict


class OperationType(Enum):
    """Types of operations in the IR."""
    GATE = "gate"                    # Quantum gate operation
    MEASUREMENT = "measurement"      # Measurement operation
    RESET = "reset"                 # Reset qubit to |0>
    BARRIER = "barrier"             # Synchronization barrier
    IF = "if"                       # Conditional execution
    LOOP = "loop"                   # Loop construct
    CLASSICAL = "classical"         # Classical computation


@dataclass
class Operation(ABC):
    """
    Abstract base class for all operations in the quantum IR.
    
    All operations must have a name and list of target qubits.
    Additional properties depend on the specific operation type.
    """
    name: str
    targets: List[Qubit]
    controls: List[Qubit] = field(default_factory=list)
    params: ParameterDict = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Internal tracking
    _uuid: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)
    _timestamp: Optional[int] = field(default=None, init=False)
    
    def __post_init__(self):
        """Validate operation parameters."""
        if not self.name:
            raise ValueError("Operation name cannot be empty")
        
        if not isinstance(self.targets, list):
            raise TypeError("Targets must be a list")
        
        if not isinstance(self.controls, list):
            raise TypeError("Controls must be a list")
        
        # Ensure no overlap between targets and controls
        target_ids = {q.id for q in self.targets}
        control_ids = {q.id for q in self.controls}
        if target_ids & control_ids:
            raise ValueError(f"Qubits cannot be both target and control: {target_ids & control_ids}")
        
        # Convert parameters to ParameterValue objects
        self._normalize_parameters()
    
    def _normalize_parameters(self):
        """Convert all parameters to ParameterValue objects."""
        for key, value in self.params.items():
            if not isinstance(value, ParameterValue):
                self.params[key] = ParameterValue(value)
    
    @property
    def operation_type(self) -> OperationType:
        """Get the type of this operation."""
        return OperationType.GATE  # Default for base class
    
    @property
    def uuid(self) -> str:
        """Get unique identifier for this operation."""
        return self._uuid
    
    @property
    def all_qubits(self) -> List[Qubit]:
        """Get all qubits involved in this operation (targets + controls)."""
        return self.targets + self.controls
    
    @property
    def num_qubits(self) -> int:
        """Get total number of qubits involved."""
        return len(self.all_qubits)
    
    @property
    def is_parameterized(self) -> bool:
        """Check if operation has symbolic parameters."""
        return any(param.is_symbolic for param in self.params.values())
    
    def get_parameter(self, name: str) -> Optional[ParameterValue]:
        """Get a parameter by name."""
        return self.params.get(name)
    
    def substitute_parameters(self, substitutions: Dict[str, Parameter]) -> 'Operation':
        """
        Create a new operation with parameters substituted.
        
        Args:
            substitutions: Dictionary mapping parameter names to values
            
        Returns:
            New operation with substituted parameters
        """
        # Convert substitutions to ParameterValue objects
        param_subs = {}
        for name, value in substitutions.items():
            if isinstance(value, ParameterValue):
                param_subs[name] = value
            else:
                param_subs[name] = ParameterValue(value)
        
        # Create new parameters dict
        new_params = {}
        for name, param in self.params.items():
            if name in param_subs:
                new_params[name] = param_subs[name]
            else:
                new_params[name] = param
        
        # Create new operation (this will need to be overridden in subclasses for proper typing)
        return self._copy_with_params(new_params)
    
    @abstractmethod
    def _copy_with_params(self, new_params: ParameterDict) -> 'Operation':
        """Create a copy of this operation with new parameters."""
        pass
    
    def commutes_with(self, other: 'Operation') -> bool:
        """
        Check if this operation commutes with another.
        
        Two operations commute if they can be reordered without affecting
        the circuit's semantics. This is used for optimization.
        """
        # Simple check: operations on disjoint sets of qubits commute
        self_qubits = set(q.id for q in self.all_qubits)
        other_qubits = set(q.id for q in other.all_qubits)
        
        # If they share no qubits, they commute
        if not (self_qubits & other_qubits):
            return True
        
        # More sophisticated commutation rules would go here
        # For now, assume operations on same qubits don't commute
        return False
    
    def validate(self) -> bool:
        """
        Validate that this operation is well-formed.
        
        Returns:
            True if valid, raises exception otherwise
        """
        # Check that all qubits are properly formed
        for qubit in self.all_qubits:
            if not isinstance(qubit, Qubit):
                raise TypeError(f"Expected Qubit, got {type(qubit)}")
        
        return True
    
    def __hash__(self) -> int:
        """Hash based on UUID."""
        return hash(self._uuid)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on UUID."""
        if not isinstance(other, Operation):
            return NotImplemented
        return self._uuid == other._uuid
    
    def __str__(self) -> str:
        """String representation."""
        parts = [self.name]
        
        if self.params:
            param_strs = [f"{k}={v}" for k, v in self.params.items()]
            parts.append(f"({', '.join(param_strs)})")
        
        if self.controls:
            control_str = ",".join(str(q) for q in self.controls)
            parts.append(f"ctrl:[{control_str}]")
        
        target_str = ",".join(str(q) for q in self.targets)
        parts.append(f"[{target_str}]")
        
        return " ".join(parts)
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"{self.__class__.__name__}({self})"


@dataclass
class GateOperation(Operation):
    """
    Represents a quantum gate operation.
    
    This is the most common type of operation, representing unitary
    transformations applied to qubits.
    """
    
    @property
    def operation_type(self) -> OperationType:
        return OperationType.GATE
    
    def _copy_with_params(self, new_params: ParameterDict) -> 'GateOperation':
        """Create a copy with new parameters."""
        return GateOperation(
            name=self.name,
            targets=self.targets.copy(),
            controls=self.controls.copy(),
            params=new_params,
            metadata=self.metadata.copy()
        )


@dataclass
class MeasurementOperation(Operation):
    """
    Represents a measurement operation.
    
    Measurements collapse the quantum state and produce classical results.
    """
    classical_target: Optional[str] = None  # Name of classical register/bit
    
    @property
    def operation_type(self) -> OperationType:
        return OperationType.MEASUREMENT
    
    def _copy_with_params(self, new_params: ParameterDict) -> 'MeasurementOperation':
        """Create a copy with new parameters."""
        return MeasurementOperation(
            name=self.name,
            targets=self.targets.copy(),
            controls=self.controls.copy(),
            params=new_params,
            metadata=self.metadata.copy(),
            classical_target=self.classical_target
        )


@dataclass
class ResetOperation(Operation):
    """
    Represents a reset operation that sets qubits to |0> state.
    """
    
    @property
    def operation_type(self) -> OperationType:
        return OperationType.RESET
    
    def _copy_with_params(self, new_params: ParameterDict) -> 'ResetOperation':
        """Create a copy with new parameters."""
        return ResetOperation(
            name=self.name,
            targets=self.targets.copy(),
            controls=self.controls.copy(),
            params=new_params,
            metadata=self.metadata.copy()
        )


@dataclass
class BarrierOperation(Operation):
    """
    Represents a barrier operation for synchronization.
    
    Barriers prevent reordering of operations across them during optimization.
    """
    
    @property
    def operation_type(self) -> OperationType:
        return OperationType.BARRIER
    
    def _copy_with_params(self, new_params: ParameterDict) -> 'BarrierOperation':
        """Create a copy with new parameters."""
        return BarrierOperation(
            name=self.name,
            targets=self.targets.copy(),
            controls=self.controls.copy(),
            params=new_params,
            metadata=self.metadata.copy()
        )


@dataclass
class IfOperation(Operation):
    """
    Represents conditional execution based on classical register values.
    
    This enables classical control flow in quantum circuits.
    """
    condition: str = ""  # Classical register or condition expression
    then_operations: List[Operation] = field(default_factory=list)
    else_operations: List[Operation] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate operation parameters."""
        super().__post_init__()
        if not self.condition:
            raise ValueError("IfOperation requires a condition")
    
    @property
    def operation_type(self) -> OperationType:
        return OperationType.IF
    
    def _copy_with_params(self, new_params: ParameterDict) -> 'IfOperation':
        """Create a copy with new parameters."""
        return IfOperation(
            name=self.name,
            targets=self.targets.copy(),
            controls=self.controls.copy(),
            params=new_params,
            metadata=self.metadata.copy(),
            condition=self.condition,
            then_operations=self.then_operations.copy(),
            else_operations=self.else_operations.copy()
        )


@dataclass  
class LoopOperation(Operation):
    """
    Represents loop constructs in quantum circuits.
    
    Supports both fixed-count loops and while-style conditional loops.
    """
    loop_body: List[Operation] = field(default_factory=list)
    loop_count: Optional[int] = None  # For fixed-count loops
    loop_condition: Optional[str] = None  # For conditional loops
    
    @property
    def operation_type(self) -> OperationType:
        return OperationType.LOOP
    
    def _copy_with_params(self, new_params: ParameterDict) -> 'LoopOperation':
        """Create a copy with new parameters."""
        return LoopOperation(
            name=self.name,
            targets=self.targets.copy(),
            controls=self.controls.copy(),
            params=new_params,
            metadata=self.metadata.copy(),
            loop_body=self.loop_body.copy(),
            loop_count=self.loop_count,
            loop_condition=self.loop_condition
        )


@dataclass
class ClassicalOperation(Operation):
    """
    Represents classical computations on classical registers.
    
    This allows embedding classical logic within quantum circuits.
    """
    computation: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    input_registers: List[str] = field(default_factory=list)
    output_registers: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate operation parameters."""
        super().__post_init__()
        if self.computation is None:
            raise ValueError("ClassicalOperation requires a computation function")
    
    @property
    def operation_type(self) -> OperationType:
        return OperationType.CLASSICAL
    
    def _copy_with_params(self, new_params: ParameterDict) -> 'ClassicalOperation':
        """Create a copy with new parameters."""
        return ClassicalOperation(
            name=self.name,
            targets=self.targets.copy(),
            controls=self.controls.copy(),
            params=new_params,
            metadata=self.metadata.copy(),
            computation=self.computation,
            input_registers=self.input_registers.copy(),
            output_registers=self.output_registers.copy()
        ) 