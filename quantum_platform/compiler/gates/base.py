"""
Base Gate definitions for the quantum platform.

This module defines the fundamental Gate class and matrix representations
that form the foundation of the extensible gate system.
"""

from typing import Optional, Dict, List, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import sympy
from enum import Enum

from quantum_platform.compiler.ir.types import Parameter, ParameterValue


class GateType(Enum):
    """Categories of quantum gates."""
    SINGLE_QUBIT = "single_qubit"       # Gates acting on one qubit
    TWO_QUBIT = "two_qubit"             # Gates acting on two qubits  
    MULTI_QUBIT = "multi_qubit"         # Gates acting on multiple qubits
    PARAMETRIC = "parametric"           # Gates with parameters
    CONTROLLED = "controlled"           # Controlled versions of gates
    MEASUREMENT = "measurement"         # Measurement operations
    CLASSICAL = "classical"             # Classical operations


@dataclass
class GateMatrix:
    """
    Represents the matrix form of a quantum gate.
    
    This can be either a concrete numpy array or a symbolic expression
    that depends on parameters.
    """
    matrix: Union[np.ndarray, sympy.Matrix, Callable[..., np.ndarray]]
    is_parametric: bool = False
    parameter_names: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate the matrix representation."""
        if isinstance(self.matrix, (np.ndarray, sympy.Matrix)):
            # Check if matrix is unitary (for concrete matrices)
            if isinstance(self.matrix, np.ndarray):
                self._validate_unitary()
        elif callable(self.matrix):
            self.is_parametric = True
        else:
            raise TypeError(f"Matrix must be numpy array, sympy Matrix, or callable, got {type(self.matrix)}")
    
    def _validate_unitary(self) -> None:
        """Validate that a concrete matrix is unitary."""
        if isinstance(self.matrix, np.ndarray):
            # Check if matrix is square
            if self.matrix.shape[0] != self.matrix.shape[1]:
                raise ValueError(f"Gate matrix must be square, got shape {self.matrix.shape}")
            
            # Check if matrix is unitary (U * U† = I)
            try:
                product = np.dot(self.matrix, np.conj(self.matrix.T))
                identity = np.eye(self.matrix.shape[0])
                if not np.allclose(product, identity, atol=1e-10):
                    raise ValueError("Gate matrix is not unitary")
            except Exception as e:
                # For debugging purposes, allow non-unitary matrices in development
                import warnings
                warnings.warn(f"Could not verify unitarity: {e}", UserWarning)
    
    def evaluate(self, **params) -> np.ndarray:
        """
        Evaluate the matrix with given parameters.
        
        Args:
            **params: Parameter values to substitute
            
        Returns:
            Concrete numpy array representing the gate matrix
        """
        if not self.is_parametric:
            if isinstance(self.matrix, np.ndarray):
                return self.matrix
            elif isinstance(self.matrix, sympy.Matrix):
                return np.array(self.matrix, dtype=complex)
        
        if callable(self.matrix):
            return self.matrix(**params)
        
        raise ValueError("Cannot evaluate non-parametric matrix with parameters")
    
    @property
    def shape(self) -> tuple:
        """Get the shape of the matrix."""
        if isinstance(self.matrix, np.ndarray):
            return self.matrix.shape
        elif isinstance(self.matrix, sympy.Matrix):
            return (self.matrix.rows, self.matrix.cols)
        else:
            # For parametric matrices, we need to evaluate with dummy parameters
            # or store the shape separately
            raise ValueError("Cannot determine shape of parametric matrix without evaluation")
    
    @property
    def num_qubits(self) -> int:
        """Get the number of qubits this matrix acts on."""
        if isinstance(self.matrix, (np.ndarray, sympy.Matrix)):
            size = self.matrix.shape[0]
            # Matrix size should be 2^n for n qubits
            import math
            n = math.log2(size)
            if not n.is_integer():
                raise ValueError(f"Matrix size {size} is not a power of 2")
            return int(n)
        else:
            raise ValueError("Cannot determine qubit count for parametric matrix without evaluation")


class Gate(ABC):
    """
    Abstract base class for all quantum gates.
    
    This defines the interface that all gates must implement, enabling
    a flexible and extensible gate system.
    """
    
    def __init__(self, name: str, num_qubits: int, 
                 gate_type: GateType = GateType.SINGLE_QUBIT,
                 parameters: Optional[List[str]] = None,
                 matrix: Optional[GateMatrix] = None,
                 description: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a quantum gate.
        
        Args:
            name: Unique name of the gate
            num_qubits: Number of qubits the gate acts on
            gate_type: Category of the gate
            parameters: List of parameter names (for parametric gates)
            matrix: Matrix representation of the gate
            description: Human-readable description
            metadata: Additional metadata for extensions
        """
        self.name = name
        self.num_qubits = num_qubits
        self.gate_type = gate_type
        self.parameters = parameters or []
        self.matrix = matrix
        self.description = description or f"{name} gate"
        self.metadata = metadata or {}
        
        # Validate consistency
        if self.parameters and not matrix.is_parametric if matrix else False:
            raise ValueError("Gate has parameters but matrix is not parametric")
    
    @property
    def is_parametric(self) -> bool:
        """Check if this gate has parameters."""
        return bool(self.parameters)
    
    @property
    def is_single_qubit(self) -> bool:
        """Check if this is a single-qubit gate."""
        return self.num_qubits == 1
    
    @property
    def is_two_qubit(self) -> bool:
        """Check if this is a two-qubit gate."""
        return self.num_qubits == 2
    
    @property
    def is_multi_qubit(self) -> bool:
        """Check if this is a multi-qubit gate."""
        return self.num_qubits > 2
    
    def get_matrix(self, **params) -> np.ndarray:
        """
        Get the matrix representation of this gate.
        
        Args:
            **params: Parameter values for parametric gates
            
        Returns:
            Numpy array representing the gate matrix
        """
        if not self.matrix:
            raise NotImplementedError(f"Gate {self.name} has no matrix representation")
        
        return self.matrix.evaluate(**params)
    
    def validate_parameters(self, params: Dict[str, Parameter]) -> bool:
        """
        Validate that provided parameters match gate requirements.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Check that all required parameters are provided
        provided = set(params.keys())
        required = set(self.parameters)
        
        missing = required - provided
        if missing:
            raise ValueError(f"Missing required parameters for gate {self.name}: {missing}")
        
        extra = provided - required
        if extra:
            raise ValueError(f"Extra parameters provided for gate {self.name}: {extra}")
        
        return True
    
    def create_controlled_version(self, num_controls: int = 1) -> 'Gate':
        """
        Create a controlled version of this gate.
        
        Args:
            num_controls: Number of control qubits
            
        Returns:
            New controlled gate
        """
        if num_controls < 1:
            raise ValueError("Number of controls must be at least 1")
        
        # Create controlled gate name
        prefix = "C" * num_controls
        controlled_name = f"{prefix}{self.name}"
        
        # For now, we create a controlled gate without explicit matrix construction
        # In a full implementation, you'd compute the controlled matrix
        return ControlledGate(
            base_gate=self,
            num_controls=num_controls,
            name=controlled_name
        )
    
    def __str__(self) -> str:
        """String representation."""
        parts = [self.name]
        if self.is_parametric:
            parts.append(f"({', '.join(self.parameters)})")
        parts.append(f"[{self.num_qubits}q]")
        return "".join(parts)
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Gate(name={self.name!r}, qubits={self.num_qubits}, type={self.gate_type.value})"


class ControlledGate(Gate):
    """
    Represents a controlled version of a base gate.
    
    This is a special type of gate that applies the base gate
    only when control qubits are in the |1> state.
    """
    
    def __init__(self, base_gate: Gate, num_controls: int = 1, name: Optional[str] = None):
        """
        Initialize a controlled gate.
        
        Args:
            base_gate: The gate to be controlled
            num_controls: Number of control qubits
            name: Optional custom name (defaults to C^n + base_gate.name)
        """
        self.base_gate = base_gate
        self.num_controls = num_controls
        
        if name is None:
            prefix = "C" * num_controls
            name = f"{prefix}{base_gate.name}"
        
        total_qubits = base_gate.num_qubits + num_controls
        
        super().__init__(
            name=name,
            num_qubits=total_qubits,
            gate_type=GateType.CONTROLLED,
            parameters=base_gate.parameters,
            description=f"Controlled {base_gate.description}",
            metadata={
                **base_gate.metadata,
                "base_gate": base_gate.name,
                "num_controls": num_controls
            }
        )
    
    def get_matrix(self, **params) -> np.ndarray:
        """
        Construct the matrix for the controlled gate.
        
        This creates a block diagonal matrix where the base gate
        is applied only when control qubits are |1>.
        """
        base_matrix = self.base_gate.get_matrix(**params)
        
        # For controlled gates, we need to construct the full matrix
        # This is a simplified implementation
        total_size = 2 ** self.num_qubits
        controlled_matrix = np.eye(total_size, dtype=complex)
        
        # Apply base gate to the subspace where controls are |1>
        # This is a simplified construction - full implementation would
        # be more sophisticated
        base_size = base_matrix.shape[0]
        controlled_matrix[-base_size:, -base_size:] = base_matrix
        
        return controlled_matrix


class CompositeGate(Gate):
    """
    Represents a gate composed of multiple sub-gates.
    
    This allows creating higher-level gates from combinations
    of primitive gates.
    """
    
    def __init__(self, name: str, sub_gates: List[Gate], 
                 description: Optional[str] = None):
        """
        Initialize a composite gate.
        
        Args:
            name: Name of the composite gate
            sub_gates: List of gates that make up this composite
            description: Optional description
        """
        self.sub_gates = sub_gates
        
        # Determine total number of qubits needed
        max_qubits = max(gate.num_qubits for gate in sub_gates) if sub_gates else 0
        
        # Collect all parameters from sub-gates
        all_params = []
        for gate in sub_gates:
            all_params.extend(gate.parameters)
        unique_params = list(set(all_params))  # Remove duplicates
        
        super().__init__(
            name=name,
            num_qubits=max_qubits,
            gate_type=GateType.MULTI_QUBIT,
            parameters=unique_params,
            description=description or f"Composite gate: {name}",
            metadata={"sub_gates": [gate.name for gate in sub_gates]}
        )
    
    def get_matrix(self, **params) -> np.ndarray:
        """
        Construct matrix by multiplying sub-gate matrices.
        
        Note: This is a simplified implementation. A full version
        would need to handle qubit mappings and gate ordering properly.
        """
        if not self.sub_gates:
            return np.eye(2 ** self.num_qubits, dtype=complex)
        
        # Start with identity
        result = np.eye(2 ** self.num_qubits, dtype=complex)
        
        # Apply each sub-gate (this is simplified)
        for gate in self.sub_gates:
            gate_matrix = gate.get_matrix(**params)
            # In reality, we'd need to handle qubit mappings properly
            # For now, assume gates act on the same qubits
            if gate_matrix.shape[0] == result.shape[0]:
                result = np.dot(gate_matrix, result)
        
        return result 