"""
Base simulation interfaces and result classes.

This module defines the abstract interfaces that all quantum simulators
must implement, as well as standard result formats.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np

from quantum_platform.compiler.ir.circuit import QuantumCircuit


@dataclass
class SimulationResult:
    """
    Results from quantum circuit simulation.
    
    This class encapsulates all information returned from a simulation,
    including measurement outcomes, final state, and execution metadata.
    """
    
    # Basic execution info
    circuit_name: str
    shots: int
    execution_time: float  # seconds
    
    # Measurement results
    measurement_counts: Dict[str, int] = field(default_factory=dict)  # bitstring -> count
    classical_registers: Dict[str, List[int]] = field(default_factory=dict)  # register -> values
    
    # State information (optional, for debugging/analysis)
    final_state: Optional[np.ndarray] = None  # final state vector
    final_probabilities: Optional[Dict[str, float]] = None  # bitstring -> probability
    
    # Execution metadata
    success: bool = True
    error_message: Optional[str] = None
    memory_used: Optional[int] = None  # bytes
    simulation_method: str = "unknown"
    
    # Raw shot data (if requested)
    shot_data: Optional[List[Dict[str, int]]] = None  # per-shot measurements
    
    def get_counts(self, register_name: Optional[str] = None) -> Dict[str, int]:
        """
        Get measurement counts.
        
        Args:
            register_name: Specific register to get counts for, or None for all
            
        Returns:
            Dictionary mapping bitstrings to counts
        """
        if register_name and register_name in self.classical_registers:
            # Convert register values to counts
            values = self.classical_registers[register_name]
            counts = {}
            for value in values:
                bitstring = format(value, 'b')
                counts[bitstring] = counts.get(bitstring, 0) + 1
            return counts
        
        return self.measurement_counts.copy()
    
    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities."""
        if self.final_probabilities:
            return self.final_probabilities.copy()
        
        # Calculate from counts
        total = sum(self.measurement_counts.values())
        if total == 0:
            return {}
        
        return {
            bitstring: count / total 
            for bitstring, count in self.measurement_counts.items()
        }
    
    def get_expectation_value(self, observable: str) -> float:
        """
        Calculate expectation value of an observable.
        
        Args:
            observable: Observable specification (e.g., "Z0", "X1", "Z0*Z1")
            
        Returns:
            Expectation value
        """
        # For now, just implement Z expectation on single qubits
        if observable.startswith('Z') and len(observable) == 2:
            qubit_idx = int(observable[1])
            prob_0 = 0.0
            prob_1 = 0.0
            
            for bitstring, prob in self.get_probabilities().items():
                if len(bitstring) > qubit_idx:
                    if bitstring[-(qubit_idx+1)] == '0':
                        prob_0 += prob
                    else:
                        prob_1 += prob
            
            return prob_0 - prob_1
        
        raise NotImplementedError(f"Observable {observable} not implemented")
    
    def __str__(self) -> str:
        """String representation of results."""
        lines = [
            f"SimulationResult({self.circuit_name})",
            f"  Shots: {self.shots}",
            f"  Time: {self.execution_time:.3f}s", 
            f"  Success: {self.success}",
        ]
        
        if self.measurement_counts:
            lines.append("  Measurement counts:")
            for bitstring, count in sorted(self.measurement_counts.items()):
                prob = count / self.shots if self.shots > 0 else 0
                lines.append(f"    {bitstring}: {count} ({prob:.3f})")
        
        if self.error_message:
            lines.append(f"  Error: {self.error_message}")
            
        return "\n".join(lines)


class QuantumSimulator(ABC):
    """
    Abstract base class for quantum circuit simulators.
    
    All simulator implementations must inherit from this class and implement
    the required methods. This provides a consistent interface for different
    simulation backends.
    """
    
    def __init__(self, name: str = "QuantumSimulator"):
        """
        Initialize the simulator.
        
        Args:
            name: Human-readable name for this simulator
        """
        self.name = name
        self._max_qubits = 30  # Default limit for state vector simulation
        self._default_shots = 1024
    
    @property 
    def max_qubits(self) -> int:
        """Maximum number of qubits this simulator can handle."""
        return self._max_qubits
    
    @property
    def supports_shots(self) -> bool:
        """Whether this simulator supports multiple shots."""
        return True
    
    @property 
    def supports_statevector(self) -> bool:
        """Whether this simulator can return state vectors."""
        return False
    
    @abstractmethod
    def run(self, 
            circuit: QuantumCircuit, 
            shots: int = 1024,
            initial_state: Optional[Union[str, np.ndarray]] = None,
            **kwargs) -> SimulationResult:
        """
        Execute a quantum circuit.
        
        Args:
            circuit: The quantum circuit to simulate
            shots: Number of measurement shots to perform
            initial_state: Initial quantum state (|0...0> if None)
            **kwargs: Additional simulator-specific options
            
        Returns:
            Simulation results
            
        Raises:
            ValueError: If circuit is invalid or too large
            RuntimeError: If simulation fails
        """
        pass
    
    def validate_circuit(self, circuit: QuantumCircuit) -> None:
        """
        Validate that this simulator can run the given circuit.
        
        Args:
            circuit: Circuit to validate
            
        Raises:
            ValueError: If circuit cannot be simulated
        """
        if circuit.num_qubits > self.max_qubits:
            raise ValueError(
                f"Circuit has {circuit.num_qubits} qubits, but simulator "
                f"can only handle {self.max_qubits}"
            )
        
        # Check for unsupported operations
        unsupported = []
        for op in circuit.operations:
            if not self._supports_operation(op):
                unsupported.append(op.name)
        
        if unsupported:
            raise ValueError(
                f"Simulator does not support operations: {set(unsupported)}"
            )
    
    def _supports_operation(self, operation) -> bool:
        """Check if this simulator supports a given operation."""
        # By default, assume all operations are supported
        # Subclasses can override for specific limitations
        return True
    
    def estimate_memory(self, num_qubits: int) -> int:
        """
        Estimate memory requirements for simulating a circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            
        Returns:
            Estimated memory in bytes
        """
        # State vector: 2^n complex numbers * 16 bytes each
        return 2**num_qubits * 16
    
    def estimate_runtime(self, circuit: QuantumCircuit, shots: int) -> float:
        """
        Estimate runtime for simulating a circuit.
        
        Args:
            circuit: Circuit to simulate
            shots: Number of shots
            
        Returns:
            Estimated runtime in seconds
        """
        # Very rough estimate: assume each gate takes ~1ms
        # and overhead of ~1ms per shot for measurements
        gate_time = circuit.num_operations * 0.001
        shot_time = shots * 0.001
        return gate_time + shot_time
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}(max_qubits={self.max_qubits})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()


class SimulatorError(Exception):
    """Exception raised by quantum simulators."""
    pass


class ResourceError(SimulatorError):
    """Exception raised when simulator resource limits are exceeded."""
    pass


class StateError(SimulatorError):
    """Exception raised when quantum state becomes invalid."""
    pass 