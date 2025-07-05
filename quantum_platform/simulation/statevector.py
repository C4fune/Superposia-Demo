"""
Enhanced State Vector Simulator with Multi-Shot Support

This module provides an enhanced quantum state vector simulator that supports
realistic multi-shot execution with proper sampling and measurement simulation.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter

from ..compiler.ir.circuit import QuantumCircuit
from ..compiler.ir.operation import Operation


@dataclass
class SimulationResult:
    """Result from quantum circuit simulation."""
    counts: Dict[str, int]
    shots: int
    execution_time: Optional[float] = None  # milliseconds
    statevector: Optional[np.ndarray] = None
    probabilities: Optional[Dict[str, float]] = None
    
    # Individual shot results (for detailed analysis)
    shot_results: Optional[List[str]] = None
    
    # Simulation metadata
    num_qubits: int = 0
    circuit_depth: int = 0
    gate_count: int = 0
    
    def get_probability(self, outcome: str) -> float:
        """Get probability of a specific outcome."""
        if self.probabilities and outcome in self.probabilities:
            return self.probabilities[outcome]
        elif self.counts and self.shots > 0:
            return self.counts.get(outcome, 0) / self.shots
        return 0.0
    
    def get_most_frequent(self) -> Tuple[str, int]:
        """Get the most frequent outcome."""
        if not self.counts:
            return "", 0
        
        max_outcome = max(self.counts.items(), key=lambda x: x[1])
        return max_outcome


class StateVectorSimulator:
    """Enhanced quantum state vector simulator with multi-shot support."""
    
    def __init__(self, max_qubits: int = 30, seed: Optional[int] = None):
        self.max_qubits = max_qubits
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Simulation state
        self.statevector: Optional[np.ndarray] = None
        self.num_qubits = 0
        self.measurement_bases: Dict[int, str] = {}  # qubit -> basis
        
    def run(self, circuit: QuantumCircuit, shots: int = 1000,
            return_statevector: bool = False,
            return_individual_shots: bool = False) -> SimulationResult:
        """
        Run quantum circuit simulation with multi-shot execution.
        
        Args:
            circuit: Quantum circuit to simulate
            shots: Number of measurement shots
            return_statevector: Whether to return final statevector
            return_individual_shots: Whether to return individual shot results
        """
        
        if circuit.num_qubits > self.max_qubits:
            raise ValueError(
                f"Circuit has {circuit.num_qubits} qubits, "
                f"but simulator supports maximum {self.max_qubits}"
            )
        
        # Ensure reproducible results by resetting seed before each run
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        
        import time
        start_time = time.time()
        
        # Initialize simulation
        self.num_qubits = circuit.num_qubits
        self._initialize_statevector()
        
        # Apply circuit operations
        has_measurements = self._apply_circuit(circuit)
        
        # Perform measurements based on shots
        if has_measurements or shots > 1:
            # Multi-shot execution with sampling
            counts, shot_results = self._perform_multi_shot_measurement(shots)
        else:
            # Single execution - return statevector probabilities
            counts = self._get_probability_counts(shots)
            shot_results = None
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Calculate probabilities
        probabilities = {
            outcome: count / shots for outcome, count in counts.items()
        } if shots > 0 else {}
        
        return SimulationResult(
            counts=counts,
            shots=shots,
            execution_time=execution_time,
            statevector=self.statevector.copy() if return_statevector else None,
            probabilities=probabilities,
            shot_results=shot_results if return_individual_shots else None,
            num_qubits=self.num_qubits,
            circuit_depth=getattr(circuit, 'depth', len(circuit.operations)),
            gate_count=len(circuit.operations)
        )
    
    def _initialize_statevector(self):
        """Initialize the quantum statevector to |00...0⟩."""
        dimension = 2 ** self.num_qubits
        self.statevector = np.zeros(dimension, dtype=complex)
        self.statevector[0] = 1.0  # |00...0⟩ state
        self.measurement_bases = {}
    
    def _apply_circuit(self, circuit: QuantumCircuit) -> bool:
        """
        Apply all operations in the circuit.
        
        Returns:
            True if circuit contains measurement operations
        """
        has_measurements = False
        
        for operation in circuit.operations:
            if self._is_measurement(operation):
                has_measurements = True
                self._apply_measurement(operation)
            else:
                self._apply_gate(operation)
        
        return has_measurements
    
    def _is_measurement(self, operation: Operation) -> bool:
        """Check if operation is a measurement."""
        return operation.__class__.__name__.lower() in ['measure', 'measurement']
    
    def _apply_gate(self, operation: Operation):
        """Apply a quantum gate operation."""
        # Get the gate name from the operation
        gate_name = getattr(operation, 'name', None)
        if not gate_name:
            # Fallback to class name analysis
            gate_name = operation.__class__.__name__.lower()
        
        # Normalize gate name
        gate_name = gate_name.upper()
        targets = [target.id for target in operation.targets]
        
        # Handle controls for two-qubit gates
        controls = []
        if hasattr(operation, 'controls') and operation.controls:
            controls = [control.id for control in operation.controls]
        
        if gate_name == 'H':
            self._apply_hadamard(targets[0])
        elif gate_name == 'X':
            self._apply_pauli_x(targets[0])
        elif gate_name == 'Y':
            self._apply_pauli_y(targets[0])
        elif gate_name == 'Z':
            self._apply_pauli_z(targets[0])
        elif gate_name in ['CNOT', 'CX']:
            # For CNOT, we need to get control from the operation
            if controls:
                control_qubit = controls[0]
            else:
                # If no explicit control, assume it's encoded in the operation string
                # Check if operation has control information
                op_str = str(operation)
                if 'ctrl:' in op_str:
                    # Extract control qubit from string representation
                    import re
                    match = re.search(r'ctrl:\[q(\d+)\]', op_str)
                    if match:
                        control_qubit = int(match.group(1))
                    else:
                        # Fallback: assume first qubit is control for CNOT
                        control_qubit = 0 if targets[0] != 0 else 1
                else:
                    # Default: assume targets[0] is control for compatibility
                    control_qubit = targets[0]
                    
            target_qubit = targets[0] if targets else 1
            self._apply_cnot(control_qubit, target_qubit)
        elif gate_name == 'RX':
            angle = list(operation.parameters.values())[0] if operation.parameters else 0
            self._apply_rotation_x(targets[0], angle)
        elif gate_name == 'RY':
            angle = list(operation.parameters.values())[0] if operation.parameters else 0
            self._apply_rotation_y(targets[0], angle)
        elif gate_name == 'RZ':
            angle = list(operation.parameters.values())[0] if operation.parameters else 0
            self._apply_rotation_z(targets[0], angle)
        elif gate_name == 'SWAP':
            if len(targets) >= 2:
                self._apply_swap(targets[0], targets[1])
        else:
            # For unknown gates, just skip them
            pass
    
    def _apply_measurement(self, operation: Operation):
        """Apply measurement operation (marks qubits for measurement)."""
        for target in operation.targets:
            self.measurement_bases[target.id] = 'computational'
    
    def _apply_hadamard(self, qubit: int):
        """Apply Hadamard gate."""
        h_matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self._apply_single_qubit_gate(qubit, h_matrix)
    
    def _apply_pauli_x(self, qubit: int):
        """Apply Pauli-X gate."""
        x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        self._apply_single_qubit_gate(qubit, x_matrix)
    
    def _apply_pauli_y(self, qubit: int):
        """Apply Pauli-Y gate."""
        y_matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._apply_single_qubit_gate(qubit, y_matrix)
    
    def _apply_pauli_z(self, qubit: int):
        """Apply Pauli-Z gate."""
        z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        self._apply_single_qubit_gate(qubit, z_matrix)
    
    def _apply_rotation_x(self, qubit: int, angle: float):
        """Apply rotation around X axis."""
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        rx_matrix = np.array([
            [cos_half, -1j * sin_half],
            [-1j * sin_half, cos_half]
        ], dtype=complex)
        self._apply_single_qubit_gate(qubit, rx_matrix)
    
    def _apply_rotation_y(self, qubit: int, angle: float):
        """Apply rotation around Y axis."""
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        ry_matrix = np.array([
            [cos_half, -sin_half],
            [sin_half, cos_half]
        ], dtype=complex)
        self._apply_single_qubit_gate(qubit, ry_matrix)
    
    def _apply_rotation_z(self, qubit: int, angle: float):
        """Apply rotation around Z axis."""
        exp_pos = np.exp(1j * angle / 2)
        exp_neg = np.exp(-1j * angle / 2)
        rz_matrix = np.array([
            [exp_neg, 0],
            [0, exp_pos]
        ], dtype=complex)
        self._apply_single_qubit_gate(qubit, rz_matrix)
    
    def _apply_cnot(self, control: int, target: int):
        """Apply CNOT gate."""
        # Create CNOT matrix for the full system
        dimension = 2 ** self.num_qubits
        cnot_matrix = np.eye(dimension, dtype=complex)
        
        # Apply CNOT logic
        for i in range(dimension):
            # Extract bit values
            control_bit = (i >> (self.num_qubits - 1 - control)) & 1
            target_bit = (i >> (self.num_qubits - 1 - target)) & 1
            
            if control_bit == 1:
                # Flip target bit
                flipped_target = 1 - target_bit
                j = i ^ (1 << (self.num_qubits - 1 - target))
                cnot_matrix[j, i] = 1.0
                cnot_matrix[i, i] = 0.0
        
        self.statevector = cnot_matrix @ self.statevector
    
    def _apply_swap(self, qubit1: int, qubit2: int):
        """Apply SWAP gate."""
        # SWAP = CNOT(a,b) CNOT(b,a) CNOT(a,b)
        self._apply_cnot(qubit1, qubit2)
        self._apply_cnot(qubit2, qubit1)
        self._apply_cnot(qubit1, qubit2)
    
    def _apply_single_qubit_gate(self, qubit: int, gate_matrix: np.ndarray):
        """Apply a single-qubit gate to the statevector."""
        # Create full system matrix
        dimension = 2 ** self.num_qubits
        
        # Apply gate using tensor product structure
        new_statevector = np.zeros_like(self.statevector)
        
        for i in range(dimension):
            # Extract qubit state
            qubit_state = (i >> (self.num_qubits - 1 - qubit)) & 1
            
            # Apply gate to this qubit
            for new_qubit_state in range(2):
                gate_element = gate_matrix[new_qubit_state, qubit_state]
                
                if abs(gate_element) > 1e-12:  # Non-zero element
                    # Calculate new state index
                    j = i ^ ((qubit_state ^ new_qubit_state) << (self.num_qubits - 1 - qubit))
                    new_statevector[j] += gate_element * self.statevector[i]
        
        self.statevector = new_statevector
    
    def _apply_generic_gate(self, operation: Operation):
        """Apply a generic gate operation (placeholder)."""
        # For now, just pass - unknown gates are ignored
        pass
    
    def _perform_multi_shot_measurement(self, shots: int) -> Tuple[Dict[str, int], List[str]]:
        """Perform multi-shot measurement with proper sampling."""
        
        # Calculate probabilities from statevector
        probabilities = np.abs(self.statevector) ** 2
        
        # Sample outcomes
        shot_results = []
        for _ in range(shots):
            # Sample from probability distribution
            outcome_index = np.random.choice(len(probabilities), p=probabilities)
            
            # Convert to bitstring
            bitstring = format(outcome_index, f'0{self.num_qubits}b')
            shot_results.append(bitstring)
        
        # Count outcomes
        counts = Counter(shot_results)
        
        return dict(counts), shot_results
    
    def _get_probability_counts(self, shots: int) -> Dict[str, int]:
        """Get counts based on exact probabilities (for single-shot or statevector mode)."""
        
        probabilities = np.abs(self.statevector) ** 2
        counts = {}
        
        for i, prob in enumerate(probabilities):
            if prob > 1e-12:  # Only include non-zero probabilities
                bitstring = format(i, f'0{self.num_qubits}b')
                counts[bitstring] = int(round(prob * shots))
        
        return counts
    
    def get_statevector(self) -> np.ndarray:
        """Get current statevector."""
        return self.statevector.copy() if self.statevector is not None else None
    
    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities."""
        if self.statevector is None:
            return {}
        
        probabilities = np.abs(self.statevector) ** 2
        result = {}
        
        for i, prob in enumerate(probabilities):
            if prob > 1e-12:
                bitstring = format(i, f'0{self.num_qubits}b')
                result[bitstring] = float(prob)
        
        return result
    
    def reset(self):
        """Reset simulator state."""
        self.statevector = None
        self.num_qubits = 0
        self.measurement_bases = {}


# Factory function for backward compatibility
def create_simulator(max_qubits: int = 30, seed: Optional[int] = None) -> StateVectorSimulator:
    """Create a state vector simulator instance."""
    return StateVectorSimulator(max_qubits=max_qubits, seed=seed) 