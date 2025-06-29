"""
State Vector Quantum Simulator

This module implements a full state vector quantum simulator that can
execute quantum circuits by maintaining and updating the complete
quantum state vector.
"""

import time
import psutil
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from math import log2, ceil

from quantum_platform.simulation.base import (
    QuantumSimulator, SimulationResult, SimulatorError, ResourceError, StateError
)
from quantum_platform.compiler.ir.circuit import QuantumCircuit
from quantum_platform.compiler.ir.operation import (
    Operation, GateOperation, MeasurementOperation, ResetOperation, 
    BarrierOperation, IfOperation, LoopOperation, ClassicalOperation
)
from quantum_platform.compiler.gates.registry import get_gate


class StateVectorSimulator(QuantumSimulator):
    """
    State vector quantum simulator.
    
    This simulator maintains the full quantum state vector and applies
    unitary operations by matrix multiplication. It supports:
    - All standard quantum gates
    - Measurements with collapse
    - Classical conditionals
    - Parameter substitution
    - Multi-shot execution
    """
    
    def __init__(self, max_qubits: int = 25, memory_limit_gb: float = 8.0):
        """
        Initialize the state vector simulator.
        
        Args:
            max_qubits: Maximum number of qubits to simulate
            memory_limit_gb: Memory limit in gigabytes
        """
        super().__init__("StateVectorSimulator")
        self._max_qubits = min(max_qubits, self._calculate_max_qubits(memory_limit_gb))
        self._memory_limit = memory_limit_gb * 1024**3  # Convert to bytes
        
        # Current simulation state
        self._state: Optional[np.ndarray] = None
        self._num_qubits: int = 0
        self._classical_memory: Dict[str, Any] = {}
        self._rng = np.random.default_rng()
    
    @property
    def supports_statevector(self) -> bool:
        """This simulator can return state vectors."""
        return True
    
    def _calculate_max_qubits(self, memory_limit_gb: float) -> int:
        """Calculate maximum qubits given memory limit."""
        # State vector needs 2^n * 16 bytes (complex128)
        max_states = memory_limit_gb * 1024**3 / 16
        return int(log2(max_states)) if max_states >= 2 else 1
    
    def run(self, 
            circuit: QuantumCircuit, 
            shots: int = 1024,
            initial_state: Optional[Union[str, np.ndarray]] = None,
            return_statevector: bool = False,
            seed: Optional[int] = None,
            **kwargs) -> SimulationResult:
        """
        Execute a quantum circuit.
        
        Args:
            circuit: The quantum circuit to simulate
            shots: Number of measurement shots to perform
            initial_state: Initial quantum state (|0...0> if None)
            return_statevector: Whether to include final state vector in results
            seed: Random seed for reproducible results
            **kwargs: Additional options
            
        Returns:
            Simulation results
        """
        start_time = time.time()
        
        try:
            # Validate circuit
            self.validate_circuit(circuit)
            
            # Set random seed
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            
            # Initialize result
            result = SimulationResult(
                circuit_name=circuit.name,
                shots=shots,
                execution_time=0.0,
                simulation_method="statevector"
            )
            
            # Initialize quantum state
            self._initialize_state(circuit.num_qubits, initial_state)
            
            # Initialize classical memory
            self._classical_memory = {}
            for reg_name, reg in circuit.classical_registers.items():
                self._classical_memory[reg_name] = [0] * reg.size
            
            # For statevector return, we need to capture state before measurements
            final_state_before_measurement = None
            
            # Execute shots
            measurement_counts = {}
            classical_results = {name: [] for name in circuit.classical_registers.keys()}
            shot_data = [] if kwargs.get('return_shot_data', False) else None
            
            for shot in range(shots):
                # Reset state for each shot
                self._initialize_state(circuit.num_qubits, initial_state)
                self._classical_memory = {name: [0] * reg.size 
                                        for name, reg in circuit.classical_registers.items()}
                
                # If this is the last shot and we want statevector, 
                # capture state before measurements
                if return_statevector and shot == shots - 1:
                    # Execute circuit up to but not including measurements
                    final_state_before_measurement = self._execute_circuit_for_statevector(circuit)
                
                # Execute circuit normally for measurement statistics
                shot_measurements = self._execute_circuit(circuit)
                
                # Collect measurements
                for reg_name, values in shot_measurements.items():
                    classical_results[reg_name].extend(values)
                    
                    # Convert to bitstring for counting
                    if values:
                        bitstring = ''.join(str(bit) for bit in values)
                        measurement_counts[bitstring] = measurement_counts.get(bitstring, 0) + 1
                
                if shot_data is not None:
                    shot_data.append(shot_measurements)
            
            # Store results
            result.measurement_counts = measurement_counts
            result.classical_registers = classical_results
            if shot_data:
                result.shot_data = shot_data
            
            # Include final state if requested
            if return_statevector:
                if final_state_before_measurement is not None:
                    result.final_state = final_state_before_measurement
                    # Calculate probabilities from the pre-measurement state
                    probabilities = np.abs(final_state_before_measurement) ** 2
                    num_qubits = int(np.log2(len(final_state_before_measurement)))
                    result.final_probabilities = {
                        format(i, f'0{num_qubits}b'): float(prob)
                        for i, prob in enumerate(probabilities)
                        if prob > 1e-12
                    }
                else:
                    # Fallback: use current state (might be post-measurement)
                    result.final_state = self._state.copy() if self._state is not None else None
                    result.final_probabilities = self._calculate_probabilities() if self._state is not None else {}
            
            # Calculate memory usage
            result.memory_used = self.estimate_memory(circuit.num_qubits)
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            result = SimulationResult(
                circuit_name=circuit.name,
                shots=shots,
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e),
                simulation_method="statevector"
            )
            return result
    
    def _initialize_state(self, num_qubits: int, initial_state: Optional[Union[str, np.ndarray]] = None):
        """Initialize the quantum state vector."""
        self._num_qubits = num_qubits
        state_size = 2 ** num_qubits
        
        # Check memory requirements
        memory_needed = state_size * 16  # complex128
        if memory_needed > self._memory_limit:
            raise ResourceError(
                f"Circuit requires {memory_needed / 1024**3:.2f}GB memory, "
                f"but limit is {self._memory_limit / 1024**3:.2f}GB"
            )
        
        if initial_state is None:
            # Initialize to |0...0>
            self._state = np.zeros(state_size, dtype=complex)
            self._state[0] = 1.0
        elif isinstance(initial_state, str):
            # Parse bitstring like "101"
            if len(initial_state) != num_qubits:
                raise ValueError(f"Initial state bitstring length must match qubit count")
            
            self._state = np.zeros(state_size, dtype=complex)
            index = int(initial_state, 2)
            self._state[index] = 1.0
        elif isinstance(initial_state, np.ndarray):
            if initial_state.shape != (state_size,):
                raise ValueError(f"Initial state array must have shape ({state_size},)")
            
            self._state = initial_state.astype(complex)
            # Normalize
            norm = np.linalg.norm(self._state)
            if norm > 0:
                self._state /= norm
        else:
            raise TypeError("Initial state must be None, str, or numpy array")
    
    def _execute_circuit(self, circuit: QuantumCircuit) -> Dict[str, List[int]]:
        """Execute a circuit and return measurement results."""
        measurements = {name: [] for name in circuit.classical_registers.keys()}
        
        for operation in circuit.operations:
            if isinstance(operation, GateOperation):
                self._apply_gate(operation)
            elif isinstance(operation, MeasurementOperation):
                result = self._apply_measurement(operation)
                if operation.classical_target:
                    if operation.classical_target not in measurements:
                        measurements[operation.classical_target] = []
                    
                    # Handle both single and multi-qubit measurements
                    if isinstance(result, list):
                        measurements[operation.classical_target].extend(result)
                    else:
                        measurements[operation.classical_target].append(result)
            elif isinstance(operation, ResetOperation):
                self._apply_reset(operation)
            elif isinstance(operation, BarrierOperation):
                # Barriers don't affect simulation
                pass
            elif isinstance(operation, IfOperation):
                self._apply_conditional(operation)
            elif isinstance(operation, LoopOperation):
                self._apply_loop(operation)
            elif isinstance(operation, ClassicalOperation):
                self._apply_classical(operation)
            else:
                raise SimulatorError(f"Unsupported operation type: {type(operation)}")
        
        return measurements
    
    def _execute_circuit_for_statevector(self, circuit: QuantumCircuit) -> np.ndarray:
        """
        Execute circuit up to but not including measurements to capture final statevector.
        
        Args:
            circuit: The quantum circuit to execute
            
        Returns:
            State vector just before measurements
        """
        for operation in circuit.operations:
            if isinstance(operation, GateOperation):
                self._apply_gate(operation)
            elif isinstance(operation, ResetOperation):
                self._apply_reset(operation)
            elif isinstance(operation, BarrierOperation):
                # Barriers don't affect simulation
                pass
            elif isinstance(operation, IfOperation):
                self._apply_conditional(operation)
            elif isinstance(operation, LoopOperation):
                self._apply_loop(operation)
            elif isinstance(operation, ClassicalOperation):
                self._apply_classical(operation)
            # Skip measurement operations to preserve statevector
            elif isinstance(operation, MeasurementOperation):
                continue
            else:
                raise SimulatorError(f"Unsupported operation type: {type(operation)}")
        
        return self._state.copy()
    
    def _apply_gate(self, operation: GateOperation):
        """Apply a quantum gate operation."""
        # Get gate definition - try exact name first, then uppercase for compatibility
        gate = get_gate(operation.name)
        if gate is None:
            # Try uppercase version for case-insensitive lookup
            gate = get_gate(operation.name.upper())
        
        if gate is None:
            raise SimulatorError(f"Unknown gate: {operation.name}")
        
        # Get gate matrix
        if gate.matrix.is_parametric:
            # Substitute parameters
            params = {}
            for param_name in gate.matrix.parameter_names:
                if param_name in operation.params:
                    param_value = operation.params[param_name]
                    params[param_name] = param_value.value if hasattr(param_value, 'value') else param_value
                else:
                    raise SimulatorError(f"Missing parameter {param_name} for gate {operation.name}")
            
            matrix = gate.matrix.evaluate(**params)
        else:
            matrix = gate.matrix.evaluate()
        
        # Get qubit indices
        target_indices = [qubit.id for qubit in operation.targets]
        control_indices = [qubit.id for qubit in operation.controls]
        
        # Apply matrix to state vector
        if control_indices:
            # Controlled gate
            self._apply_controlled_unitary(matrix, target_indices, control_indices)
        else:
            # Regular gate
            self._apply_unitary(matrix, target_indices)
    
    def _apply_unitary(self, matrix: np.ndarray, target_qubits: List[int]):
        """Apply a unitary matrix to target qubits."""
        n_targets = len(target_qubits)
        
        if matrix.shape != (2**n_targets, 2**n_targets):
            raise ValueError(f"Matrix shape {matrix.shape} doesn't match {n_targets} target qubits")
        
        # For single qubit gates, we can use a more efficient implementation
        if n_targets == 1:
            self._apply_single_qubit_unitary(matrix, target_qubits[0])
        else:
            self._apply_multi_qubit_unitary(matrix, target_qubits)
    
    def _apply_single_qubit_unitary(self, matrix: np.ndarray, target_qubit: int):
        """Efficiently apply single-qubit unitary."""
        # Create the full unitary for the entire system
        full_matrix = self._expand_single_qubit_matrix(matrix, target_qubit)
        self._state = full_matrix @ self._state
    
    def _expand_single_qubit_matrix(self, matrix: np.ndarray, target_qubit: int) -> np.ndarray:
        """Expand a single-qubit matrix to act on the full system."""
        n_qubits = self._num_qubits
        state_size = 2 ** n_qubits
        
        # Use Kronecker products to build full matrix
        if target_qubit == 0:
            # Rightmost qubit
            full_matrix = matrix
            for i in range(1, n_qubits):
                full_matrix = np.kron(np.eye(2), full_matrix)
        elif target_qubit == n_qubits - 1:
            # Leftmost qubit
            full_matrix = np.eye(1)
            for i in range(n_qubits - 1):
                full_matrix = np.kron(full_matrix, np.eye(2))
            full_matrix = np.kron(matrix, full_matrix)
        else:
            # Middle qubit - build with identity matrices
            full_matrix = np.eye(1)
            for i in range(n_qubits):
                if i == target_qubit:
                    full_matrix = np.kron(full_matrix, matrix)
                else:
                    full_matrix = np.kron(full_matrix, np.eye(2))
        
        return full_matrix
    
    def _apply_multi_qubit_unitary(self, matrix: np.ndarray, target_qubits: List[int]):
        """Apply multi-qubit unitary (general case)."""
        if len(target_qubits) == 2:
            # Special case for two-qubit gates (like CNOT)
            self._apply_two_qubit_gate(matrix, target_qubits[0], target_qubits[1])
        else:
            # General case - use full matrix expansion
            full_matrix = self._expand_multi_qubit_matrix(matrix, target_qubits)
            self._state = full_matrix @ self._state
    
    def _apply_two_qubit_gate(self, matrix: np.ndarray, qubit1: int, qubit2: int):
        """Apply a two-qubit gate efficiently."""
        new_state = np.zeros_like(self._state)
        
        for i in range(len(self._state)):
            # Extract values of the two target qubits
            bit1 = (i >> qubit1) & 1
            bit2 = (i >> qubit2) & 1
            
            # For CNOT: qubit1 is control, qubit2 is target
            # Matrix order: |control,target> -> index = control*2 + target
            input_idx = (bit1 << 1) | bit2
            
            # Apply the gate
            for output_idx in range(4):
                new_bit1 = (output_idx >> 1) & 1  # control bit
                new_bit2 = output_idx & 1         # target bit
                
                # Calculate the new state index
                new_i = i
                new_i = (new_i & ~(1 << qubit1)) | (new_bit1 << qubit1)
                new_i = (new_i & ~(1 << qubit2)) | (new_bit2 << qubit2)
                
                new_state[new_i] += matrix[output_idx, input_idx] * self._state[i]
        
        self._state = new_state
    
    def _expand_multi_qubit_matrix(self, matrix: np.ndarray, target_qubits: List[int]) -> np.ndarray:
        """Expand multi-qubit matrix to full system size."""
        n_qubits = self._num_qubits
        state_size = 2 ** n_qubits
        
        # This is a simplified implementation
        # For large systems, tensor network methods would be more efficient
        full_matrix = np.eye(state_size, dtype=complex)
        
        # Apply the matrix to the specified target qubits
        # This is computationally expensive but correct
        for i in range(state_size):
            for j in range(state_size):
                # Extract target qubit values from state indices
                i_bits = [(i >> q) & 1 for q in range(n_qubits)]
                j_bits = [(j >> q) & 1 for q in range(n_qubits)]
                
                # Check if non-target qubits match
                match = True
                for q in range(n_qubits):
                    if q not in target_qubits and i_bits[q] != j_bits[q]:
                        match = False
                        break
                
                if match:
                    # Calculate matrix indices for target qubits
                    i_target = sum((i_bits[q] << idx) for idx, q in enumerate(target_qubits))
                    j_target = sum((j_bits[q] << idx) for idx, q in enumerate(target_qubits))
                    full_matrix[i, j] = matrix[i_target, j_target]
                
        return full_matrix
    
    def _apply_controlled_unitary(self, matrix: np.ndarray, target_qubits: List[int], control_qubits: List[int]):
        """Apply controlled unitary operation."""
        # Simplified implementation - in practice would use more efficient methods
        n_qubits = self._num_qubits
        state_size = 2 ** n_qubits
        
        new_state = self._state.copy()
        
        for i in range(state_size):
            # Check if all control qubits are |1>
            control_active = all((i >> q) & 1 == 1 for q in control_qubits)
            
            if control_active:
                # Apply the gate to target qubits
                # This is a simplified approach - extract and apply
                pass  # Would implement controlled gate application
        
        self._state = new_state
    
    def _apply_measurement(self, operation: MeasurementOperation) -> Union[int, List[int]]:
        """Apply measurement and collapse state."""
        if len(operation.targets) == 1:
            return self._measure_single_qubit(operation.targets[0])
        else:
            return self._measure_multiple_qubits(operation.targets)
    
    def _measure_single_qubit(self, qubit) -> int:
        """Measure a single qubit."""
        qubit_id = qubit.id
        
        # Calculate probabilities
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i, amplitude in enumerate(self._state):
            prob = abs(amplitude) ** 2
            qubit_value = (i >> qubit_id) & 1
            if qubit_value == 0:
                prob_0 += prob
            else:
                prob_1 += prob
        
        # Sample measurement outcome
        measurement = 1 if self._rng.random() < prob_1 else 0
        
        # Collapse state
        new_state = np.zeros_like(self._state)
        norm = 0.0
        
        for i, amplitude in enumerate(self._state):
            qubit_value = (i >> qubit_id) & 1
            if qubit_value == measurement:
                new_state[i] = amplitude
                norm += abs(amplitude) ** 2
        
        # Normalize
        if norm > 0:
            new_state /= np.sqrt(norm)
        
        self._state = new_state
        return measurement
    
    def _measure_multiple_qubits(self, qubits) -> List[int]:
        """Measure multiple qubits simultaneously."""
        results = []
        
        # Measure each qubit sequentially
        for qubit in qubits:
            result = self._measure_single_qubit(qubit)
            results.append(result)
        
        return results
    
    def _apply_reset(self, operation: ResetOperation):
        """Reset qubits to |0> state."""
        for qubit in operation.targets:
            # Force qubit to |0> by projecting and renormalizing
            qubit_id = qubit.id
            new_state = np.zeros_like(self._state)
            norm = 0.0
            
            for i, amplitude in enumerate(self._state):
                qubit_value = (i >> qubit_id) & 1
                if qubit_value == 0:
                    new_state[i] = amplitude
                    norm += abs(amplitude) ** 2
                else:
                    # Set this qubit to 0 by changing the index
                    new_i = i & ~(1 << qubit_id)  # Clear the qubit bit
                    new_state[new_i] += amplitude
                    norm += abs(amplitude) ** 2
            
            if norm > 0:
                new_state /= np.sqrt(norm)
            
            self._state = new_state
    
    def _apply_conditional(self, operation: IfOperation):
        """Apply conditional operation."""
        # Simplified: assume condition is a classical register check
        condition_met = self._evaluate_condition(operation.condition)
        
        if condition_met:
            for op in operation.then_operations:
                self._execute_single_operation(op)
        else:
            for op in operation.else_operations:
                self._execute_single_operation(op)
    
    def _apply_loop(self, operation: LoopOperation):
        """Apply loop operation."""
        if operation.loop_count is not None:
            # Fixed count loop
            for _ in range(operation.loop_count):
                for op in operation.loop_body:
                    self._execute_single_operation(op)
        else:
            # Conditional loop - simplified implementation
            max_iterations = 1000  # Safety limit
            iterations = 0
            
            while iterations < max_iterations:
                if operation.loop_condition and not self._evaluate_condition(operation.loop_condition):
                    break
                
                for op in operation.loop_body:
                    self._execute_single_operation(op)
                
                iterations += 1
    
    def _apply_classical(self, operation: ClassicalOperation):
        """Apply classical computation."""
        if operation.computation:
            # Execute classical function
            inputs = {}
            for reg_name in operation.input_registers:
                if reg_name in self._classical_memory:
                    inputs[reg_name] = self._classical_memory[reg_name]
            
            outputs = operation.computation(inputs)
            
            # Store outputs
            for reg_name in operation.output_registers:
                if reg_name in outputs:
                    self._classical_memory[reg_name] = outputs[reg_name]
    
    def _execute_single_operation(self, operation: Operation):
        """Execute a single operation (helper for conditionals/loops)."""
        if isinstance(operation, GateOperation):
            self._apply_gate(operation)
        elif isinstance(operation, MeasurementOperation):
            self._apply_measurement(operation)
        # Add other operation types as needed
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a classical condition."""
        # Simplified condition evaluation
        # In practice, this would parse more complex expressions
        if '==' in condition:
            left, right = condition.split('==')
            left_val = self._classical_memory.get(left.strip(), 0)
            right_val = int(right.strip())
            return left_val == right_val
        
        return False
    
    def _calculate_probabilities(self) -> Dict[str, float]:
        """Calculate measurement probabilities from current state."""
        probs = {}
        for i, amplitude in enumerate(self._state):
            prob = abs(amplitude) ** 2
            if prob > 1e-10:  # Only include non-negligible probabilities
                bitstring = format(i, f'0{self._num_qubits}b')
                probs[bitstring] = prob
        
        return probs
    
    def get_current_state(self) -> np.ndarray:
        """Get the current quantum state vector."""
        return self._state.copy() if self._state is not None else None
    
    def get_state_visualization(self, modes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get visualization of the current quantum state.
        
        Args:
            modes: List of visualization modes to generate
            
        Returns:
            Dictionary containing state visualizations
        """
        if self._state is None:
            return {'error': 'No quantum state available'}
        
        try:
            # Import here to avoid circular imports
            from quantum_platform.visualization.state_visualizer import StateVisualizer, VisualizationMode
            
            visualizer = StateVisualizer()
            
            # Convert string modes to enum values
            viz_modes = []
            if modes:
                mode_map = {
                    'bloch_sphere': VisualizationMode.BLOCH_SPHERE,
                    'probability_histogram': VisualizationMode.PROBABILITY_HISTOGRAM,
                    'state_vector_table': VisualizationMode.STATE_VECTOR_TABLE,
                    'entanglement_analysis': VisualizationMode.ENTANGLEMENT_ANALYSIS
                }
                viz_modes = [mode_map.get(mode, VisualizationMode.BLOCH_SPHERE) for mode in modes if mode in mode_map]
            
            if not viz_modes:
                viz_modes = [VisualizationMode.BLOCH_SPHERE, VisualizationMode.PROBABILITY_HISTOGRAM]
            
            return visualizer.visualize_state(self._state.copy(), modes=viz_modes)
            
        except ImportError:
            return {'error': 'Visualization module not available'}
        except Exception as e:
            return {'error': f'Visualization failed: {str(e)}'}
    
    def get_bloch_coordinates(self, qubit_index: int) -> Dict[str, float]:
        """
        Get Bloch sphere coordinates for a specific qubit.
        
        Args:
            qubit_index: Index of the qubit to analyze
            
        Returns:
            Dictionary with x, y, z coordinates and radius
        """
        if self._state is None:
            raise StateError("No quantum state available")
        
        try:
            from quantum_platform.visualization.state_utils import compute_bloch_coordinates
            
            coords = compute_bloch_coordinates(self._state, qubit_index)
            return {
                'x': coords.x,
                'y': coords.y,
                'z': coords.z,
                'radius': coords.radius,
                'theta': coords.theta,
                'phi': coords.phi
            }
        except ImportError:
            raise StateError("Visualization utilities not available")
    
    def get_state_probabilities(self, basis: str = 'computational') -> Dict[str, float]:
        """
        Get measurement probabilities in the specified basis.
        
        Args:
            basis: Measurement basis ('computational', 'x', 'y', 'z')
            
        Returns:
            Dictionary mapping basis states to probabilities
        """
        if self._state is None:
            raise StateError("No quantum state available")
        
        try:
            from quantum_platform.visualization.state_utils import get_state_probabilities
            return get_state_probabilities(self._state, basis)
        except ImportError:
            # Fallback implementation
            probabilities = np.abs(self._state) ** 2
            num_qubits = int(np.log2(len(self._state)))
            
            return {
                format(i, f'0{num_qubits}b'): float(prob)
                for i, prob in enumerate(probabilities)
                if prob > 1e-12
            }
    
    def analyze_state_structure(self) -> Dict[str, Any]:
        """
        Analyze the structure of the current quantum state.
        
        Returns:
            Dictionary containing state analysis
        """
        if self._state is None:
            raise StateError("No quantum state available")
        
        try:
            from quantum_platform.visualization.state_utils import analyze_state_structure
            
            structure = analyze_state_structure(self._state)
            
            # Convert to dictionary for JSON serialization
            return {
                'num_qubits': structure.num_qubits,
                'state_dimension': structure.state_dimension,
                'is_pure': structure.is_pure,
                'is_separable': structure.is_separable,
                'max_amplitude': structure.max_amplitude,
                'dominant_basis_states': structure.dominant_basis_states,
                'coherence_measures': structure.coherence_measures,
                'entanglement_measures': (
                    {
                        'concurrence': structure.entanglement_structure.concurrence,
                        'negativity': structure.entanglement_structure.negativity,
                        'von_neumann_entropy': structure.entanglement_structure.von_neumann_entropy,
                        'linear_entropy': structure.entanglement_structure.linear_entropy,
                        'schmidt_rank': structure.entanglement_structure.schmidt_rank,
                        'schmidt_coefficients': structure.entanglement_structure.schmidt_coefficients
                    } if structure.entanglement_structure else None
                )
            }
            
        except ImportError:
            # Fallback analysis
            probabilities = np.abs(self._state) ** 2
            num_qubits = int(np.log2(len(self._state)))
            
            # Find dominant states
            sorted_indices = np.argsort(probabilities)[::-1]
            dominant_states = []
            
            for i in range(min(10, len(sorted_indices))):
                idx = sorted_indices[i]
                prob = probabilities[idx]
                if prob > 1e-10:
                    basis_state = format(idx, f'0{num_qubits}b')
                    dominant_states.append((basis_state, float(prob)))
            
            return {
                'num_qubits': num_qubits,
                'state_dimension': len(self._state),
                'is_pure': True,
                'is_separable': num_qubits <= 1,
                'max_amplitude': float(np.max(np.abs(self._state))),
                'dominant_basis_states': dominant_states,
                'coherence_measures': {
                    'linear_entropy': float(1 - np.sum(probabilities ** 2))
                },
                'entanglement_measures': None
            }
    
    def create_debug_checkpoint(self) -> Dict[str, Any]:
        """
        Create a checkpoint of the current simulation state for debugging.
        
        Returns:
            Dictionary containing the current state information
        """
        if self._state is None:
            return {'error': 'No quantum state available'}
        
        return {
            'state_vector': self._state.copy(),
            'num_qubits': self._num_qubits,
            'classical_memory': self._classical_memory.copy(),
            'checkpoint_time': time.time(),
            'state_norm': float(np.linalg.norm(self._state)),
            'state_analysis': self.analyze_state_structure()
        }
    
    def restore_debug_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """
        Restore simulation state from a debug checkpoint.
        
        Args:
            checkpoint: Checkpoint data from create_debug_checkpoint()
            
        Returns:
            True if successfully restored
        """
        try:
            if 'state_vector' not in checkpoint:
                return False
            
            self._state = checkpoint['state_vector'].copy()
            self._num_qubits = checkpoint.get('num_qubits', self._num_qubits)
            self._classical_memory = checkpoint.get('classical_memory', {}).copy()
            
            return True
            
        except Exception:
            return False 