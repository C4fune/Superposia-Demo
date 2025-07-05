"""
Zero-Noise Extrapolation (ZNE) Error Mitigation

This module implements Zero-Noise Extrapolation techniques for quantum error mitigation
by scaling circuit noise levels and extrapolating to zero noise.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
from threading import Lock

from ..compiler.ir.circuit import QuantumCircuit
from ..compiler.ir.qubit import Qubit
from ..compiler.ir.operation import GateOperation
from ..compiler.gates.standard import X, Y, Z, H, S, T, RX, RY, RZ, CNOT
from ..hardware.results import AggregatedResult
from ..hardware.transpilation.transpiler import CircuitTranspiler
from ..errors import MitigationError
from ..observability.logging import get_logger


class NoiseScalingMethod(Enum):
    """Methods for scaling noise in quantum circuits."""
    GATE_FOLDING = "gate_folding"  # Fold gates G -> G†G†G
    PARAMETER_SCALING = "parameter_scaling"  # Scale rotation parameters
    IDENTITY_INSERTION = "identity_insertion"  # Insert identity operations
    RANDOM_PAULI = "random_pauli"  # Insert random Pauli pairs


class ExtrapolationMethod(Enum):
    """Methods for extrapolating to zero noise."""
    LINEAR = "linear"  # Linear extrapolation
    POLYNOMIAL = "polynomial"  # Polynomial fitting
    EXPONENTIAL = "exponential"  # Exponential fitting
    RICHARDSON = "richardson"  # Richardson extrapolation


@dataclass
class ZNEResult:
    """Result of Zero-Noise Extrapolation."""
    
    # Original and extrapolated results
    original_result: AggregatedResult
    scaled_results: List[AggregatedResult]
    extrapolated_result: AggregatedResult
    
    # ZNE parameters
    noise_factors: List[float]
    scaling_method: NoiseScalingMethod
    extrapolation_method: ExtrapolationMethod
    
    # Quality metrics
    extrapolation_error: float
    confidence_interval: Tuple[float, float]
    r_squared: float  # Goodness of fit
    
    # Execution metadata
    total_shots: int
    overhead_factor: float
    execution_time: float
    
    def get_error_reduction(self) -> float:
        """Calculate estimated error reduction."""
        if not self.original_result.counts or not self.extrapolated_result.counts:
            return 0.0
        
        # Compare dominant outcome probabilities
        original_max = max(self.original_result.probabilities.values())
        extrapolated_max = max(self.extrapolated_result.probabilities.values())
        
        # Error reduction as improvement in dominant outcome
        if original_max > 0:
            return (extrapolated_max - original_max) / original_max
        return 0.0


class ZNEMitigator:
    """Zero-Noise Extrapolation error mitigator."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._transpiler = CircuitTranspiler()
        
    def scale_noise(self, circuit: QuantumCircuit, 
                   noise_factor: float,
                   method: NoiseScalingMethod = NoiseScalingMethod.GATE_FOLDING) -> QuantumCircuit:
        """Scale noise in a quantum circuit."""
        
        if noise_factor < 1.0:
            raise ValueError(f"Noise factor must be >= 1.0, got {noise_factor}")
        
        if method == NoiseScalingMethod.GATE_FOLDING:
            return self._scale_by_gate_folding(circuit, noise_factor)
        elif method == NoiseScalingMethod.PARAMETER_SCALING:
            return self._scale_by_parameter_scaling(circuit, noise_factor)
        elif method == NoiseScalingMethod.IDENTITY_INSERTION:
            return self._scale_by_identity_insertion(circuit, noise_factor)
        elif method == NoiseScalingMethod.RANDOM_PAULI:
            return self._scale_by_random_pauli(circuit, noise_factor)
        else:
            raise ValueError(f"Unknown noise scaling method: {method}")
    
    def _scale_by_gate_folding(self, circuit: QuantumCircuit, noise_factor: float) -> QuantumCircuit:
        """Scale noise by folding gates: G -> G†G†G."""
        # Create new circuit
        scaled_circuit = QuantumCircuit(
            name=f"{circuit.name}_folded_{noise_factor}",
            num_qubits=circuit.num_qubits
        )
        
        # Copy qubits
        scaled_circuit.qubits = circuit.qubits.copy()
        
        # Determine folding strategy
        if noise_factor == 1.0:
            # No scaling needed
            scaled_circuit.operations = circuit.operations.copy()
            return scaled_circuit
        
        # For now, implement simple doubling for noise_factor = 2
        # More sophisticated folding can be added later
        fold_factor = int(noise_factor)
        
        for operation in circuit.operations:
            if isinstance(operation, GateOperation):
                # Add original gate
                scaled_circuit.add_operation(operation)
                
                # Add folded gates (inverse pairs)
                for _ in range(fold_factor - 1):
                    # Add inverse gate
                    inverse_op = self._get_inverse_gate(operation)
                    if inverse_op:
                        scaled_circuit.add_operation(inverse_op)
                        scaled_circuit.add_operation(operation)
            else:
                # Non-gate operations (measurements, etc.)
                scaled_circuit.add_operation(operation)
        
        return scaled_circuit
    
    def _scale_by_parameter_scaling(self, circuit: QuantumCircuit, noise_factor: float) -> QuantumCircuit:
        """Scale noise by scaling rotation parameters."""
        scaled_circuit = QuantumCircuit(
            name=f"{circuit.name}_param_scaled_{noise_factor}",
            num_qubits=circuit.num_qubits
        )
        
        # Copy qubits
        scaled_circuit.qubits = circuit.qubits.copy()
        
        for operation in circuit.operations:
            if isinstance(operation, GateOperation) and self._is_rotation_gate(operation):
                # Scale rotation parameters
                scaled_op = self._scale_rotation_parameters(operation, noise_factor)
                scaled_circuit.add_operation(scaled_op)
            else:
                # Non-rotation gates remain unchanged
                scaled_circuit.add_operation(operation)
        
        return scaled_circuit
    
    def _scale_by_identity_insertion(self, circuit: QuantumCircuit, noise_factor: float) -> QuantumCircuit:
        """Scale noise by inserting identity operations."""
        scaled_circuit = QuantumCircuit(
            name=f"{circuit.name}_identity_scaled_{noise_factor}",
            num_qubits=circuit.num_qubits
        )
        
        # Copy qubits
        scaled_circuit.qubits = circuit.qubits.copy()
        
        # Calculate number of identities to insert
        num_identities = int((noise_factor - 1.0) * len(circuit.operations))
        
        for i, operation in enumerate(circuit.operations):
            # Add original operation
            scaled_circuit.add_operation(operation)
            
            # Insert identity operations
            if i < num_identities:
                # Create identity as X†X or Z†Z
                if isinstance(operation, GateOperation) and operation.targets:
                    target = operation.targets[0]
                    # Add X gate followed by X gate (identity)
                    x_op = GateOperation(name="X", targets=[target])
                    scaled_circuit.add_operation(x_op)
                    scaled_circuit.add_operation(x_op)
        
        return scaled_circuit
    
    def _scale_by_random_pauli(self, circuit: QuantumCircuit, noise_factor: float) -> QuantumCircuit:
        """Scale noise by inserting random Pauli pairs."""
        scaled_circuit = QuantumCircuit(
            name=f"{circuit.name}_pauli_scaled_{noise_factor}",
            num_qubits=circuit.num_qubits
        )
        
        # Copy qubits
        scaled_circuit.qubits = circuit.qubits.copy()
        
        # Random Pauli gates
        pauli_gates = ['X', 'Y', 'Z']
        
        # Calculate number of Pauli pairs to insert
        num_pairs = int((noise_factor - 1.0) * len(circuit.operations))
        
        for i, operation in enumerate(circuit.operations):
            # Add original operation
            scaled_circuit.add_operation(operation)
            
            # Insert random Pauli pairs
            if i < num_pairs and isinstance(operation, GateOperation) and operation.targets:
                target = operation.targets[0]
                # Choose random Pauli gate
                pauli_name = np.random.choice(pauli_gates)
                
                # Add Pauli gate twice (identity)
                pauli_op = GateOperation(name=pauli_name, targets=[target])
                scaled_circuit.add_operation(pauli_op)
                scaled_circuit.add_operation(pauli_op)
        
        return scaled_circuit
    
    def _get_inverse_gate(self, gate_op: GateOperation) -> Optional[GateOperation]:
        """Get the inverse of a gate operation."""
        gate_name = gate_op.name.upper()
        
        # Self-inverse gates
        if gate_name in ['X', 'Y', 'Z', 'H', 'CNOT', 'CX']:
            return GateOperation(
                name=gate_name,
                targets=gate_op.targets.copy(),
                controls=gate_op.controls.copy()
            )
        
        # S† = S‡
        if gate_name == 'S':
            return GateOperation(
                name='SDAGGER',
                targets=gate_op.targets.copy(),
                controls=gate_op.controls.copy()
            )
        
        # T† = T‡  
        if gate_name == 'T':
            return GateOperation(
                name='TDAGGER',
                targets=gate_op.targets.copy(),
                controls=gate_op.controls.copy()
            )
        
        # Rotation gates: R(-θ)
        if gate_name in ['RX', 'RY', 'RZ'] and gate_op.parameters:
            inverse_params = {}
            for param_name, param_value in gate_op.parameters.items():
                inverse_params[param_name] = -param_value
            
            return GateOperation(
                name=gate_name,
                targets=gate_op.targets.copy(),
                controls=gate_op.controls.copy(),
                parameters=inverse_params
            )
        
        # Default: return None for unknown gates
        return None
    
    def _is_rotation_gate(self, gate_op: GateOperation) -> bool:
        """Check if a gate is a rotation gate."""
        return gate_op.name.upper() in ['RX', 'RY', 'RZ'] and gate_op.parameters
    
    def _scale_rotation_parameters(self, gate_op: GateOperation, noise_factor: float) -> GateOperation:
        """Scale rotation gate parameters."""
        scaled_params = {}
        for param_name, param_value in gate_op.parameters.items():
            scaled_params[param_name] = param_value * noise_factor
        
        return GateOperation(
            name=gate_op.name,
            targets=gate_op.targets.copy(),
            controls=gate_op.controls.copy(),
            parameters=scaled_params
        )
    
    def extrapolate_to_zero_noise(self, results: List[AggregatedResult],
                                  noise_factors: List[float],
                                  method: ExtrapolationMethod = ExtrapolationMethod.LINEAR) -> Tuple[AggregatedResult, float, float]:
        """Extrapolate measurement results to zero noise."""
        
        if len(results) != len(noise_factors):
            raise ValueError("Number of results must match number of noise factors")
        
        if len(results) < 2:
            raise ValueError("At least 2 results needed for extrapolation")
        
        # Extract expectation values or probabilities for extrapolation
        if method == ExtrapolationMethod.LINEAR:
            return self._linear_extrapolation(results, noise_factors)
        elif method == ExtrapolationMethod.POLYNOMIAL:
            return self._polynomial_extrapolation(results, noise_factors)
        elif method == ExtrapolationMethod.EXPONENTIAL:
            return self._exponential_extrapolation(results, noise_factors)
        elif method == ExtrapolationMethod.RICHARDSON:
            return self._richardson_extrapolation(results, noise_factors)
        else:
            raise ValueError(f"Unknown extrapolation method: {method}")
    
    def _linear_extrapolation(self, results: List[AggregatedResult], 
                             noise_factors: List[float]) -> Tuple[AggregatedResult, float, float]:
        """Linear extrapolation to zero noise."""
        
        # Extract dominant outcome probabilities for extrapolation
        probabilities = []
        for result in results:
            if result.probabilities:
                # Use probability of most likely outcome
                max_prob = max(result.probabilities.values())
                probabilities.append(max_prob)
            else:
                probabilities.append(0.0)
        
        # Fit linear model: y = a + b*x
        X = np.array(noise_factors).reshape(-1, 1)
        y = np.array(probabilities)
        
        # Add constant term
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # Least squares fit
        try:
            coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            a, b = coeffs
            
            # Extrapolate to zero noise
            zero_noise_prob = a  # y(0) = a
            
            # Calculate R-squared
            y_pred = X_with_const @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Create extrapolated result
            extrapolated_result = self._create_extrapolated_result(
                results[0], zero_noise_prob, "linear_extrapolation"
            )
            
            return extrapolated_result, abs(b), r_squared
            
        except np.linalg.LinAlgError:
            # Fallback to original result
            return results[0], 0.0, 0.0
    
    def _polynomial_extrapolation(self, results: List[AggregatedResult], 
                                 noise_factors: List[float]) -> Tuple[AggregatedResult, float, float]:
        """Polynomial extrapolation to zero noise."""
        
        # Extract probabilities
        probabilities = []
        for result in results:
            if result.probabilities:
                max_prob = max(result.probabilities.values())
                probabilities.append(max_prob)
            else:
                probabilities.append(0.0)
        
        # Fit polynomial (degree 2)
        try:
            coeffs = np.polyfit(noise_factors, probabilities, deg=2)
            
            # Extrapolate to zero noise
            zero_noise_prob = np.polyval(coeffs, 0.0)
            
            # Calculate R-squared
            y_pred = np.polyval(coeffs, noise_factors)
            ss_res = np.sum((np.array(probabilities) - y_pred) ** 2)
            ss_tot = np.sum((np.array(probabilities) - np.mean(probabilities)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Create extrapolated result
            extrapolated_result = self._create_extrapolated_result(
                results[0], zero_noise_prob, "polynomial_extrapolation"
            )
            
            return extrapolated_result, abs(coeffs[1]), r_squared
            
        except (np.linalg.LinAlgError, np.RankWarning):
            # Fallback to linear
            return self._linear_extrapolation(results, noise_factors)
    
    def _exponential_extrapolation(self, results: List[AggregatedResult], 
                                  noise_factors: List[float]) -> Tuple[AggregatedResult, float, float]:
        """Exponential extrapolation to zero noise."""
        
        # Extract probabilities
        probabilities = []
        for result in results:
            if result.probabilities:
                max_prob = max(result.probabilities.values())
                probabilities.append(max_prob)
            else:
                probabilities.append(0.0)
        
        # Fit exponential model: y = a * exp(b * x)
        try:
            # Take log to linearize
            log_probs = np.log(np.maximum(probabilities, 1e-10))
            coeffs = np.polyfit(noise_factors, log_probs, deg=1)
            
            # Extrapolate to zero noise
            zero_noise_prob = np.exp(coeffs[1])  # exp(b * 0 + a) = exp(a)
            
            # Calculate R-squared in log space
            y_pred = np.polyval(coeffs, noise_factors)
            ss_res = np.sum((log_probs - y_pred) ** 2)
            ss_tot = np.sum((log_probs - np.mean(log_probs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Create extrapolated result
            extrapolated_result = self._create_extrapolated_result(
                results[0], zero_noise_prob, "exponential_extrapolation"
            )
            
            return extrapolated_result, abs(coeffs[0]), r_squared
            
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to linear
            return self._linear_extrapolation(results, noise_factors)
    
    def _richardson_extrapolation(self, results: List[AggregatedResult], 
                                 noise_factors: List[float]) -> Tuple[AggregatedResult, float, float]:
        """Richardson extrapolation to zero noise."""
        
        # Richardson extrapolation requires specific noise scaling
        # For now, implement simple Richardson with 2 points
        if len(results) < 2:
            return results[0], 0.0, 0.0
        
        # Extract probabilities
        prob1 = max(results[0].probabilities.values()) if results[0].probabilities else 0.0
        prob2 = max(results[1].probabilities.values()) if results[1].probabilities else 0.0
        
        # Richardson formula: f(0) ≈ 2*f(h) - f(2h)
        # Assuming noise_factors[1] = 2 * noise_factors[0]
        zero_noise_prob = 2 * prob1 - prob2
        
        # Clamp to valid probability range
        zero_noise_prob = np.clip(zero_noise_prob, 0.0, 1.0)
        
        # Create extrapolated result
        extrapolated_result = self._create_extrapolated_result(
            results[0], zero_noise_prob, "richardson_extrapolation"
        )
        
        return extrapolated_result, abs(prob1 - prob2), 1.0
    
    def _create_extrapolated_result(self, base_result: AggregatedResult, 
                                   zero_noise_prob: float, 
                                   method: str) -> AggregatedResult:
        """Create extrapolated result from base result."""
        
        # Find the most likely outcome
        if not base_result.probabilities:
            return base_result
        
        dominant_outcome = max(base_result.probabilities.keys(), 
                             key=lambda k: base_result.probabilities[k])
        
        # Create new counts with extrapolated probability
        total_shots = base_result.total_shots
        new_counts = {}
        
        # Set dominant outcome to extrapolated probability
        new_counts[dominant_outcome] = int(zero_noise_prob * total_shots)
        
        # Distribute remaining shots proportionally
        remaining_shots = total_shots - new_counts[dominant_outcome]
        remaining_prob = 1.0 - zero_noise_prob
        
        for outcome, original_count in base_result.counts.items():
            if outcome != dominant_outcome:
                original_prob = original_count / total_shots
                if remaining_prob > 0:
                    new_prob = original_prob / remaining_prob * (1.0 - zero_noise_prob)
                    new_counts[outcome] = int(new_prob * total_shots)
                else:
                    new_counts[outcome] = 0
        
        # Create new result
        return AggregatedResult(
            counts=new_counts,
            total_shots=total_shots,
            successful_shots=total_shots,
            backend_name=base_result.backend_name,
            metadata={
                **base_result.metadata,
                'extrapolation_method': method,
                'zero_noise_probability': zero_noise_prob
            }
        )
    
    def apply_zne(self, circuit: QuantumCircuit,
                  noise_factors: List[float],
                  execution_func: Callable[[QuantumCircuit], AggregatedResult],
                  scaling_method: NoiseScalingMethod = NoiseScalingMethod.GATE_FOLDING,
                  extrapolation_method: ExtrapolationMethod = ExtrapolationMethod.LINEAR) -> ZNEResult:
        """Apply Zero-Noise Extrapolation to a quantum circuit."""
        
        start_time = time.time()
        
        # Execute original circuit
        original_result = execution_func(circuit)
        
        # Generate noise-scaled circuits and execute them
        scaled_results = []
        for noise_factor in noise_factors:
            if noise_factor == 1.0:
                # Use original result
                scaled_results.append(original_result)
            else:
                # Scale noise and execute
                scaled_circuit = self.scale_noise(circuit, noise_factor, scaling_method)
                scaled_result = execution_func(scaled_circuit)
                scaled_results.append(scaled_result)
        
        # Extrapolate to zero noise
        extrapolated_result, extrapolation_error, r_squared = self.extrapolate_to_zero_noise(
            scaled_results, noise_factors, extrapolation_method
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Calculate overhead factor
        overhead_factor = len(noise_factors)
        
        return ZNEResult(
            original_result=original_result,
            scaled_results=scaled_results,
            extrapolated_result=extrapolated_result,
            noise_factors=noise_factors,
            scaling_method=scaling_method,
            extrapolation_method=extrapolation_method,
            extrapolation_error=extrapolation_error,
            confidence_interval=(0.0, 1.0),  # Placeholder
            r_squared=r_squared,
            total_shots=sum(r.total_shots for r in scaled_results),
            overhead_factor=overhead_factor,
            execution_time=execution_time
        )


# Global instance
_zne_mitigator = None
_zne_lock = Lock()


def get_zne_mitigator() -> ZNEMitigator:
    """Get global ZNE mitigator instance."""
    global _zne_mitigator
    if _zne_mitigator is None:
        with _zne_lock:
            if _zne_mitigator is None:
                _zne_mitigator = ZNEMitigator()
    return _zne_mitigator


def apply_zne_mitigation(circuit: QuantumCircuit,
                        execution_func: Callable[[QuantumCircuit], AggregatedResult],
                        noise_factors: Optional[List[float]] = None,
                        scaling_method: NoiseScalingMethod = NoiseScalingMethod.GATE_FOLDING,
                        extrapolation_method: ExtrapolationMethod = ExtrapolationMethod.LINEAR) -> ZNEResult:
    """Apply Zero-Noise Extrapolation to a circuit."""
    
    # Default noise factors
    if noise_factors is None:
        noise_factors = [1.0, 2.0, 3.0]
    
    mitigator = get_zne_mitigator()
    return mitigator.apply_zne(
        circuit, noise_factors, execution_func,
        scaling_method, extrapolation_method
    ) 