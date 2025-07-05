"""
Measurement Error Mitigation

This module provides measurement error mitigation capabilities using calibration
matrices to correct readout errors in quantum measurements.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import json
from threading import Lock

from ..compiler.ir.circuit import QuantumCircuit
from ..compiler.ir.qubit import Qubit
from ..compiler.language.dsl import QuantumProgram
from ..compiler.language.operations import H, X
from ..hardware.results import AggregatedResult, ShotResult
from ..hardware.hal import QuantumHardwareBackend
from ..errors import MitigationError
from ..observability.logging import get_logger


@dataclass
class CalibrationMatrix:
    """Calibration matrix for measurement error mitigation."""
    
    # The confusion matrix M where M[i,j] = P(measure i | prepared j)
    matrix: np.ndarray
    
    # Inverse matrix for correction
    inverse_matrix: np.ndarray
    
    # Metadata
    num_qubits: int
    backend_name: str
    created_at: datetime
    calibration_shots: int
    
    # Quality metrics
    readout_fidelity: Dict[int, float]  # Per-qubit readout fidelity
    crosstalk_matrix: Optional[np.ndarray] = None  # Cross-talk between qubits
    
    def __post_init__(self):
        """Validate calibration matrix."""
        if self.matrix.shape != (2**self.num_qubits, 2**self.num_qubits):
            raise ValueError(f"Invalid matrix shape: {self.matrix.shape}")
        
        # Check if matrix is invertible
        try:
            if self.inverse_matrix is None:
                self.inverse_matrix = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError:
            raise MitigationError("Calibration matrix is singular and cannot be inverted")
    
    def is_valid(self, max_age_hours: float = 24.0) -> bool:
        """Check if calibration is still valid."""
        age = datetime.now() - self.created_at
        return age < timedelta(hours=max_age_hours)
    
    def get_correction_factor(self, bitstring: str) -> float:
        """Get correction factor for a specific bitstring."""
        if len(bitstring) != self.num_qubits:
            raise ValueError(f"Bitstring length {len(bitstring)} != {self.num_qubits}")
        
        # Convert bitstring to index
        index = int(bitstring, 2)
        
        # Get diagonal element of inverse matrix (correction factor)
        return self.inverse_matrix[index, index]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'matrix': self.matrix.tolist(),
            'inverse_matrix': self.inverse_matrix.tolist(),
            'num_qubits': self.num_qubits,
            'backend_name': self.backend_name,
            'created_at': self.created_at.isoformat(),
            'calibration_shots': self.calibration_shots,
            'readout_fidelity': self.readout_fidelity,
            'crosstalk_matrix': self.crosstalk_matrix.tolist() if self.crosstalk_matrix is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationMatrix':
        """Create from dictionary."""
        return cls(
            matrix=np.array(data['matrix']),
            inverse_matrix=np.array(data['inverse_matrix']),
            num_qubits=data['num_qubits'],
            backend_name=data['backend_name'],
            created_at=datetime.fromisoformat(data['created_at']),
            calibration_shots=data['calibration_shots'],
            readout_fidelity=data['readout_fidelity'],
            crosstalk_matrix=np.array(data['crosstalk_matrix']) if data['crosstalk_matrix'] else None
        )


@dataclass
class MitigationResult:
    """Result of measurement error mitigation."""
    
    # Original and mitigated results
    original_counts: Dict[str, int]
    mitigated_counts: Dict[str, int]
    
    # Mitigation metadata
    calibration_matrix: CalibrationMatrix
    mitigation_method: str
    total_shots: int
    
    # Quality metrics
    mitigation_overhead: float  # Time overhead
    fidelity_improvement: float  # Estimated improvement
    
    def get_mitigation_factor(self) -> float:
        """Calculate overall mitigation factor."""
        if not self.original_counts or not self.mitigated_counts:
            return 1.0
        
        # Compare entropy before and after mitigation
        original_probs = np.array(list(self.original_counts.values())) / self.total_shots
        mitigated_probs = np.array(list(self.mitigated_counts.values())) / self.total_shots
        
        # Remove zero probabilities for entropy calculation
        original_probs = original_probs[original_probs > 0]
        mitigated_probs = mitigated_probs[mitigated_probs > 0]
        
        original_entropy = -np.sum(original_probs * np.log2(original_probs))
        mitigated_entropy = -np.sum(mitigated_probs * np.log2(mitigated_probs))
        
        # Return ratio (higher entropy often indicates better mitigation)
        return mitigated_entropy / original_entropy if original_entropy > 0 else 1.0


class MeasurementMitigator:
    """Measurement error mitigation using calibration matrices."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._calibration_cache: Dict[str, CalibrationMatrix] = {}
        self._cache_lock = Lock()
        
    def generate_calibration_circuits(self, num_qubits: int) -> List[QuantumCircuit]:
        """Generate calibration circuits for measurement error characterization."""
        circuits = []
        
        # Generate all possible computational basis states
        for state in range(2**num_qubits):
            # Create circuit to prepare specific state
            with QuantumProgram() as qp:
                qubits = qp.allocate(num_qubits)
                
                # Apply X gates to prepare desired state
                for i in range(num_qubits):
                    if (state >> i) & 1:
                        X(qubits[i])
                
                # Measure all qubits
                qp.measure(qubits, f"calibration_state_{state:0{num_qubits}b}")
            
            qp.circuit.name = f"calibration_state_{state:0{num_qubits}b}"
            circuits.append(qp.circuit)
        
        return circuits
    
    def build_calibration_matrix(self, calibration_results: List[AggregatedResult]) -> CalibrationMatrix:
        """Build calibration matrix from measurement results."""
        if not calibration_results:
            raise ValueError("No calibration results provided")
        
        # Determine number of qubits from first result
        first_result = calibration_results[0]
        if not first_result.counts:
            raise ValueError("No measurement counts in calibration results")
        
        # Get number of qubits from bitstring length
        num_qubits = len(list(first_result.counts.keys())[0])
        matrix_size = 2**num_qubits
        
        # Initialize confusion matrix
        confusion_matrix = np.zeros((matrix_size, matrix_size))
        
        # Process each calibration result
        for prepared_state, result in enumerate(calibration_results):
            if prepared_state >= matrix_size:
                break
                
            total_shots = result.total_shots
            if total_shots == 0:
                continue
            
            # Fill confusion matrix row for this prepared state
            for measured_bitstring, count in result.counts.items():
                measured_state = int(measured_bitstring, 2)
                confusion_matrix[measured_state, prepared_state] = count / total_shots
        
        # Calculate readout fidelity for each qubit
        readout_fidelity = {}
        for qubit_idx in range(num_qubits):
            # Calculate single-qubit fidelity
            fidelity = 0.0
            for state in range(2):
                # Find results where this qubit should be in 'state'
                correct_measurements = 0
                total_measurements = 0
                
                for prepared_state in range(matrix_size):
                    if (prepared_state >> qubit_idx) & 1 == state:
                        result = calibration_results[prepared_state]
                        for measured_bitstring, count in result.counts.items():
                            total_measurements += count
                            if (int(measured_bitstring, 2) >> qubit_idx) & 1 == state:
                                correct_measurements += count
                
                if total_measurements > 0:
                    fidelity += (correct_measurements / total_measurements) * 0.5
            
            readout_fidelity[qubit_idx] = fidelity
        
        # Create calibration matrix
        calibration_matrix = CalibrationMatrix(
            matrix=confusion_matrix,
            inverse_matrix=np.linalg.inv(confusion_matrix),
            num_qubits=num_qubits,
            backend_name=first_result.backend_name if hasattr(first_result, 'backend_name') else "unknown",
            created_at=datetime.now(),
            calibration_shots=sum(r.total_shots for r in calibration_results),
            readout_fidelity=readout_fidelity
        )
        
        return calibration_matrix
    
    def apply_mitigation(self, result: AggregatedResult, 
                        calibration_matrix: CalibrationMatrix) -> MitigationResult:
        """Apply measurement error mitigation to results."""
        start_time = time.time()
        
        # Validate compatibility
        if not result.counts:
            raise ValueError("No measurement counts to mitigate")
        
        bitstring_length = len(list(result.counts.keys())[0])
        if bitstring_length != calibration_matrix.num_qubits:
            raise ValueError(f"Qubit count mismatch: result={bitstring_length}, calibration={calibration_matrix.num_qubits}")
        
        # Convert counts to probability vector
        total_shots = result.total_shots
        prob_vector = np.zeros(2**calibration_matrix.num_qubits)
        
        for bitstring, count in result.counts.items():
            index = int(bitstring, 2)
            prob_vector[index] = count / total_shots
        
        # Apply inverse calibration matrix
        mitigated_probs = calibration_matrix.inverse_matrix @ prob_vector
        
        # Ensure non-negative probabilities (clamp to 0)
        mitigated_probs = np.maximum(mitigated_probs, 0)
        
        # Renormalize to ensure probabilities sum to 1
        prob_sum = np.sum(mitigated_probs)
        if prob_sum > 0:
            mitigated_probs /= prob_sum
        
        # Convert back to counts
        mitigated_counts = {}
        for i, prob in enumerate(mitigated_probs):
            if prob > 0:
                bitstring = format(i, f'0{calibration_matrix.num_qubits}b')
                mitigated_counts[bitstring] = int(prob * total_shots)
        
        # Calculate quality metrics
        mitigation_overhead = time.time() - start_time
        
        # Estimate fidelity improvement
        fidelity_improvement = self._estimate_fidelity_improvement(
            result.counts, mitigated_counts, calibration_matrix
        )
        
        return MitigationResult(
            original_counts=result.counts.copy(),
            mitigated_counts=mitigated_counts,
            calibration_matrix=calibration_matrix,
            mitigation_method="calibration_matrix",
            total_shots=total_shots,
            mitigation_overhead=mitigation_overhead,
            fidelity_improvement=fidelity_improvement
        )
    
    def _estimate_fidelity_improvement(self, original_counts: Dict[str, int],
                                     mitigated_counts: Dict[str, int],
                                     calibration_matrix: CalibrationMatrix) -> float:
        """Estimate fidelity improvement from mitigation."""
        try:
            # Use average readout fidelity as a proxy
            avg_fidelity = np.mean(list(calibration_matrix.readout_fidelity.values()))
            
            # Simple heuristic: improvement is proportional to error rate
            error_rate = 1.0 - avg_fidelity
            improvement = error_rate * 0.5  # Assume 50% error reduction
            
            return improvement
        except Exception:
            return 0.0
    
    def cache_calibration(self, backend_name: str, num_qubits: int, 
                         calibration_matrix: CalibrationMatrix):
        """Cache calibration matrix."""
        cache_key = f"{backend_name}_{num_qubits}"
        with self._cache_lock:
            self._calibration_cache[cache_key] = calibration_matrix
        
        self.logger.info(f"Cached calibration matrix for {cache_key}")
    
    def get_cached_calibration(self, backend_name: str, num_qubits: int) -> Optional[CalibrationMatrix]:
        """Get cached calibration matrix if valid."""
        cache_key = f"{backend_name}_{num_qubits}"
        with self._cache_lock:
            if cache_key in self._calibration_cache:
                calibration = self._calibration_cache[cache_key]
                if calibration.is_valid():
                    return calibration
                else:
                    # Remove expired calibration
                    del self._calibration_cache[cache_key]
        
        return None


# Global instance
_measurement_mitigator = None
_mitigator_lock = Lock()


def get_measurement_mitigator() -> MeasurementMitigator:
    """Get global measurement mitigator instance."""
    global _measurement_mitigator
    if _measurement_mitigator is None:
        with _mitigator_lock:
            if _measurement_mitigator is None:
                _measurement_mitigator = MeasurementMitigator()
    return _measurement_mitigator


def perform_measurement_calibration(backend: QuantumHardwareBackend,
                                   num_qubits: int,
                                   shots: int = 1000) -> CalibrationMatrix:
    """Perform measurement calibration for a backend."""
    mitigator = get_measurement_mitigator()
    
    # Generate calibration circuits
    calibration_circuits = mitigator.generate_calibration_circuits(num_qubits)
    
    # Execute calibration circuits
    calibration_results = []
    for circuit in calibration_circuits:
        try:
            # Submit and wait for results
            result = backend.submit_and_wait(circuit, shots)
            
            # Convert to AggregatedResult
            aggregated_result = AggregatedResult(
                counts=result.counts,
                total_shots=shots,
                successful_shots=shots,
                backend_name=backend.name
            )
            calibration_results.append(aggregated_result)
            
        except Exception as e:
            raise MitigationError(f"Calibration circuit execution failed: {e}")
    
    # Build calibration matrix
    calibration_matrix = mitigator.build_calibration_matrix(calibration_results)
    
    # Cache the calibration
    mitigator.cache_calibration(backend.name, num_qubits, calibration_matrix)
    
    return calibration_matrix


def apply_measurement_mitigation(result: AggregatedResult,
                                backend: QuantumHardwareBackend,
                                calibration_matrix: Optional[CalibrationMatrix] = None) -> MitigationResult:
    """Apply measurement mitigation to results."""
    mitigator = get_measurement_mitigator()
    
    # Get calibration matrix if not provided
    if calibration_matrix is None:
        num_qubits = len(list(result.counts.keys())[0]) if result.counts else 0
        calibration_matrix = mitigator.get_cached_calibration(backend.name, num_qubits)
        
        if calibration_matrix is None:
            raise MitigationError(f"No calibration matrix available for {backend.name} with {num_qubits} qubits")
    
    # Apply mitigation
    return mitigator.apply_mitigation(result, calibration_matrix) 