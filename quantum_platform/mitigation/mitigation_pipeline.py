"""
Error Mitigation Pipeline

This module provides a comprehensive pipeline for applying multiple error
mitigation techniques in a coordinated manner.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..compiler.ir.circuit import QuantumCircuit
from ..hardware.hal import QuantumHardwareBackend
from ..hardware.results import AggregatedResult
from ..errors import MitigationError
from ..observability.logging import get_logger

from .measurement_mitigation import (
    MeasurementMitigator, MitigationResult, CalibrationMatrix,
    get_measurement_mitigator, apply_measurement_mitigation
)
from .zero_noise_extrapolation import (
    ZNEMitigator, ZNEResult, NoiseScalingMethod, ExtrapolationMethod,
    get_zne_mitigator, apply_zne_mitigation
)
from .error_correction import (
    ErrorCorrectionCode, ErrorCorrectionResult,
    get_error_correction_code, encode_circuit
)
from .calibration_manager import (
    CalibrationManager, CalibrationData,
    get_calibration_manager, refresh_calibration
)


class MitigationLevel(Enum):
    """Level of error mitigation to apply."""
    NONE = "none"
    BASIC = "basic"              # Measurement mitigation only
    MODERATE = "moderate"        # Measurement + ZNE
    AGGRESSIVE = "aggressive"    # All techniques
    CUSTOM = "custom"            # Custom configuration


@dataclass
class MitigationOptions:
    """Options for error mitigation pipeline."""
    
    # Overall mitigation level
    level: MitigationLevel = MitigationLevel.BASIC
    
    # Measurement error mitigation
    enable_measurement_mitigation: bool = True
    auto_calibration: bool = True
    calibration_shots: int = 1000
    max_calibration_age_hours: float = 24.0
    
    # Zero-noise extrapolation
    enable_zne: bool = False
    zne_noise_factors: List[float] = None
    zne_scaling_method: NoiseScalingMethod = NoiseScalingMethod.GATE_FOLDING
    zne_extrapolation_method: ExtrapolationMethod = ExtrapolationMethod.LINEAR
    
    # Error correction
    enable_error_correction: bool = False
    error_correction_code: str = "bit_flip"
    
    # Pipeline options
    parallel_execution: bool = True
    cache_results: bool = True
    quality_threshold: float = 0.8
    
    def __post_init__(self):
        """Set default values based on mitigation level."""
        if self.level == MitigationLevel.NONE:
            self.enable_measurement_mitigation = False
            self.enable_zne = False
            self.enable_error_correction = False
            
        elif self.level == MitigationLevel.BASIC:
            self.enable_measurement_mitigation = True
            self.enable_zne = False
            self.enable_error_correction = False
            
        elif self.level == MitigationLevel.MODERATE:
            self.enable_measurement_mitigation = True
            self.enable_zne = True
            self.enable_error_correction = False
            
        elif self.level == MitigationLevel.AGGRESSIVE:
            self.enable_measurement_mitigation = True
            self.enable_zne = True
            self.enable_error_correction = True
        
        # Set default ZNE noise factors
        if self.zne_noise_factors is None:
            self.zne_noise_factors = [1.0, 2.0, 3.0]


@dataclass
class MitigationPipelineResult:
    """Result of error mitigation pipeline."""
    
    # Original and final results
    original_result: AggregatedResult
    mitigated_result: AggregatedResult
    
    # Individual mitigation results
    measurement_mitigation_result: Optional[MitigationResult] = None
    zne_result: Optional[ZNEResult] = None
    error_correction_result: Optional[ErrorCorrectionResult] = None
    
    # Pipeline metadata
    options: MitigationOptions = None
    total_execution_time: float = 0.0
    total_shots: int = 0
    overhead_factor: float = 1.0
    
    # Quality metrics
    fidelity_improvement: float = 0.0
    confidence_score: float = 0.0
    
    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of improvements from mitigation."""
        summary = {
            'original_dominant_probability': 0.0,
            'mitigated_dominant_probability': 0.0,
            'improvement_factor': 1.0,
            'applied_techniques': []
        }
        
        if self.original_result.probabilities:
            summary['original_dominant_probability'] = max(self.original_result.probabilities.values())
        
        if self.mitigated_result.probabilities:
            summary['mitigated_dominant_probability'] = max(self.mitigated_result.probabilities.values())
        
        if summary['original_dominant_probability'] > 0:
            summary['improvement_factor'] = (
                summary['mitigated_dominant_probability'] / summary['original_dominant_probability']
            )
        
        # List applied techniques
        if self.measurement_mitigation_result:
            summary['applied_techniques'].append('measurement_mitigation')
        if self.zne_result:
            summary['applied_techniques'].append('zero_noise_extrapolation')
        if self.error_correction_result:
            summary['applied_techniques'].append('error_correction')
        
        return summary


class MitigationPipeline:
    """Comprehensive error mitigation pipeline."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Mitigation components
        self.measurement_mitigator = get_measurement_mitigator()
        self.zne_mitigator = get_zne_mitigator()
        self.calibration_manager = get_calibration_manager()
        
        # Results cache
        self._results_cache: Dict[str, MitigationPipelineResult] = {}
        self._cache_lock = threading.Lock()
    
    def apply_mitigation(self, circuit: QuantumCircuit, backend: QuantumHardwareBackend,
                        shots: int = 1000, options: Optional[MitigationOptions] = None) -> MitigationPipelineResult:
        """Apply error mitigation pipeline to a quantum circuit."""
        
        start_time = time.time()
        
        # Use default options if not provided
        if options is None:
            options = MitigationOptions()
        
        # Check cache if enabled
        if options.cache_results:
            cache_key = self._get_cache_key(circuit, backend, shots, options)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
        
        try:
            # Step 1: Apply error correction encoding (if enabled)
            working_circuit = circuit
            error_correction_result = None
            
            if options.enable_error_correction:
                try:
                    error_correction_result = self._apply_error_correction(working_circuit, options)
                    working_circuit = error_correction_result.processed_circuit
                except Exception as e:
                    self.logger.warning(f"Error correction failed: {e}")
            
            # Step 2: Execute circuit (potentially with ZNE)
            if options.enable_zne:
                # Apply ZNE mitigation
                zne_result = self._apply_zne_mitigation(working_circuit, backend, shots, options)
                execution_result = zne_result.extrapolated_result
                total_shots = zne_result.total_shots
            else:
                # Standard execution
                execution_result = self._execute_circuit(working_circuit, backend, shots)
                zne_result = None
                total_shots = shots
            
            # Step 3: Apply measurement error mitigation (if enabled)
            measurement_mitigation_result = None
            final_result = execution_result
            
            if options.enable_measurement_mitigation:
                try:
                    measurement_mitigation_result = self._apply_measurement_mitigation(
                        execution_result, backend, options
                    )
                    
                    # Use mitigated counts for final result
                    final_result = AggregatedResult(
                        counts=measurement_mitigation_result.mitigated_counts,
                        total_shots=execution_result.total_shots,
                        successful_shots=execution_result.successful_shots,
                        backend_name=backend.name,
                        metadata={
                            **execution_result.metadata,
                            'measurement_mitigation_applied': True
                        }
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Measurement mitigation failed: {e}")
            
            # Calculate metrics
            execution_time = time.time() - start_time
            overhead_factor = self._calculate_overhead_factor(options, total_shots, shots)
            fidelity_improvement = self._estimate_fidelity_improvement(
                execution_result, final_result, options
            )
            confidence_score = self._calculate_confidence_score(
                measurement_mitigation_result, zne_result, options
            )
            
            # Create pipeline result
            pipeline_result = MitigationPipelineResult(
                original_result=execution_result,
                mitigated_result=final_result,
                measurement_mitigation_result=measurement_mitigation_result,
                zne_result=zne_result,
                error_correction_result=error_correction_result,
                options=options,
                total_execution_time=execution_time,
                total_shots=total_shots,
                overhead_factor=overhead_factor,
                fidelity_improvement=fidelity_improvement,
                confidence_score=confidence_score
            )
            
            # Cache result if enabled
            if options.cache_results:
                self._cache_result(cache_key, pipeline_result)
            
            return pipeline_result
            
        except Exception as e:
            raise MitigationError(f"Mitigation pipeline failed: {e}")
    
    def _apply_error_correction(self, circuit: QuantumCircuit, 
                              options: MitigationOptions) -> ErrorCorrectionResult:
        """Apply error correction encoding."""
        code = get_error_correction_code(options.error_correction_code)
        return code.encode(circuit)
    
    def _apply_zne_mitigation(self, circuit: QuantumCircuit, backend: QuantumHardwareBackend,
                            shots: int, options: MitigationOptions) -> ZNEResult:
        """Apply zero-noise extrapolation."""
        
        def execution_func(circ: QuantumCircuit) -> AggregatedResult:
            result = self._execute_circuit(circ, backend, shots)
            return result
        
        return apply_zne_mitigation(
            circuit=circuit,
            execution_func=execution_func,
            noise_factors=options.zne_noise_factors,
            scaling_method=options.zne_scaling_method,
            extrapolation_method=options.zne_extrapolation_method
        )
    
    def _apply_measurement_mitigation(self, result: AggregatedResult, 
                                    backend: QuantumHardwareBackend,
                                    options: MitigationOptions) -> MitigationResult:
        """Apply measurement error mitigation."""
        
        # Get number of qubits
        num_qubits = len(list(result.counts.keys())[0]) if result.counts else 0
        
        # Get or create calibration
        calibration_matrix = self._get_calibration_matrix(backend, num_qubits, options)
        
        # Apply mitigation
        return self.measurement_mitigator.apply_mitigation(result, calibration_matrix)
    
    def _get_calibration_matrix(self, backend: QuantumHardwareBackend, 
                               num_qubits: int, options: MitigationOptions) -> CalibrationMatrix:
        """Get calibration matrix for measurement mitigation."""
        
        device_id = getattr(backend, 'device_name', backend.name)
        
        # Try to get cached calibration
        calibration_data = self.calibration_manager.get_calibration(
            backend.name, device_id, num_qubits, "measurement_mitigation"
        )
        
        # Check if calibration needs refresh
        if (calibration_data is None or 
            calibration_data.get_age_hours() > options.max_calibration_age_hours):
            
            if options.auto_calibration:
                # Perform new calibration
                calibration_result = refresh_calibration(
                    backend, num_qubits, "measurement_mitigation", options.calibration_shots
                )
                
                if calibration_result.success:
                    calibration_data = calibration_result.calibration_data
                else:
                    raise MitigationError(f"Calibration failed: {calibration_result.error_message}")
            else:
                raise MitigationError("No valid calibration available and auto-calibration disabled")
        
        # Convert calibration data to matrix
        if calibration_data:
            return CalibrationMatrix.from_dict(calibration_data.data)
        else:
            raise MitigationError("No calibration matrix available")
    
    def _execute_circuit(self, circuit: QuantumCircuit, backend: QuantumHardwareBackend,
                        shots: int) -> AggregatedResult:
        """Execute circuit on backend."""
        
        try:
            # Submit and wait for results
            result = backend.submit_and_wait(circuit, shots)
            
            # Convert to AggregatedResult
            return AggregatedResult(
                counts=result.counts,
                total_shots=shots,
                successful_shots=shots,
                backend_name=backend.name,
                metadata=getattr(result, 'metadata', {})
            )
        except Exception as e:
            from ..errors import ExecutionError
            raise ExecutionError(f"Circuit execution failed: {e}")
    
    def _calculate_overhead_factor(self, options: MitigationOptions, 
                                  total_shots: int, original_shots: int) -> float:
        """Calculate overhead factor from mitigation."""
        base_overhead = total_shots / original_shots if original_shots > 0 else 1.0
        
        # Add overhead for calibration
        if options.enable_measurement_mitigation and options.auto_calibration:
            # Calibration overhead is amortized over many runs
            calibration_overhead = 1.1  # 10% overhead estimate
            base_overhead *= calibration_overhead
        
        return base_overhead
    
    def _estimate_fidelity_improvement(self, original_result: AggregatedResult,
                                     mitigated_result: AggregatedResult,
                                     options: MitigationOptions) -> float:
        """Estimate fidelity improvement from mitigation."""
        
        if not original_result.probabilities or not mitigated_result.probabilities:
            return 0.0
        
        # Compare dominant outcome probabilities
        original_max = max(original_result.probabilities.values())
        mitigated_max = max(mitigated_result.probabilities.values())
        
        if original_max > 0:
            return (mitigated_max - original_max) / original_max
        
        return 0.0
    
    def _calculate_confidence_score(self, measurement_result: Optional[MitigationResult],
                                  zne_result: Optional[ZNEResult],
                                  options: MitigationOptions) -> float:
        """Calculate confidence score for mitigation."""
        
        base_confidence = 0.5
        
        # Add confidence from measurement mitigation
        if measurement_result and measurement_result.calibration_matrix:
            avg_fidelity = np.mean(list(measurement_result.calibration_matrix.readout_fidelity.values()))
            base_confidence += avg_fidelity * 0.3
        
        # Add confidence from ZNE
        if zne_result:
            r_squared = zne_result.r_squared
            base_confidence += r_squared * 0.2
        
        return min(base_confidence, 1.0)
    
    def _get_cache_key(self, circuit: QuantumCircuit, backend: QuantumHardwareBackend,
                      shots: int, options: MitigationOptions) -> str:
        """Generate cache key for results."""
        import hashlib
        
        # Create hash from circuit, backend, shots, and options
        circuit_str = str(circuit.operations)  # Simplified
        backend_str = backend.name
        options_str = str(options)
        
        combined = f"{circuit_str}_{backend_str}_{shots}_{options_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[MitigationPipelineResult]:
        """Get cached result if available."""
        with self._cache_lock:
            return self._results_cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: MitigationPipelineResult):
        """Cache result."""
        with self._cache_lock:
            self._results_cache[cache_key] = result
    
    def get_recommended_options(self, circuit: QuantumCircuit, 
                               backend: QuantumHardwareBackend,
                               target_fidelity: float = 0.9) -> MitigationOptions:
        """Get recommended mitigation options for a circuit and backend."""
        
        # Analyze circuit characteristics
        num_qubits = circuit.num_qubits
        depth = circuit.depth
        has_measurements = any(
            hasattr(op, 'operation_type') and op.operation_type.name == 'MEASUREMENT'
            for op in circuit.operations
        )
        
        # Determine recommended level
        if num_qubits <= 2 and depth <= 10:
            level = MitigationLevel.BASIC
        elif num_qubits <= 5 and depth <= 50:
            level = MitigationLevel.MODERATE
        else:
            level = MitigationLevel.AGGRESSIVE
        
        # Create options
        options = MitigationOptions(level=level)
        
        # Adjust based on backend characteristics
        if hasattr(backend, 'provider') and backend.provider == 'local':
            # Local simulator doesn't need aggressive mitigation
            options.enable_measurement_mitigation = False
            options.enable_zne = False
        
        return options


# Global instance
_mitigation_pipeline = None
_pipeline_lock = threading.Lock()


def get_mitigation_pipeline() -> MitigationPipeline:
    """Get global mitigation pipeline instance."""
    global _mitigation_pipeline
    if _mitigation_pipeline is None:
        with _pipeline_lock:
            if _mitigation_pipeline is None:
                _mitigation_pipeline = MitigationPipeline()
    return _mitigation_pipeline


def create_mitigation_pipeline() -> MitigationPipeline:
    """Create a new mitigation pipeline instance."""
    return MitigationPipeline()


def apply_mitigation_pipeline(circuit: QuantumCircuit, backend: QuantumHardwareBackend,
                            shots: int = 1000, options: Optional[MitigationOptions] = None) -> MitigationPipelineResult:
    """Apply error mitigation pipeline to a circuit."""
    pipeline = get_mitigation_pipeline()
    return pipeline.apply_mitigation(circuit, backend, shots, options) 