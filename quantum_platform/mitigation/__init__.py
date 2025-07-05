"""
Quantum Error Mitigation and Correction Module

This module provides comprehensive error mitigation and correction capabilities
for quantum computing operations, including measurement error mitigation,
zero-noise extrapolation (ZNE), and error correction code support.
"""

from .measurement_mitigation import (
    MeasurementMitigator,
    CalibrationMatrix,
    MitigationResult,
    get_measurement_mitigator,
    perform_measurement_calibration,
    apply_measurement_mitigation
)

from .zero_noise_extrapolation import (
    ZNEMitigator,
    NoiseScalingMethod,
    ExtrapolationMethod,
    ZNEResult,
    get_zne_mitigator,
    apply_zne_mitigation
)

from .error_correction import (
    ErrorCorrectionCode,
    BitFlipCode,
    PhaseFlipCode,
    ShorCode,
    get_error_correction_code,
    encode_circuit,
    decode_circuit
)

from .calibration_manager import (
    CalibrationManager,
    CalibrationData,
    CalibrationResult,
    get_calibration_manager,
    refresh_calibration,
    get_cached_calibration
)

from .mitigation_pipeline import (
    MitigationPipeline,
    MitigationOptions,
    MitigationLevel,
    create_mitigation_pipeline,
    apply_mitigation_pipeline
)

__all__ = [
    # Measurement Mitigation
    'MeasurementMitigator',
    'CalibrationMatrix',
    'MitigationResult',
    'get_measurement_mitigator',
    'perform_measurement_calibration',
    'apply_measurement_mitigation',
    
    # Zero-Noise Extrapolation
    'ZNEMitigator',
    'NoiseScalingMethod',
    'ExtrapolationMethod',
    'ZNEResult',
    'get_zne_mitigator',
    'apply_zne_mitigation',
    
    # Error Correction
    'ErrorCorrectionCode',
    'BitFlipCode',
    'PhaseFlipCode',
    'ShorCode',
    'get_error_correction_code',
    'encode_circuit',
    'decode_circuit',
    
    # Calibration Management
    'CalibrationManager',
    'CalibrationData',
    'CalibrationResult',
    'get_calibration_manager',
    'refresh_calibration',
    'get_cached_calibration',
    
    # Mitigation Pipeline
    'MitigationPipeline',
    'MitigationOptions',
    'MitigationLevel',
    'create_mitigation_pipeline',
    'apply_mitigation_pipeline'
] 