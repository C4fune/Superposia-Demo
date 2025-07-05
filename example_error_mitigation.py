#!/usr/bin/env python3
"""
Error Mitigation and Correction Example

This example demonstrates the comprehensive error mitigation and correction
capabilities of the quantum platform, including:

1. Measurement Error Mitigation
2. Zero-Noise Extrapolation (ZNE)
3. Error Correction Codes
4. Calibration Management
5. Integrated Mitigation Pipeline

Usage:
    python example_error_mitigation.py
"""

import time
import numpy as np
from typing import Dict, List

# Platform imports
from quantum_platform.compiler.language.dsl import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, Y, Z, CNOT, RX, RY, RZ
from quantum_platform.hardware.backends import LocalSimulatorBackend
from quantum_platform.hardware.results import AggregatedResult

# Error mitigation imports
from quantum_platform.mitigation import (
    # Measurement mitigation
    MeasurementMitigator,
    get_measurement_mitigator,
    perform_measurement_calibration,
    apply_measurement_mitigation,
    
    # Zero-noise extrapolation
    ZNEMitigator,
    NoiseScalingMethod,
    ExtrapolationMethod,
    get_zne_mitigator,
    apply_zne_mitigation,
    
    # Error correction
    BitFlipCode,
    PhaseFlipCode,
    ShorCode,
    get_error_correction_code,
    encode_circuit,
    decode_circuit,
    
    # Calibration management
    CalibrationManager,
    get_calibration_manager,
    refresh_calibration,
    
    # Mitigation pipeline
    MitigationPipeline,
    MitigationOptions,
    MitigationLevel,
    create_mitigation_pipeline,
    apply_mitigation_pipeline
)

# Logging
from quantum_platform.observability.logging import get_logger

logger = get_logger(__name__)


def create_test_circuits():
    """Create test circuits for demonstrating error mitigation."""
    circuits = {}
    
    # 1. Single qubit superposition
    with QuantumProgram() as qp:
        qubit = qp.allocate(1)
        H(qubit)
        qp.measure(qubit, "result")
    
    circuits["single_qubit_superposition"] = qp.circuit
    
    # 2. Bell state
    with QuantumProgram() as qp:
        qubits = qp.allocate(2)
        H(qubits[0])
        CNOT(qubits[0], qubits[1])
        qp.measure(qubits, "bell_result")
    
    circuits["bell_state"] = qp.circuit
    
    # 3. Three qubit GHZ state
    with QuantumProgram() as qp:
        qubits = qp.allocate(3)
        H(qubits[0])
        CNOT(qubits[0], qubits[1])
        CNOT(qubits[1], qubits[2])
        qp.measure(qubits, "ghz_result")
    
    circuits["ghz_state"] = qp.circuit
    
    # 4. Parameterized rotation circuit
    with QuantumProgram() as qp:
        qubit = qp.allocate(1)
        RX(qubit, np.pi/4)
        RY(qubit, np.pi/3)
        RZ(qubit, np.pi/6)
        qp.measure(qubit, "rotation_result")
    
    circuits["parameterized_rotation"] = qp.circuit
    
    return circuits


def demonstrate_measurement_mitigation():
    """Demonstrate measurement error mitigation."""
    logger.info("=== Measurement Error Mitigation Demo ===")
    
    # Create backend
    backend = LocalSimulatorBackend()
    backend.initialize()
    
    # Create test circuit
    circuits = create_test_circuits()
    test_circuit = circuits["bell_state"]
    
    print(f"Circuit: {test_circuit.name}")
    print(f"Qubits: {test_circuit.num_qubits}")
    print(f"Depth: {test_circuit.depth}")
    
    # 1. Perform calibration
    print("\n1. Performing measurement calibration...")
    start_time = time.time()
    
    try:
        calibration_matrix = perform_measurement_calibration(
            backend=backend,
            num_qubits=test_circuit.num_qubits,
            shots=1000
        )
        
        calibration_time = time.time() - start_time
        print(f"   Calibration completed in {calibration_time:.2f} seconds")
        print(f"   Calibration matrix shape: {calibration_matrix.matrix.shape}")
        print(f"   Average readout fidelity: {np.mean(list(calibration_matrix.readout_fidelity.values())):.4f}")
        
        # Show calibration matrix
        print("\n   Calibration Matrix:")
        for i, row in enumerate(calibration_matrix.matrix):
            print(f"   Row {i}: {[f'{x:.3f}' for x in row]}")
    
    except Exception as e:
        print(f"   Calibration failed: {e}")
        return
    
    # 2. Execute circuit without mitigation
    print("\n2. Executing circuit without mitigation...")
    original_result = backend.submit_and_wait(test_circuit, shots=1000)
    
    aggregated_original = AggregatedResult(
        counts=original_result.counts,
        total_shots=1000,
        successful_shots=1000,
        backend_name=backend.name
    )
    
    print(f"   Original results: {aggregated_original.counts}")
    print(f"   Original probabilities: {aggregated_original.probabilities}")
    
    # 3. Apply measurement mitigation
    print("\n3. Applying measurement mitigation...")
    try:
        mitigation_result = apply_measurement_mitigation(
            result=aggregated_original,
            backend=backend,
            calibration_matrix=calibration_matrix
        )
        
        print(f"   Mitigated results: {mitigation_result.mitigated_counts}")
        print(f"   Mitigation factor: {mitigation_result.get_mitigation_factor():.3f}")
        print(f"   Fidelity improvement: {mitigation_result.fidelity_improvement:.4f}")
        print(f"   Mitigation overhead: {mitigation_result.mitigation_overhead:.6f} seconds")
        
        # Compare outcomes
        print("\n   Comparison:")
        for outcome in set(list(aggregated_original.counts.keys()) + list(mitigation_result.mitigated_counts.keys())):
            original_count = aggregated_original.counts.get(outcome, 0)
            mitigated_count = mitigation_result.mitigated_counts.get(outcome, 0)
            print(f"   {outcome}: {original_count} -> {mitigated_count}")
    
    except Exception as e:
        print(f"   Measurement mitigation failed: {e}")
    
    print()


def demonstrate_zne_mitigation():
    """Demonstrate Zero-Noise Extrapolation."""
    logger.info("=== Zero-Noise Extrapolation Demo ===")
    
    # Create backend
    backend = LocalSimulatorBackend()
    backend.initialize()
    
    # Create test circuit
    circuits = create_test_circuits()
    test_circuit = circuits["parameterized_rotation"]
    
    print(f"Circuit: {test_circuit.name}")
    print(f"Qubits: {test_circuit.num_qubits}")
    print(f"Depth: {test_circuit.depth}")
    
    # Define execution function
    def execute_circuit(circuit):
        result = backend.submit_and_wait(circuit, shots=1000)
        return AggregatedResult(
            counts=result.counts,
            total_shots=1000,
            successful_shots=1000,
            backend_name=backend.name
        )
    
    # 1. Test different noise scaling methods
    scaling_methods = [
        NoiseScalingMethod.GATE_FOLDING,
        NoiseScalingMethod.PARAMETER_SCALING,
        NoiseScalingMethod.IDENTITY_INSERTION
    ]
    
    for method in scaling_methods:
        print(f"\n1. Testing {method.value} noise scaling...")
        
        try:
            zne_result = apply_zne_mitigation(
                circuit=test_circuit,
                execution_func=execute_circuit,
                noise_factors=[1.0, 2.0, 3.0],
                scaling_method=method,
                extrapolation_method=ExtrapolationMethod.LINEAR
            )
            
            print(f"   Original result: {zne_result.original_result.counts}")
            print(f"   Extrapolated result: {zne_result.extrapolated_result.counts}")
            print(f"   Error reduction: {zne_result.get_error_reduction():.4f}")
            print(f"   R-squared: {zne_result.r_squared:.4f}")
            print(f"   Overhead factor: {zne_result.overhead_factor:.1f}x")
            print(f"   Execution time: {zne_result.execution_time:.2f} seconds")
            
        except Exception as e:
            print(f"   ZNE with {method.value} failed: {e}")
    
    # 2. Test different extrapolation methods
    extrapolation_methods = [
        ExtrapolationMethod.LINEAR,
        ExtrapolationMethod.POLYNOMIAL,
        ExtrapolationMethod.EXPONENTIAL
    ]
    
    print("\n2. Testing different extrapolation methods...")
    for method in extrapolation_methods:
        print(f"\n   Testing {method.value} extrapolation...")
        
        try:
            zne_result = apply_zne_mitigation(
                circuit=test_circuit,
                execution_func=execute_circuit,
                noise_factors=[1.0, 1.5, 2.0, 2.5],
                scaling_method=NoiseScalingMethod.GATE_FOLDING,
                extrapolation_method=method
            )
            
            print(f"   R-squared: {zne_result.r_squared:.4f}")
            print(f"   Extrapolation error: {zne_result.extrapolation_error:.4f}")
            print(f"   Error reduction: {zne_result.get_error_reduction():.4f}")
            
        except Exception as e:
            print(f"   {method.value} extrapolation failed: {e}")
    
    print()


def demonstrate_error_correction():
    """Demonstrate error correction codes."""
    logger.info("=== Error Correction Demo ===")
    
    # Create simple single-qubit circuit
    with QuantumProgram() as qp:
        qubit = qp.allocate(1)
        H(qubit)
        qp.measure(qubit, "result")
    
    test_circuit = qp.circuit
    
    print(f"Original circuit: {test_circuit.num_qubits} qubits, {test_circuit.depth} depth")
    
    # Test different error correction codes
    error_codes = ["bit_flip", "phase_flip", "shor"]
    
    for code_name in error_codes:
        print(f"\n1. Testing {code_name} error correction...")
        
        try:
            # Get error correction code
            error_code = get_error_correction_code(code_name)
            print(f"   Code: {error_code.name}")
            print(f"   Logical qubits: {error_code.logical_qubits}")
            print(f"   Physical qubits: {error_code.physical_qubits}")
            print(f"   Distance: {error_code.distance}")
            
            # Encode circuit
            encoding_result = encode_circuit(test_circuit, code_name)
            print(f"   Encoded circuit: {encoding_result.processed_circuit.num_qubits} qubits")
            print(f"   Code rate: {encoding_result.get_code_rate():.3f}")
            print(f"   Overhead ratio: {encoding_result.get_overhead_ratio():.1f}x")
            
            # Generate syndrome circuit
            syndrome_circuit = error_code.generate_syndrome_circuit()
            print(f"   Syndrome circuit: {syndrome_circuit.num_qubits} qubits")
            
            # Test error correction
            print(f"   Testing syndrome correction...")
            test_syndromes = ["00", "01", "10", "11"]
            
            for syndrome in test_syndromes:
                if len(syndrome) == 2:  # Only test 2-bit syndromes
                    corrections = error_code.correct_errors(syndrome)
                    print(f"   Syndrome {syndrome}: {corrections}")
            
        except Exception as e:
            print(f"   {code_name} error correction failed: {e}")
    
    print()


def demonstrate_calibration_management():
    """Demonstrate calibration data management."""
    logger.info("=== Calibration Management Demo ===")
    
    # Get calibration manager
    calibration_manager = get_calibration_manager()
    
    # Create backend
    backend = LocalSimulatorBackend()
    backend.initialize()
    
    print("1. Calibration Manager Status:")
    stats = calibration_manager.get_calibration_stats()
    print(f"   Total calibrations: {stats['total_calibrations']}")
    print(f"   Calibration types: {stats['calibration_types']}")
    print(f"   Average quality score: {stats['average_quality_score']:.3f}")
    print(f"   Cache directory: {stats['cache_directory']}")
    
    # 2. Refresh calibration
    print("\n2. Refreshing calibration...")
    try:
        calibration_result = refresh_calibration(
            backend=backend,
            num_qubits=2,
            calibration_type="measurement_mitigation",
            shots=1000
        )
        
        if calibration_result.success:
            print(f"   Calibration successful!")
            print(f"   Execution time: {calibration_result.execution_time:.2f} seconds")
            print(f"   Circuits executed: {calibration_result.circuits_executed}")
            print(f"   Total shots: {calibration_result.total_shots}")
            print(f"   Quality score: {calibration_result.quality_score:.3f}")
            print(f"   Recommended refresh: {calibration_result.recommended_refresh_hours:.1f} hours")
        else:
            print(f"   Calibration failed: {calibration_result.error_message}")
    
    except Exception as e:
        print(f"   Calibration refresh failed: {e}")
    
    # 3. Check calibration status
    print("\n3. Checking calibration status...")
    needs_refresh = calibration_manager.needs_refresh(
        backend_name=backend.name,
        device_id=backend.name,
        num_qubits=2,
        calibration_type="measurement_mitigation"
    )
    print(f"   Needs refresh: {needs_refresh}")
    
    # 4. Clear expired calibrations
    print("\n4. Clearing expired calibrations...")
    cleared_count = calibration_manager.clear_expired_calibrations()
    print(f"   Cleared {cleared_count} expired calibrations")
    
    print()


def demonstrate_mitigation_pipeline():
    """Demonstrate the integrated mitigation pipeline."""
    logger.info("=== Mitigation Pipeline Demo ===")
    
    # Create backend
    backend = LocalSimulatorBackend()
    backend.initialize()
    
    # Create test circuit
    circuits = create_test_circuits()
    test_circuit = circuits["ghz_state"]
    
    print(f"Circuit: {test_circuit.name}")
    print(f"Qubits: {test_circuit.num_qubits}")
    print(f"Depth: {test_circuit.depth}")
    
    # Test different mitigation levels
    mitigation_levels = [
        MitigationLevel.NONE,
        MitigationLevel.BASIC,
        MitigationLevel.MODERATE,
        MitigationLevel.AGGRESSIVE
    ]
    
    pipeline = create_mitigation_pipeline()
    
    for level in mitigation_levels:
        print(f"\n1. Testing {level.value} mitigation level...")
        
        try:
            # Create options
            options = MitigationOptions(level=level)
            
            # Apply mitigation pipeline
            start_time = time.time()
            pipeline_result = apply_mitigation_pipeline(
                circuit=test_circuit,
                backend=backend,
                shots=1000,
                options=options
            )
            total_time = time.time() - start_time
            
            # Show results
            print(f"   Original counts: {pipeline_result.original_result.counts}")
            print(f"   Mitigated counts: {pipeline_result.mitigated_result.counts}")
            print(f"   Fidelity improvement: {pipeline_result.fidelity_improvement:.4f}")
            print(f"   Confidence score: {pipeline_result.confidence_score:.3f}")
            print(f"   Overhead factor: {pipeline_result.overhead_factor:.1f}x")
            print(f"   Total execution time: {total_time:.2f} seconds")
            
            # Show improvement summary
            improvement = pipeline_result.get_improvement_summary()
            print(f"   Applied techniques: {improvement['applied_techniques']}")
            print(f"   Improvement factor: {improvement['improvement_factor']:.3f}")
            
        except Exception as e:
            print(f"   {level.value} mitigation failed: {e}")
    
    # 2. Get recommended options
    print("\n2. Getting recommended mitigation options...")
    try:
        recommended_options = pipeline.get_recommended_options(
            circuit=test_circuit,
            backend=backend,
            target_fidelity=0.95
        )
        
        print(f"   Recommended level: {recommended_options.level.value}")
        print(f"   Measurement mitigation: {recommended_options.enable_measurement_mitigation}")
        print(f"   ZNE: {recommended_options.enable_zne}")
        print(f"   Error correction: {recommended_options.enable_error_correction}")
        print(f"   Auto calibration: {recommended_options.auto_calibration}")
        
        # Apply recommended options
        pipeline_result = apply_mitigation_pipeline(
            circuit=test_circuit,
            backend=backend,
            shots=1000,
            options=recommended_options
        )
        
        print(f"   Recommended mitigation results:")
        print(f"   Fidelity improvement: {pipeline_result.fidelity_improvement:.4f}")
        print(f"   Confidence score: {pipeline_result.confidence_score:.3f}")
        
    except Exception as e:
        print(f"   Recommended options failed: {e}")
    
    print()


def compare_mitigation_effectiveness():
    """Compare the effectiveness of different mitigation techniques."""
    logger.info("=== Mitigation Effectiveness Comparison ===")
    
    # Create backend
    backend = LocalSimulatorBackend()
    backend.initialize()
    
    # Create test circuits
    circuits = create_test_circuits()
    
    # Test each circuit with different mitigation approaches
    for circuit_name, circuit in circuits.items():
        print(f"\n{circuit_name.upper()} ({circuit.num_qubits} qubits, depth {circuit.depth})")
        print("-" * 50)
        
        # Baseline (no mitigation)
        try:
            baseline_result = backend.submit_and_wait(circuit, shots=1000)
            baseline_aggregated = AggregatedResult(
                counts=baseline_result.counts,
                total_shots=1000,
                successful_shots=1000,
                backend_name=backend.name
            )
            
            baseline_max_prob = max(baseline_aggregated.probabilities.values()) if baseline_aggregated.probabilities else 0
            print(f"Baseline max probability: {baseline_max_prob:.4f}")
            
            # Test different mitigation approaches
            mitigation_approaches = [
                ("Measurement only", MitigationOptions(level=MitigationLevel.BASIC)),
                ("ZNE only", MitigationOptions(
                    level=MitigationLevel.CUSTOM,
                    enable_measurement_mitigation=False,
                    enable_zne=True
                )),
                ("Combined", MitigationOptions(level=MitigationLevel.MODERATE))
            ]
            
            for approach_name, options in mitigation_approaches:
                try:
                    pipeline_result = apply_mitigation_pipeline(
                        circuit=circuit,
                        backend=backend,
                        shots=1000,
                        options=options
                    )
                    
                    mitigated_max_prob = max(pipeline_result.mitigated_result.probabilities.values()) if pipeline_result.mitigated_result.probabilities else 0
                    improvement = (mitigated_max_prob - baseline_max_prob) / baseline_max_prob if baseline_max_prob > 0 else 0
                    
                    print(f"{approach_name:15} | Max prob: {mitigated_max_prob:.4f} | Improvement: {improvement:+.2%} | Overhead: {pipeline_result.overhead_factor:.1f}x")
                    
                except Exception as e:
                    print(f"{approach_name:15} | FAILED: {e}")
            
        except Exception as e:
            print(f"Baseline execution failed: {e}")
    
    print()


def main():
    """Main demonstration function."""
    print("Quantum Error Mitigation and Correction Demo")
    print("=" * 50)
    
    try:
        # Individual component demonstrations
        demonstrate_measurement_mitigation()
        demonstrate_zne_mitigation()
        demonstrate_error_correction()
        demonstrate_calibration_management()
        demonstrate_mitigation_pipeline()
        
        # Effectiveness comparison
        compare_mitigation_effectiveness()
        
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main() 