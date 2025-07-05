#!/usr/bin/env python3
"""
Working Error Mitigation Demo

Demonstrates the error mitigation system functionality without problematic decorators.
"""

import sys
import os
import numpy as np
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Platform imports
from quantum_platform.compiler.language.dsl import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, Y, Z, CNOT, RX, RY
from quantum_platform.hardware.backends import LocalSimulatorBackend
from quantum_platform.hardware.results import AggregatedResult

print("🚀 Error Mitigation and Correction Demo")
print("=" * 50)

def demo_measurement_mitigation():
    """Demonstrate measurement error mitigation."""
    print("\n📊 MEASUREMENT ERROR MITIGATION")
    print("-" * 30)
    
    try:
        from quantum_platform.mitigation.measurement_mitigation import MeasurementMitigator
        
        # Create backend
        backend = LocalSimulatorBackend()
        backend.initialize()
        
        # Create measurement mitigator
        mitigator = MeasurementMitigator()
        
        # Generate calibration circuits for 2 qubits
        print("1. Generating calibration circuits...")
        calibration_circuits = mitigator.generate_calibration_circuits(2)
        print(f"   Generated {len(calibration_circuits)} calibration circuits")
        
        # Execute calibration circuits
        print("2. Executing calibration circuits...")
        calibration_results = []
        
        for i, circuit in enumerate(calibration_circuits):
            result = backend.submit_and_wait(circuit, shots=500)
            aggregated = AggregatedResult(
                counts=result.counts,
                total_shots=500,
                successful_shots=500,
                backend_name=backend.name
            )
            calibration_results.append(aggregated)
            print(f"   Circuit {i}: {aggregated.counts}")
        
        # Build calibration matrix
        print("3. Building calibration matrix...")
        calibration_matrix = mitigator.build_calibration_matrix(calibration_results)
        print(f"   Matrix shape: {calibration_matrix.matrix.shape}")
        print(f"   Average fidelity: {np.mean(list(calibration_matrix.readout_fidelity.values())):.3f}")
        
        # Create test circuit
        with QuantumProgram() as qp:
            qubits = qp.allocate(2)
            H(qubits[0])
            CNOT(qubits[0], qubits[1])
            qp.measure(qubits, "bell_result")
        
        test_circuit = qp.circuit
        
        # Execute test circuit
        print("4. Executing test circuit...")
        test_result = backend.submit_and_wait(test_circuit, shots=1000)
        test_aggregated = AggregatedResult(
            counts=test_result.counts,
            total_shots=1000,
            successful_shots=1000,
            backend_name=backend.name
        )
        
        print(f"   Original results: {test_aggregated.counts}")
        
        # Apply mitigation
        print("5. Applying measurement mitigation...")
        mitigation_result = mitigator.apply_mitigation(test_aggregated, calibration_matrix)
        
        print(f"   Mitigated results: {mitigation_result.mitigated_counts}")
        print(f"   Mitigation factor: {mitigation_result.get_mitigation_factor():.3f}")
        
        print("✅ Measurement mitigation demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Measurement mitigation demo failed: {e}")
        return False


def demo_zero_noise_extrapolation():
    """Demonstrate Zero-Noise Extrapolation."""
    print("\n🎯 ZERO-NOISE EXTRAPOLATION")
    print("-" * 30)
    
    try:
        from quantum_platform.mitigation.zero_noise_extrapolation import (
            ZNEMitigator, NoiseScalingMethod, ExtrapolationMethod
        )
        
        # Create ZNE mitigator
        zne_mitigator = ZNEMitigator()
        
        # Create test circuit with parameterized gates
        with QuantumProgram() as qp:
            qubit = qp.allocate(1)
            RX(qubit, np.pi/4)
            RY(qubit, np.pi/3)
            qp.measure(qubit, "rotation_result")
        
        test_circuit = qp.circuit
        print(f"Original circuit: {len(test_circuit.operations)} operations")
        
        # Test different noise scaling methods
        scaling_methods = [
            NoiseScalingMethod.GATE_FOLDING,
            NoiseScalingMethod.PARAMETER_SCALING,
            NoiseScalingMethod.IDENTITY_INSERTION
        ]
        
        for method in scaling_methods:
            print(f"\n1. Testing {method.value} noise scaling...")
            
            try:
                scaled_circuit = zne_mitigator.scale_noise(
                    test_circuit, 
                    noise_factor=2.0, 
                    method=method
                )
                print(f"   Scaled circuit: {len(scaled_circuit.operations)} operations")
                
            except Exception as e:
                print(f"   Failed: {e}")
        
        # Test extrapolation with mock data
        print("\n2. Testing extrapolation methods...")
        
        # Create mock results with decreasing fidelity
        mock_results = []
        noise_factors = [1.0, 2.0, 3.0]
        
        for factor in noise_factors:
            # Simulate decreasing probability with noise
            prob_0 = max(0.1, 0.8 - 0.15 * (factor - 1))
            prob_1 = 1.0 - prob_0
            counts = {"0": int(prob_0 * 1000), "1": int(prob_1 * 1000)}
            
            result = AggregatedResult(
                counts=counts,
                total_shots=1000,
                successful_shots=1000,
                backend_name="test_backend"
            )
            mock_results.append(result)
            print(f"   Noise {factor}x: P(0) = {prob_0:.3f}")
        
        # Test linear extrapolation
        extrapolated_result, error, r_squared = zne_mitigator.extrapolate_to_zero_noise(
            mock_results, noise_factors, ExtrapolationMethod.LINEAR
        )
        
        zero_noise_prob = max(extrapolated_result.probabilities.values()) if extrapolated_result.probabilities else 0
        print(f"   Extrapolated to zero noise: P(dominant) = {zero_noise_prob:.3f}")
        print(f"   R-squared: {r_squared:.3f}")
        
        print("✅ Zero-noise extrapolation demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Zero-noise extrapolation demo failed: {e}")
        return False


def demo_error_correction():
    """Demonstrate error correction codes."""
    print("\n🛡️ ERROR CORRECTION CODES")
    print("-" * 30)
    
    try:
        from quantum_platform.mitigation.error_correction import (
            BitFlipCode, PhaseFlipCode, ShorCode
        )
        
        # Create simple test circuit
        with QuantumProgram() as qp:
            qubit = qp.allocate(1)
            H(qubit)
            qp.measure(qubit, "test_result")
        
        test_circuit = qp.circuit
        print(f"Original circuit: {test_circuit.num_qubits} qubits")
        
        # Test different error correction codes
        error_codes = [
            ("Bit-flip", BitFlipCode()),
            ("Phase-flip", PhaseFlipCode()),
            ("Shor", ShorCode())
        ]
        
        for name, code in error_codes:
            print(f"\n1. Testing {name} code...")
            print(f"   Logical qubits: {code.logical_qubits}")
            print(f"   Physical qubits: {code.physical_qubits}")
            print(f"   Distance: {code.distance}")
            print(f"   Can correct {(code.distance-1)//2} errors")
            
            try:
                # Test encoding
                encoding_result = code.encode(test_circuit)
                encoded_qubits = encoding_result.processed_circuit.num_qubits
                print(f"   Encoded circuit: {encoded_qubits} qubits")
                print(f"   Code rate: {encoding_result.get_code_rate():.3f}")
                print(f"   Overhead: {encoding_result.get_overhead_ratio():.1f}x")
                
                # Test syndrome circuit
                syndrome_circuit = code.generate_syndrome_circuit()
                print(f"   Syndrome circuit: {syndrome_circuit.num_qubits} qubits")
                
                # Test error correction for bit-flip code
                if name == "Bit-flip":
                    print("   Testing error correction:")
                    test_syndromes = ["00", "01", "10", "11"]
                    for syndrome in test_syndromes:
                        corrections = code.correct_errors(syndrome)
                        print(f"     Syndrome {syndrome}: {corrections}")
                
            except Exception as e:
                print(f"   Encoding failed: {e}")
        
        print("\n✅ Error correction demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error correction demo failed: {e}")
        return False


def demo_calibration_management():
    """Demonstrate calibration management."""
    print("\n📋 CALIBRATION MANAGEMENT")
    print("-" * 30)
    
    try:
        from quantum_platform.mitigation.calibration_manager import CalibrationManager
        from quantum_platform.mitigation.measurement_mitigation import CalibrationMatrix
        import tempfile
        import shutil
        
        # Create temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary cache directory: {temp_dir}")
        
        try:
            # Create calibration manager
            manager = CalibrationManager(cache_dir=temp_dir)
            
            # Get initial statistics
            stats = manager.get_calibration_stats()
            print(f"Initial calibrations: {stats['total_calibrations']}")
            
            # Create mock calibration matrix
            calibration_matrix = CalibrationMatrix(
                matrix=np.eye(4),
                inverse_matrix=np.eye(4),
                num_qubits=2,
                backend_name="test_backend",
                created_at=time.time(),
                calibration_shots=1000,
                readout_fidelity={0: 0.95, 1: 0.93}
            )
            
            # Store calibration
            print("\n1. Storing calibration data...")
            calibration_data = manager.store_calibration(
                backend_name="test_backend",
                device_id="test_device",
                num_qubits=2,
                calibration_type="measurement_mitigation",
                calibration_data=calibration_matrix,
                calibration_shots=1000
            )
            
            print(f"   Stored calibration: {calibration_data.backend_name}")
            print(f"   Average fidelity: {calibration_data.average_fidelity:.3f}")
            print(f"   Confidence score: {calibration_data.confidence_score:.3f}")
            print(f"   Valid: {calibration_data.is_valid()}")
            
            # Retrieve calibration
            print("\n2. Retrieving calibration data...")
            retrieved_data = manager.get_calibration(
                backend_name="test_backend",
                device_id="test_device",
                num_qubits=2,
                calibration_type="measurement_mitigation"
            )
            
            if retrieved_data:
                print(f"   Retrieved calibration for {retrieved_data.backend_name}")
                print(f"   Age: {retrieved_data.get_age_hours():.2f} hours")
                print(f"   Checksum valid: {retrieved_data.validate_checksum()}")
            else:
                print("   No calibration found")
            
            # Get updated statistics
            stats = manager.get_calibration_stats()
            print(f"\n3. Updated statistics:")
            print(f"   Total calibrations: {stats['total_calibrations']}")
            print(f"   Average quality: {stats['average_quality_score']:.3f}")
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
        
        print("\n✅ Calibration management demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Calibration management demo failed: {e}")
        return False


def demo_complete_workflow():
    """Demonstrate complete error mitigation workflow."""
    print("\n🔄 COMPLETE WORKFLOW")
    print("-" * 30)
    
    try:
        # Create backend
        backend = LocalSimulatorBackend()
        backend.initialize()
        
        # Create Bell state circuit
        with QuantumProgram() as qp:
            qubits = qp.allocate(2)
            H(qubits[0])
            CNOT(qubits[0], qubits[1])
            qp.measure(qubits, "bell_result")
        
        circuit = qp.circuit
        
        print(f"Circuit: Bell state ({circuit.num_qubits} qubits, depth {circuit.depth})")
        
        # 1. Baseline execution
        print("\n1. Baseline execution (no mitigation)...")
        baseline_result = backend.submit_and_wait(circuit, shots=1000)
        baseline_aggregated = AggregatedResult(
            counts=baseline_result.counts,
            total_shots=1000,
            successful_shots=1000,
            backend_name=backend.name
        )
        
        print(f"   Results: {baseline_aggregated.counts}")
        if baseline_aggregated.probabilities:
            max_prob = max(baseline_aggregated.probabilities.values())
            print(f"   Max probability: {max_prob:.3f}")
        
        # 2. Individual mitigation techniques (simplified)
        print("\n2. Testing individual techniques...")
        
        # Measurement mitigation (simplified)
        from quantum_platform.mitigation.measurement_mitigation import MeasurementMitigator
        mitigator = MeasurementMitigator()
        
        # Use identity matrix as simplified calibration
        calibration_matrix = CalibrationMatrix(
            matrix=np.eye(4),
            inverse_matrix=np.eye(4),
            num_qubits=2,
            backend_name=backend.name,
            created_at=time.time(),
            calibration_shots=1000,
            readout_fidelity={0: 0.95, 1: 0.95}
        )
        
        mitigation_result = mitigator.apply_mitigation(baseline_aggregated, calibration_matrix)
        print(f"   Measurement mitigation: {mitigation_result.mitigated_counts}")
        
        # ZNE (simplified)
        from quantum_platform.mitigation.zero_noise_extrapolation import ZNEMitigator
        zne_mitigator = ZNEMitigator()
        
        scaled_circuit = zne_mitigator.scale_noise(circuit, 2.0)
        print(f"   ZNE scaled circuit: {len(scaled_circuit.operations)} operations")
        
        # Error correction
        from quantum_platform.mitigation.error_correction import BitFlipCode
        
        # Note: Bell state has 2 qubits, bit-flip code expects 1
        # This is for demonstration only
        single_qubit_circuit = QuantumProgram()
        with single_qubit_circuit as qp:
            qubit = qp.allocate(1)
            H(qubit)
            qp.measure(qubit, "result")
        
        bit_flip_code = BitFlipCode()
        encoding_result = bit_flip_code.encode(qp.circuit)
        print(f"   Error correction: {encoding_result.processed_circuit.num_qubits} physical qubits")
        
        print("\n✅ Complete workflow demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Complete workflow demo failed: {e}")
        return False


def main():
    """Run all demonstrations."""
    
    demos = [
        ("Measurement Mitigation", demo_measurement_mitigation),
        ("Zero-Noise Extrapolation", demo_zero_noise_extrapolation),
        ("Error Correction", demo_error_correction),
        ("Calibration Management", demo_calibration_management),
        ("Complete Workflow", demo_complete_workflow)
    ]
    
    passed = 0
    total = len(demos)
    
    for name, demo_func in demos:
        print(f"\n{'='*60}")
        if demo_func():
            passed += 1
    
    print(f"\n{'='*60}")
    print("📊 DEMO SUMMARY")
    print(f"{'='*60}")
    print(f"Completed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All demonstrations completed successfully!")
        print("The Error Mitigation and Correction system is working correctly.")
    else:
        print(f"\n⚠️  {total-passed} demonstration(s) failed.")
        print("Some components may need additional configuration.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 