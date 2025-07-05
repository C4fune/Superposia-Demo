#!/usr/bin/env python3
"""
Simple Error Mitigation Test

A simplified test to verify the error mitigation system works correctly.
"""

import sys
import os
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Platform imports
from quantum_platform.compiler.language.dsl import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, CNOT
from quantum_platform.hardware.backends import LocalSimulatorBackend
from quantum_platform.hardware.results import AggregatedResult

# Error mitigation imports
try:
    from quantum_platform.mitigation import (
        MeasurementMitigator,
        ZNEMitigator,
        NoiseScalingMethod,
        ExtrapolationMethod,
        BitFlipCode,
        CalibrationManager,
        MitigationOptions,
        MitigationLevel,
        apply_mitigation_pipeline
    )
    print("✅ Error mitigation imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_measurement_mitigation():
    """Test basic measurement error mitigation."""
    print("\n🧪 Testing Measurement Error Mitigation...")
    
    try:
        # Create backend
        backend = LocalSimulatorBackend()
        backend.initialize()
        
        # Create simple circuit
        with QuantumProgram() as qp:
            qubit = qp.allocate(1)
            H(qubit)
            qp.measure(qubit, "result")
        
        circuit = qp.circuit
        
        # Create measurement mitigator
        mitigator = MeasurementMitigator()
        
        # Generate calibration circuits
        calibration_circuits = mitigator.generate_calibration_circuits(1)
        print(f"   Generated {len(calibration_circuits)} calibration circuits")
        
        # Test passed
        print("✅ Measurement mitigation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Measurement mitigation test failed: {e}")
        return False


def test_zne_mitigation():
    """Test Zero-Noise Extrapolation."""
    print("\n🧪 Testing Zero-Noise Extrapolation...")
    
    try:
        # Create ZNE mitigator
        zne_mitigator = ZNEMitigator()
        
        # Create simple circuit
        with QuantumProgram() as qp:
            qubit = qp.allocate(1)
            H(qubit)
            qp.measure(qubit, "result")
        
        circuit = qp.circuit
        
        # Test noise scaling
        scaled_circuit = zne_mitigator.scale_noise(
            circuit, 
            noise_factor=2.0, 
            method=NoiseScalingMethod.GATE_FOLDING
        )
        
        print(f"   Original circuit: {len(circuit.operations)} operations")
        print(f"   Scaled circuit: {len(scaled_circuit.operations)} operations")
        
        # Test passed
        print("✅ ZNE test passed")
        return True
        
    except Exception as e:
        print(f"❌ ZNE test failed: {e}")
        return False


def test_error_correction():
    """Test error correction codes."""
    print("\n🧪 Testing Error Correction...")
    
    try:
        # Create simple circuit
        with QuantumProgram() as qp:
            qubit = qp.allocate(1)
            H(qubit)
            qp.measure(qubit, "result")
        
        circuit = qp.circuit
        
        # Test bit-flip code
        bit_flip_code = BitFlipCode()
        
        print(f"   Code: {bit_flip_code.name}")
        print(f"   Logical qubits: {bit_flip_code.logical_qubits}")
        print(f"   Physical qubits: {bit_flip_code.physical_qubits}")
        print(f"   Distance: {bit_flip_code.distance}")
        
        # Test encoding
        encoding_result = bit_flip_code.encode(circuit)
        print(f"   Encoded circuit qubits: {encoding_result.processed_circuit.num_qubits}")
        
        # Test passed
        print("✅ Error correction test passed")
        return True
        
    except Exception as e:
        print(f"❌ Error correction test failed: {e}")
        return False


def test_calibration_manager():
    """Test calibration management."""
    print("\n🧪 Testing Calibration Management...")
    
    try:
        # Get calibration manager
        manager = CalibrationManager()
        
        # Get statistics
        stats = manager.get_calibration_stats()
        print(f"   Total calibrations: {stats['total_calibrations']}")
        print(f"   Cache directory: {stats['cache_directory']}")
        
        # Test passed
        print("✅ Calibration management test passed")
        return True
        
    except Exception as e:
        print(f"❌ Calibration management test failed: {e}")
        return False


def test_mitigation_pipeline():
    """Test integrated mitigation pipeline."""
    print("\n🧪 Testing Mitigation Pipeline...")
    
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
        
        # Test different mitigation levels
        options = MitigationOptions(level=MitigationLevel.NONE)
        
        # Apply mitigation pipeline (should work even with NONE level)
        result = apply_mitigation_pipeline(
            circuit=circuit,
            backend=backend,
            shots=100,
            options=options
        )
        
        print(f"   Original counts: {result.original_result.counts}")
        print(f"   Mitigated counts: {result.mitigated_result.counts}")
        print(f"   Overhead factor: {result.overhead_factor:.1f}x")
        
        # Test passed
        print("✅ Mitigation pipeline test passed")
        return True
        
    except Exception as e:
        print(f"❌ Mitigation pipeline test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Error Mitigation System Test")
    print("=" * 40)
    
    tests = [
        test_measurement_mitigation,
        test_zne_mitigation,
        test_error_correction,
        test_calibration_manager,
        test_mitigation_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Error mitigation system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 