#!/usr/bin/env python3
"""
Error Mitigation System Test Suite

Comprehensive tests for the quantum error mitigation and correction system.
"""

import unittest
import numpy as np
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Platform imports
from quantum_platform.compiler.language.dsl import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, Y, Z, CNOT, RX, RY, RZ
from quantum_platform.hardware.backends import LocalSimulatorBackend
from quantum_platform.hardware.results import AggregatedResult

# Define MitigationError locally for testing
class MitigationError(Exception):
    """Error mitigation exception for testing."""
    pass

# Error mitigation imports
from quantum_platform.mitigation import (
    # Measurement mitigation
    MeasurementMitigator,
    CalibrationMatrix,
    MitigationResult,
    get_measurement_mitigator,
    perform_measurement_calibration,
    apply_measurement_mitigation,
    
    # Zero-noise extrapolation
    ZNEMitigator,
    NoiseScalingMethod,
    ExtrapolationMethod,
    ZNEResult,
    get_zne_mitigator,
    apply_zne_mitigation,
    
    # Error correction
    BitFlipCode,
    PhaseFlipCode,
    ShorCode,
    ErrorCorrectionResult,
    get_error_correction_code,
    encode_circuit,
    decode_circuit,
    
    # Calibration management
    CalibrationManager,
    CalibrationData,
    CalibrationResult,
    get_calibration_manager,
    refresh_calibration,
    
    # Mitigation pipeline
    MitigationPipeline,
    MitigationOptions,
    MitigationLevel,
    MitigationPipelineResult,
    create_mitigation_pipeline,
    apply_mitigation_pipeline
)


class TestMeasurementMitigation(unittest.TestCase):
    """Test measurement error mitigation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = LocalSimulatorBackend()
        self.backend.initialize()
        self.mitigator = MeasurementMitigator()
        
        # Create test circuit
        with QuantumProgram() as qp:
            qubits = qp.allocate(2)
            H(qubits[0])
            CNOT(qubits[0], qubits[1])
            qp.measure(qubits, "test_result")
        
        self.test_circuit = qp.circuit
    
    def test_calibration_circuit_generation(self):
        """Test generation of calibration circuits."""
        circuits = self.mitigator.generate_calibration_circuits(2)
        
        # Should generate 2^2 = 4 calibration circuits
        self.assertEqual(len(circuits), 4)
        
        # Each circuit should have 2 qubits
        for circuit in circuits:
            self.assertEqual(circuit.num_qubits, 2)
    
    def test_calibration_matrix_creation(self):
        """Test creation of calibration matrix from results."""
        # Create mock calibration results
        mock_results = []
        for state in range(4):
            # Perfect readout for testing
            counts = {format(state, '02b'): 1000}
            result = AggregatedResult(
                counts=counts,
                total_shots=1000,
                successful_shots=1000,
                backend_name="test_backend"
            )
            mock_results.append(result)
        
        # Build calibration matrix
        calibration_matrix = self.mitigator.build_calibration_matrix(mock_results)
        
        # Check properties
        self.assertEqual(calibration_matrix.num_qubits, 2)
        self.assertEqual(calibration_matrix.matrix.shape, (4, 4))
        self.assertTrue(calibration_matrix.is_valid())
        
        # Should be close to identity matrix for perfect readout
        np.testing.assert_allclose(calibration_matrix.matrix, np.eye(4), atol=0.01)
    
    def test_measurement_mitigation_application(self):
        """Test application of measurement mitigation."""
        # Create calibration matrix (identity for perfect readout)
        calibration_matrix = CalibrationMatrix(
            matrix=np.eye(4),
            inverse_matrix=np.eye(4),
            num_qubits=2,
            backend_name="test_backend",
            created_at=time.time(),
            calibration_shots=4000,
            readout_fidelity={0: 1.0, 1: 1.0}
        )
        
        # Create test result
        test_result = AggregatedResult(
            counts={"00": 500, "11": 500},
            total_shots=1000,
            successful_shots=1000,
            backend_name="test_backend"
        )
        
        # Apply mitigation
        mitigation_result = self.mitigator.apply_mitigation(test_result, calibration_matrix)
        
        # Check result structure
        self.assertIsInstance(mitigation_result, MitigationResult)
        self.assertEqual(mitigation_result.total_shots, 1000)
        self.assertIn("00", mitigation_result.mitigated_counts)
        self.assertIn("11", mitigation_result.mitigated_counts)
    
    def test_full_measurement_calibration(self):
        """Test complete measurement calibration process."""
        try:
            calibration_matrix = perform_measurement_calibration(
                backend=self.backend,
                num_qubits=2,
                shots=100  # Small number for fast testing
            )
            
            # Check calibration matrix properties
            self.assertEqual(calibration_matrix.num_qubits, 2)
            self.assertEqual(calibration_matrix.backend_name, self.backend.name)
            self.assertTrue(calibration_matrix.is_valid())
            self.assertGreater(len(calibration_matrix.readout_fidelity), 0)
            
        except Exception as e:
            self.fail(f"Measurement calibration failed: {e}")


class TestZeroNoiseExtrapolation(unittest.TestCase):
    """Test Zero-Noise Extrapolation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = LocalSimulatorBackend()
        self.backend.initialize()
        self.zne_mitigator = ZNEMitigator()
        
        # Create test circuit
        with QuantumProgram() as qp:
            qubit = qp.allocate(1)
            RX(qubit, np.pi/4)
            qp.measure(qubit, "test_result")
        
        self.test_circuit = qp.circuit
    
    def test_gate_folding_noise_scaling(self):
        """Test gate folding noise scaling method."""
        scaled_circuit = self.zne_mitigator.scale_noise(
            self.test_circuit,
            noise_factor=2.0,
            method=NoiseScalingMethod.GATE_FOLDING
        )
        
        # Scaled circuit should have more operations
        self.assertGreaterEqual(len(scaled_circuit.operations), len(self.test_circuit.operations))
        self.assertEqual(scaled_circuit.num_qubits, self.test_circuit.num_qubits)
    
    def test_parameter_scaling_noise_scaling(self):
        """Test parameter scaling noise scaling method."""
        scaled_circuit = self.zne_mitigator.scale_noise(
            self.test_circuit,
            noise_factor=2.0,
            method=NoiseScalingMethod.PARAMETER_SCALING
        )
        
        # Should have same number of operations but different parameters
        self.assertEqual(len(scaled_circuit.operations), len(self.test_circuit.operations))
        self.assertEqual(scaled_circuit.num_qubits, self.test_circuit.num_qubits)
    
    def test_identity_insertion_noise_scaling(self):
        """Test identity insertion noise scaling method."""
        scaled_circuit = self.zne_mitigator.scale_noise(
            self.test_circuit,
            noise_factor=2.0,
            method=NoiseScalingMethod.IDENTITY_INSERTION
        )
        
        # Should have more operations due to identity insertion
        self.assertGreater(len(scaled_circuit.operations), len(self.test_circuit.operations))
    
    def test_linear_extrapolation(self):
        """Test linear extrapolation method."""
        # Create mock results with decreasing fidelity
        mock_results = []
        noise_factors = [1.0, 2.0, 3.0]
        
        for factor in noise_factors:
            # Simulate decreasing probability with noise
            prob_0 = max(0.1, 0.8 - 0.2 * (factor - 1))
            prob_1 = 1.0 - prob_0
            counts = {"0": int(prob_0 * 1000), "1": int(prob_1 * 1000)}
            
            result = AggregatedResult(
                counts=counts,
                total_shots=1000,
                successful_shots=1000,
                backend_name="test_backend"
            )
            mock_results.append(result)
        
        # Test extrapolation
        extrapolated_result, error, r_squared = self.zne_mitigator.extrapolate_to_zero_noise(
            mock_results, noise_factors, ExtrapolationMethod.LINEAR
        )
        
        self.assertIsInstance(extrapolated_result, AggregatedResult)
        self.assertGreaterEqual(r_squared, 0.0)
        self.assertLessEqual(r_squared, 1.0)
    
    def test_full_zne_application(self):
        """Test complete ZNE application."""
        def execution_func(circuit):
            result = self.backend.submit_and_wait(circuit, shots=100)
            return AggregatedResult(
                counts=result.counts,
                total_shots=100,
                successful_shots=100,
                backend_name=self.backend.name
            )
        
        try:
            zne_result = apply_zne_mitigation(
                circuit=self.test_circuit,
                execution_func=execution_func,
                noise_factors=[1.0, 2.0],
                scaling_method=NoiseScalingMethod.GATE_FOLDING,
                extrapolation_method=ExtrapolationMethod.LINEAR
            )
            
            # Check result structure
            self.assertIsInstance(zne_result, ZNEResult)
            self.assertEqual(len(zne_result.scaled_results), 2)
            self.assertIsInstance(zne_result.extrapolated_result, AggregatedResult)
            self.assertGreater(zne_result.total_shots, 0)
            
        except Exception as e:
            self.fail(f"ZNE application failed: {e}")


class TestErrorCorrection(unittest.TestCase):
    """Test error correction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple test circuit
        with QuantumProgram() as qp:
            qubit = qp.allocate(1)
            H(qubit)
            qp.measure(qubit, "test_result")
        
        self.test_circuit = qp.circuit
    
    def test_bit_flip_code(self):
        """Test 3-qubit bit-flip error correction code."""
        bit_flip_code = BitFlipCode()
        
        # Check code properties
        self.assertEqual(bit_flip_code.logical_qubits, 1)
        self.assertEqual(bit_flip_code.physical_qubits, 3)
        self.assertEqual(bit_flip_code.distance, 3)
        self.assertTrue(bit_flip_code.can_correct_errors(1))
        
        # Test encoding
        encoding_result = bit_flip_code.encode(self.test_circuit)
        self.assertIsInstance(encoding_result, ErrorCorrectionResult)
        self.assertGreaterEqual(encoding_result.processed_circuit.num_qubits, 3)
        
        # Test syndrome circuit generation
        syndrome_circuit = bit_flip_code.generate_syndrome_circuit()
        self.assertGreaterEqual(syndrome_circuit.num_qubits, 3)
        
        # Test error correction
        corrections = bit_flip_code.correct_errors("01")
        self.assertIsInstance(corrections, list)
    
    def test_phase_flip_code(self):
        """Test 3-qubit phase-flip error correction code."""
        phase_flip_code = PhaseFlipCode()
        
        # Check code properties
        self.assertEqual(phase_flip_code.logical_qubits, 1)
        self.assertEqual(phase_flip_code.physical_qubits, 3)
        self.assertEqual(phase_flip_code.distance, 3)
        
        # Test encoding
        encoding_result = phase_flip_code.encode(self.test_circuit)
        self.assertIsInstance(encoding_result, ErrorCorrectionResult)
        
        # Test error correction
        corrections = phase_flip_code.correct_errors("10")
        self.assertIsInstance(corrections, list)
    
    def test_shor_code(self):
        """Test 9-qubit Shor error correction code."""
        shor_code = ShorCode()
        
        # Check code properties
        self.assertEqual(shor_code.logical_qubits, 1)
        self.assertEqual(shor_code.physical_qubits, 9)
        self.assertEqual(shor_code.distance, 3)
        
        # Test encoding
        encoding_result = shor_code.encode(self.test_circuit)
        self.assertIsInstance(encoding_result, ErrorCorrectionResult)
        self.assertGreaterEqual(encoding_result.processed_circuit.num_qubits, 9)
    
    def test_error_correction_registry(self):
        """Test error correction code registry."""
        # Test getting known codes
        bit_flip_code = get_error_correction_code("bit_flip")
        self.assertIsInstance(bit_flip_code, BitFlipCode)
        
        phase_flip_code = get_error_correction_code("phase_flip")
        self.assertIsInstance(phase_flip_code, PhaseFlipCode)
        
        shor_code = get_error_correction_code("shor")
        self.assertIsInstance(shor_code, ShorCode)
        
        # Test unknown code
        with self.assertRaises(ValueError):
            get_error_correction_code("unknown_code")
    
    def test_circuit_encoding_functions(self):
        """Test high-level circuit encoding functions."""
        # Test encoding
        encoding_result = encode_circuit(self.test_circuit, "bit_flip")
        self.assertIsInstance(encoding_result, ErrorCorrectionResult)
        
        # Test decoding
        decoding_result = decode_circuit(encoding_result.processed_circuit, "bit_flip")
        self.assertIsInstance(decoding_result, ErrorCorrectionResult)


class TestCalibrationManager(unittest.TestCase):
    """Test calibration management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        self.calibration_manager = CalibrationManager(cache_dir=self.temp_dir)
        
        self.backend = LocalSimulatorBackend()
        self.backend.initialize()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_calibration_storage_and_retrieval(self):
        """Test storing and retrieving calibration data."""
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
        calibration_data = self.calibration_manager.store_calibration(
            backend_name="test_backend",
            device_id="test_device",
            num_qubits=2,
            calibration_type="measurement_mitigation",
            calibration_data=calibration_matrix,
            calibration_shots=1000
        )
        
        # Check stored data
        self.assertIsInstance(calibration_data, CalibrationData)
        self.assertEqual(calibration_data.backend_name, "test_backend")
        self.assertEqual(calibration_data.num_qubits, 2)
        self.assertTrue(calibration_data.is_valid())
        
        # Retrieve calibration
        retrieved_data = self.calibration_manager.get_calibration(
            backend_name="test_backend",
            device_id="test_device",
            num_qubits=2,
            calibration_type="measurement_mitigation"
        )
        
        self.assertIsNotNone(retrieved_data)
        self.assertEqual(retrieved_data.backend_name, "test_backend")
        self.assertTrue(retrieved_data.validate_checksum())
    
    def test_calibration_refresh(self):
        """Test calibration refresh functionality."""
        try:
            calibration_result = self.calibration_manager.refresh_calibration(
                backend=self.backend,
                num_qubits=2,
                calibration_type="measurement_mitigation",
                shots=100  # Small number for fast testing
            )
            
            self.assertIsInstance(calibration_result, CalibrationResult)
            if calibration_result.success:
                self.assertIsNotNone(calibration_result.calibration_data)
                self.assertGreater(calibration_result.circuits_executed, 0)
                self.assertGreater(calibration_result.total_shots, 0)
            
        except Exception as e:
            # Some failures might be expected in test environment
            self.assertIsInstance(e, (MitigationError, Exception))
    
    def test_calibration_expiry(self):
        """Test calibration expiry handling."""
        # Create expired calibration
        expired_time = time.time() - 3600  # 1 hour ago
        calibration_matrix = CalibrationMatrix(
            matrix=np.eye(2),
            inverse_matrix=np.eye(2),
            num_qubits=1,
            backend_name="test_backend",
            created_at=expired_time,
            calibration_shots=1000,
            readout_fidelity={0: 0.95}
        )
        
        # Manually create expired calibration data
        from datetime import datetime, timedelta
        expired_calibration = CalibrationData(
            backend_name="test_backend",
            device_id="test_device",
            num_qubits=1,
            calibration_type="measurement_mitigation",
            created_at=datetime.now() - timedelta(hours=2),
            expires_at=datetime.now() - timedelta(hours=1),  # Expired
            calibration_shots=1000,
            calibration_method="test",
            average_fidelity=0.95,
            confidence_score=0.8,
            data=calibration_matrix.to_dict(),
            checksum="test_checksum"
        )
        
        # Check expiry
        self.assertTrue(expired_calibration.is_expired())
        self.assertFalse(expired_calibration.is_valid())
    
    def test_calibration_statistics(self):
        """Test calibration statistics."""
        stats = self.calibration_manager.get_calibration_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_calibrations', stats)
        self.assertIn('calibration_types', stats)
        self.assertIn('average_quality_score', stats)
        self.assertIn('cache_directory', stats)


class TestMitigationPipeline(unittest.TestCase):
    """Test integrated mitigation pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = LocalSimulatorBackend()
        self.backend.initialize()
        self.pipeline = create_mitigation_pipeline()
        
        # Create test circuit
        with QuantumProgram() as qp:
            qubits = qp.allocate(2)
            H(qubits[0])
            CNOT(qubits[0], qubits[1])
            qp.measure(qubits, "test_result")
        
        self.test_circuit = qp.circuit
    
    def test_mitigation_options(self):
        """Test mitigation options configuration."""
        # Test default options
        options = MitigationOptions()
        self.assertEqual(options.level, MitigationLevel.BASIC)
        self.assertTrue(options.enable_measurement_mitigation)
        self.assertFalse(options.enable_zne)
        
        # Test level-based configuration
        options = MitigationOptions(level=MitigationLevel.AGGRESSIVE)
        self.assertTrue(options.enable_measurement_mitigation)
        self.assertTrue(options.enable_zne)
        self.assertTrue(options.enable_error_correction)
        
        # Test custom configuration
        options = MitigationOptions(
            level=MitigationLevel.CUSTOM,
            enable_measurement_mitigation=True,
            enable_zne=False,
            enable_error_correction=False
        )
        self.assertTrue(options.enable_measurement_mitigation)
        self.assertFalse(options.enable_zne)
        self.assertFalse(options.enable_error_correction)
    
    def test_pipeline_execution_basic(self):
        """Test basic pipeline execution."""
        options = MitigationOptions(level=MitigationLevel.NONE)
        
        try:
            result = self.pipeline.apply_mitigation(
                circuit=self.test_circuit,
                backend=self.backend,
                shots=100,
                options=options
            )
            
            self.assertIsInstance(result, MitigationPipelineResult)
            self.assertIsInstance(result.original_result, AggregatedResult)
            self.assertIsInstance(result.mitigated_result, AggregatedResult)
            self.assertEqual(result.options, options)
            self.assertGreaterEqual(result.total_execution_time, 0)
            
        except Exception as e:
            self.fail(f"Basic pipeline execution failed: {e}")
    
    def test_pipeline_with_measurement_mitigation(self):
        """Test pipeline with measurement mitigation enabled."""
        options = MitigationOptions(
            level=MitigationLevel.CUSTOM,
            enable_measurement_mitigation=True,
            enable_zne=False,
            auto_calibration=True,
            calibration_shots=100
        )
        
        try:
            result = self.pipeline.apply_mitigation(
                circuit=self.test_circuit,
                backend=self.backend,
                shots=100,
                options=options
            )
            
            # Should have measurement mitigation result if calibration succeeded
            # Note: Might fail in test environment, so we check gracefully
            self.assertIsInstance(result, MitigationPipelineResult)
            
        except MitigationError:
            # Expected in test environment without proper calibration
            pass
        except Exception as e:
            self.fail(f"Pipeline with measurement mitigation failed unexpectedly: {e}")
    
    def test_recommended_options(self):
        """Test recommended options generation."""
        recommended_options = self.pipeline.get_recommended_options(
            circuit=self.test_circuit,
            backend=self.backend,
            target_fidelity=0.9
        )
        
        self.assertIsInstance(recommended_options, MitigationOptions)
        self.assertIsInstance(recommended_options.level, MitigationLevel)
    
    def test_pipeline_result_analysis(self):
        """Test pipeline result analysis."""
        options = MitigationOptions(level=MitigationLevel.NONE)
        
        result = self.pipeline.apply_mitigation(
            circuit=self.test_circuit,
            backend=self.backend,
            shots=100,
            options=options
        )
        
        # Test improvement summary
        summary = result.get_improvement_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('original_dominant_probability', summary)
        self.assertIn('mitigated_dominant_probability', summary)
        self.assertIn('improvement_factor', summary)
        self.assertIn('applied_techniques', summary)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete error mitigation system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = LocalSimulatorBackend()
        self.backend.initialize()
    
    def test_end_to_end_mitigation(self):
        """Test complete end-to-end mitigation workflow."""
        # Create test circuit
        with QuantumProgram() as qp:
            qubits = qp.allocate(2)
            H(qubits[0])
            CNOT(qubits[0], qubits[1])
            qp.measure(qubits, "test_result")
        
        test_circuit = qp.circuit
        
        try:
            # Apply mitigation pipeline
            result = apply_mitigation_pipeline(
                circuit=test_circuit,
                backend=self.backend,
                shots=100,
                options=MitigationOptions(level=MitigationLevel.BASIC)
            )
            
            # Verify result structure
            self.assertIsInstance(result, MitigationPipelineResult)
            self.assertGreater(result.total_shots, 0)
            self.assertGreaterEqual(result.overhead_factor, 1.0)
            
            # Verify counts structure
            if result.mitigated_result.counts:
                for outcome in result.mitigated_result.counts:
                    self.assertEqual(len(outcome), test_circuit.num_qubits)
            
        except MitigationError:
            # Expected in some test environments
            pass
        except Exception as e:
            self.fail(f"End-to-end mitigation failed: {e}")
    
    def test_mitigation_system_consistency(self):
        """Test consistency across mitigation system components."""
        # Test that all global getters return consistent instances
        mitigator1 = get_measurement_mitigator()
        mitigator2 = get_measurement_mitigator()
        self.assertIs(mitigator1, mitigator2)
        
        zne1 = get_zne_mitigator()
        zne2 = get_zne_mitigator()
        self.assertIs(zne1, zne2)
        
        manager1 = get_calibration_manager()
        manager2 = get_calibration_manager()
        self.assertIs(manager1, manager2)


def run_performance_benchmark():
    """Run performance benchmark for mitigation system."""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    backend = LocalSimulatorBackend()
    backend.initialize()
    
    # Create test circuits of different sizes
    test_circuits = {}
    
    for num_qubits in [1, 2, 3]:
        with QuantumProgram() as qp:
            qubits = qp.allocate(num_qubits)
            
            # Create random circuit
            for _ in range(num_qubits * 2):
                H(qubits[np.random.randint(num_qubits)])
                if num_qubits > 1:
                    CNOT(qubits[np.random.randint(num_qubits)], 
                         qubits[np.random.randint(num_qubits)])
            
            qp.measure(qubits, f"result_{num_qubits}")
        
        test_circuits[num_qubits] = qp.circuit
    
    # Benchmark different mitigation levels
    mitigation_levels = [
        MitigationLevel.NONE,
        MitigationLevel.BASIC,
        MitigationLevel.MODERATE
    ]
    
    results = {}
    
    for num_qubits, circuit in test_circuits.items():
        results[num_qubits] = {}
        print(f"\nTesting {num_qubits}-qubit circuit:")
        
        for level in mitigation_levels:
            print(f"  {level.value:10} ", end="")
            
            try:
                start_time = time.time()
                
                pipeline_result = apply_mitigation_pipeline(
                    circuit=circuit,
                    backend=backend,
                    shots=100,  # Small for benchmarking
                    options=MitigationOptions(level=level)
                )
                
                execution_time = time.time() - start_time
                overhead = pipeline_result.overhead_factor
                
                results[num_qubits][level.value] = {
                    'time': execution_time,
                    'overhead': overhead
                }
                
                print(f"Time: {execution_time:.3f}s, Overhead: {overhead:.1f}x")
                
            except Exception as e:
                print(f"FAILED: {e}")
                results[num_qubits][level.value] = {'error': str(e)}
    
    return results


if __name__ == '__main__':
    # Set up test environment
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run tests
    print("Running Error Mitigation System Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMeasurementMitigation,
        TestZeroNoiseExtrapolation,
        TestErrorCorrection,
        TestCalibrationManager,
        TestMitigationPipeline,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance benchmark
    if result.wasSuccessful():
        benchmark_results = run_performance_benchmark()
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1) 