#!/usr/bin/env python3
"""
Comprehensive Tests for Quantum Noise Model System

This test suite validates all aspects of the noise modeling capabilities
including noise models, noisy simulation, and backend integration.
"""

import unittest
import numpy as np
import time
from unittest.mock import patch, MagicMock

# Import components to test
from quantum_platform.simulation.noise_models import (
    NoiseModel, NoiseParameter, GateErrorRates, CoherenceParameters,
    ReadoutError, DeviceNoiseModelLibrary, get_noise_library,
    create_noise_model_from_calibration
)
from quantum_platform.simulation.noisy_simulator import (
    NoisyQuantumSimulator, NoiseSimulationEngine, create_device_simulator
)
from quantum_platform.hardware.backends.noisy_simulator_backend import NoisySimulatorBackend
from quantum_platform.compiler.language.dsl import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, CNOT, measure
from quantum_platform.hardware.hal import JobStatus


class TestNoiseModels(unittest.TestCase):
    """Test noise model creation and configuration."""
    
    def test_noise_parameter_creation(self):
        """Test NoiseParameter creation and validation."""
        param = NoiseParameter(50.0, "us", "T1 time", confidence=0.95)
        
        self.assertEqual(param.value, 50.0)
        self.assertEqual(param.unit, "us")
        self.assertEqual(param.confidence, 0.95)
        
        # Test invalid confidence
        with self.assertRaises(ValueError):
            NoiseParameter(50.0, confidence=1.5)
    
    def test_gate_error_rates(self):
        """Test gate error rate handling."""
        gate_errors = GateErrorRates(
            single_qubit_error=1e-3,
            two_qubit_error=1e-2,
            measurement_error=2e-2
        )
        
        # Test default rates
        self.assertEqual(gate_errors.get_error_rate("h", 1), 1e-3)
        self.assertEqual(gate_errors.get_error_rate("cx", 2), 1e-2)
        self.assertEqual(gate_errors.get_error_rate("measure", 1), 2e-2)
        
        # Test gate-specific rates
        gate_errors.gate_specific["custom_gate"] = 5e-3
        self.assertEqual(gate_errors.get_error_rate("custom_gate", 1), 5e-3)
    
    def test_coherence_parameters(self):
        """Test coherence parameter calculations."""
        T1 = NoiseParameter(100.0, "us")
        T2 = NoiseParameter(50.0, "us")
        
        coherence = CoherenceParameters(T1, T2)
        
        # Test T2* calculation
        expected_t2_star = 1 / (1/50.0 - 1/(2*100.0))
        self.assertAlmostEqual(coherence.T2_star, expected_t2_star, places=3)
    
    def test_readout_error(self):
        """Test readout error functionality."""
        readout = ReadoutError(prob_0_given_1=0.02, prob_1_given_0=0.01)
        
        confusion_matrix = readout.get_confusion_matrix()
        
        # Check matrix structure
        self.assertEqual(confusion_matrix.shape, (2, 2))
        self.assertAlmostEqual(confusion_matrix[0, 0], 0.99)  # P(0|0)
        self.assertAlmostEqual(confusion_matrix[1, 1], 0.98)  # P(1|1)
        self.assertAlmostEqual(confusion_matrix[0, 1], 0.01)  # P(1|0)
        self.assertAlmostEqual(confusion_matrix[1, 0], 0.02)  # P(0|1)
    
    def test_noise_model_creation(self):
        """Test basic noise model creation and configuration."""
        model = NoiseModel("test_model", "Test noise model")
        
        self.assertEqual(model.name, "test_model")
        self.assertTrue(model.enabled)
        
        # Set qubit parameters
        model.set_qubit_coherence(0, T1=100, T2=50)
        model.set_qubit_readout_error(0, 0.02, 0.01)
        
        self.assertIn(0, model.coherence_params)
        self.assertIn(0, model.readout_errors)
        
        # Test gate error setting
        model.set_gate_error_rate("h", 1e-4)
        self.assertEqual(model.gate_errors.gate_specific["h"], 1e-4)
    
    def test_noise_model_serialization(self):
        """Test noise model to/from dictionary conversion."""
        model = NoiseModel("test_model", "Test model")
        model.set_qubit_coherence(0, T1=100, T2=50)
        model.set_qubit_readout_error(0, 0.02, 0.01)
        model.set_gate_error_rate("h", 1e-4)
        
        # Convert to dictionary
        model_dict = model.to_dict()
        
        # Verify structure
        self.assertIn("name", model_dict)
        self.assertIn("coherence_params", model_dict)
        self.assertIn("readout_errors", model_dict)
        self.assertIn("gate_errors", model_dict)
        
        # Convert back from dictionary
        restored_model = NoiseModel.from_dict(model_dict)
        
        self.assertEqual(restored_model.name, model.name)
        self.assertEqual(len(restored_model.coherence_params), 1)
        self.assertEqual(len(restored_model.readout_errors), 1)
    
    def test_calibration_data_import(self):
        """Test creating noise model from calibration data."""
        calibration_data = {
            "date": "2024-01-15",
            "qubits": [
                {
                    "id": 0,
                    "T1": 85.3,
                    "T2": 42.1,
                    "readout_error": {"prob_0_given_1": 0.025, "prob_1_given_0": 0.015}
                }
            ],
            "gates": {
                "single_qubit": 0.0012,
                "cx": 0.0089
            }
        }
        
        model = create_noise_model_from_calibration("Test_Device", calibration_data)
        
        self.assertEqual(model.device_name, "Test_Device")
        self.assertEqual(model.calibration_date, "2024-01-15")
        self.assertIn(0, model.coherence_params)
        self.assertEqual(model.coherence_params[0].T1.value, 85.3)


class TestNoiseLibrary(unittest.TestCase):
    """Test noise model library functionality."""
    
    def test_default_models(self):
        """Test that default noise models are loaded."""
        library = get_noise_library()
        
        models = library.list_models()
        expected_models = ["ideal", "ibm_like", "ionq_like", "google_like"]
        
        for expected in expected_models:
            self.assertIn(expected, models)
    
    def test_model_retrieval(self):
        """Test retrieving models from library."""
        library = get_noise_library()
        
        ibm_model = library.get_model("ibm_like")
        self.assertIsNotNone(ibm_model)
        self.assertEqual(ibm_model.name, "ibm_like")
        
        # Test non-existent model
        fake_model = library.get_model("non_existent")
        self.assertIsNone(fake_model)
    
    def test_custom_model_management(self):
        """Test adding and removing custom models."""
        library = DeviceNoiseModelLibrary()  # Fresh instance
        
        # Add custom model
        custom_model = NoiseModel("custom", "Custom test model")
        library.add_model(custom_model)
        
        self.assertIn("custom", library.list_models())
        
        # Remove model
        library.remove_model("custom")
        self.assertNotIn("custom", library.list_models())


class TestNoisySimulator(unittest.TestCase):
    """Test noisy quantum simulator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test circuit
        with QuantumProgram() as qp:
            q0 = qp.allocate_qubit("q0")
            q1 = qp.allocate_qubit("q1")
            
            H(q0)
            CNOT(q0, q1)
            
            measure(q0)
            measure(q1)
        
        self.test_circuit = qp.circuit
        
        # Create test noise model
        self.noise_model = NoiseModel("test", "Test noise model")
        self.noise_model.set_qubit_readout_error(0, 0.1, 0.05)
        self.noise_model.set_qubit_readout_error(1, 0.1, 0.05)
        self.noise_model.gate_errors.single_qubit_error = 0.01
        self.noise_model.gate_errors.two_qubit_error = 0.02
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        simulator = NoisyQuantumSimulator(self.noise_model)
        
        self.assertEqual(simulator.noise_model, self.noise_model)
        self.assertIsNotNone(simulator.ideal_simulator)
    
    def test_ideal_simulation(self):
        """Test simulation without noise."""
        simulator = NoisyQuantumSimulator(noise_model=None)
        
        result = simulator.run(self.test_circuit, shots=100)
        
        self.assertIsNotNone(result.counts)
        self.assertEqual(result.shots, 100)
        self.assertEqual(result.noise_model_name, "none")
        self.assertFalse(hasattr(result, 'ideal_counts') and result.ideal_counts)
    
    def test_noisy_simulation(self):
        """Test simulation with noise."""
        simulator = NoisyQuantumSimulator(self.noise_model)
        
        result = simulator.run(self.test_circuit, shots=100, compare_ideal=True)
        
        self.assertIsNotNone(result.counts)
        self.assertEqual(result.shots, 100)
        self.assertEqual(result.noise_model_name, "test")
        self.assertIsNotNone(result.ideal_counts)
        self.assertGreaterEqual(result.noise_overhead, 0.0)
    
    def test_noise_simulation_engine(self):
        """Test noise simulation engine components."""
        engine = NoiseSimulationEngine(self.noise_model)
        
        # Test measurement error application
        original_bitstring = "00"
        noisy_bitstring = engine.apply_measurement_error(original_bitstring)
        
        # Should be a valid bitstring
        self.assertEqual(len(noisy_bitstring), 2)
        self.assertTrue(all(c in '01' for c in noisy_bitstring))
    
    def test_device_simulator_factory(self):
        """Test device simulator factory functions."""
        # Test known device types
        for device_type in ["ibm_like", "ionq_like", "google_like", "ideal"]:
            simulator = create_device_simulator(device_type)
            self.assertIsInstance(simulator, NoisyQuantumSimulator)
        
        # Test unknown device type
        with self.assertRaises(ValueError):
            create_device_simulator("unknown_device")
    
    def test_comparative_analysis(self):
        """Test comparative analysis functionality."""
        simulator = NoisyQuantumSimulator(self.noise_model)
        
        analysis = simulator.run_comparative_analysis(self.test_circuit, shots=100)
        
        self.assertIn('ideal_counts', analysis)
        self.assertIn('noisy_counts', analysis)
        self.assertIn('fidelity', analysis)
        self.assertIn('hellinger_distance', analysis)
        self.assertIn('total_variation_distance', analysis)
        
        # Fidelity should be between 0 and 1
        self.assertGreaterEqual(analysis['fidelity'], 0.0)
        self.assertLessEqual(analysis['fidelity'], 1.0)
    
    def test_random_seed_reproducibility(self):
        """Test that random seed makes noise simulation reproducible."""
        simulator = NoisyQuantumSimulator(self.noise_model)
        
        # Run with same seed twice
        simulator.set_random_seed(42)
        result1 = simulator.run(self.test_circuit, shots=50)
        
        simulator.set_random_seed(42)
        result2 = simulator.run(self.test_circuit, shots=50)
        
        # Results should be identical
        self.assertEqual(result1.counts, result2.counts)


class TestNoisySimulatorBackend(unittest.TestCase):
    """Test noisy simulator backend integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.backend = NoisySimulatorBackend("test_noisy", "ibm_like", max_qubits=5)
        
        # Create test circuit
        with QuantumProgram() as qp:
            q0 = qp.allocate_qubit("q0")
            H(q0)
            measure(q0)
        
        self.test_circuit = qp.circuit
    
    def test_backend_initialization(self):
        """Test backend initialization."""
        success = self.backend.initialize()
        self.assertTrue(success)
        self.assertIsNotNone(self.backend.simulator)
        self.assertIsNotNone(self.backend.noise_model)
    
    def test_device_info(self):
        """Test device information retrieval."""
        self.backend.initialize()
        device_info = self.backend.get_device_info()
        
        self.assertEqual(device_info.name, "test_noisy")
        self.assertEqual(device_info.provider, "local")
        self.assertTrue(device_info.simulator)
        self.assertIn("noisy_monte_carlo", device_info.metadata.get("simulator_type", ""))
    
    def test_circuit_validation(self):
        """Test circuit validation."""
        self.backend.initialize()
        
        # Valid circuit should pass
        self.assertTrue(self.backend.validate_circuit(self.test_circuit))
        
        # Circuit too large should fail
        with QuantumProgram() as qp:
            qubits = [qp.allocate_qubit(f"q{i}") for i in range(50)]  # Too many qubits
            for q in qubits:
                H(q)
                measure(q)
        
        large_circuit = qp.circuit
        
        with self.assertRaises(Exception):
            self.backend.validate_circuit(large_circuit)
    
    def test_job_submission_and_execution(self):
        """Test job submission and execution."""
        self.backend.initialize()
        
        # Submit job
        job_handle = self.backend.submit_circuit(self.test_circuit, shots=50)
        
        self.assertIsNotNone(job_handle)
        self.assertEqual(job_handle.backend_name, "test_noisy")
        
        # Wait for completion
        max_wait = 10  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = self.backend.get_job_status(job_handle)
            if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                break
            time.sleep(0.1)
        
        # Retrieve results
        result = self.backend.retrieve_results(job_handle)
        
        self.assertEqual(result.status, JobStatus.COMPLETED)
        self.assertIsNotNone(result.counts)
        self.assertEqual(result.shots, 50)
        self.assertTrue(result.metadata.get("noisy_simulation", False))
    
    def test_noise_characteristics(self):
        """Test noise characteristics reporting."""
        self.backend.initialize()
        
        characteristics = self.backend.get_noise_characteristics()
        
        self.assertIn("noise_enabled", characteristics)
        self.assertIn("noise_model_name", characteristics)
        self.assertIn("coherence_params", characteristics)
        self.assertIn("gate_errors", characteristics)
    
    def test_noise_toggle(self):
        """Test noise enable/disable functionality."""
        self.backend.initialize()
        
        # Initially enabled
        self.assertTrue(self.backend.noise_model.enabled)
        
        # Disable noise
        self.backend.toggle_noise(False)
        self.assertFalse(self.backend.noise_model.enabled)
        
        # Re-enable noise
        self.backend.toggle_noise(True)
        self.assertTrue(self.backend.noise_model.enabled)
    
    def test_job_cancellation(self):
        """Test job cancellation."""
        self.backend.initialize()
        
        # Submit job
        job_handle = self.backend.submit_circuit(self.test_circuit, shots=100)
        
        # Try to cancel immediately
        cancelled = self.backend.cancel_job(job_handle)
        
        # Should either be cancelled or already completed
        final_status = self.backend.get_job_status(job_handle)
        self.assertIn(final_status, [JobStatus.CANCELLED, JobStatus.COMPLETED])


class TestIntegration(unittest.TestCase):
    """Test integration between noise components."""
    
    def test_end_to_end_noisy_simulation(self):
        """Test complete end-to-end noisy simulation workflow."""
        # Create circuit
        with QuantumProgram() as qp:
            qubits = [qp.allocate_qubit(f"q{i}") for i in range(2)]
            
            # Create entangled state
            H(qubits[0])
            CNOT(qubits[0], qubits[1])
            
            # Measure
            for q in qubits:
                measure(q)
        
        circuit = qp.circuit
        
        # Create backend
        backend = NoisySimulatorBackend("integration_test", "ibm_like")
        backend.initialize()
        
        # Submit job with noise comparison
        job_handle = backend.submit_circuit(
            circuit, 
            shots=200, 
            compare_ideal=True,
            noise_enabled=True
        )
        
        # Wait for completion
        while backend.get_job_status(job_handle) not in [JobStatus.COMPLETED, JobStatus.FAILED]:
            time.sleep(0.1)
        
        # Get results
        result = backend.retrieve_results(job_handle)
        
        # Verify results
        self.assertEqual(result.status, JobStatus.COMPLETED)
        self.assertIsNotNone(result.counts)
        self.assertTrue(result.metadata.get("noise_enabled", False))
        
        # Should have ideal counts for comparison
        if "ideal_counts" in result.metadata:
            self.assertIsNotNone(result.metadata["ideal_counts"])
    
    def test_noise_model_library_integration(self):
        """Test integration with noise model library."""
        library = get_noise_library()
        
        # Test all default models can create working simulators
        for model_name in library.list_models():
            if model_name == "ideal":
                continue  # Skip ideal model
            
            try:
                simulator = create_device_simulator(model_name)
                self.assertIsInstance(simulator, NoisyQuantumSimulator)
                
                # Test basic functionality
                with QuantumProgram() as qp:
                    q = qp.allocate_qubit("q")
                    H(q)
                    measure(q)
                
                result = simulator.run(qp.circuit, shots=10)
                self.assertIsNotNone(result.counts)
                
            except Exception as e:
                self.fail(f"Failed to create/test simulator for {model_name}: {e}")


def run_performance_benchmarks():
    """Run performance benchmarks for noise simulation."""
    print("\n🏃 Running Performance Benchmarks")
    print("=" * 40)
    
    # Create test circuit
    with QuantumProgram() as qp:
        qubits = [qp.allocate_qubit(f"q{i}") for i in range(3)]
        for q in qubits:
            H(q)
        for i in range(len(qubits) - 1):
            CNOT(qubits[i], qubits[i + 1])
        for q in qubits:
            measure(q)
    
    circuit = qp.circuit
    
    # Test different shot counts
    shot_counts = [100, 500, 1000, 2000]
    
    print(f"Circuit: {circuit.num_qubits} qubits, {len(circuit.operations)} operations")
    
    for shots in shot_counts:
        # Ideal simulation
        simulator_ideal = NoisyQuantumSimulator(noise_model=None)
        start_time = time.time()
        result_ideal = simulator_ideal.run(circuit, shots=shots)
        ideal_time = time.time() - start_time
        
        # Noisy simulation
        simulator_noisy = create_device_simulator("ibm_like")
        start_time = time.time()
        result_noisy = simulator_noisy.run(circuit, shots=shots)
        noisy_time = time.time() - start_time
        
        overhead_ratio = noisy_time / ideal_time if ideal_time > 0 else float('inf')
        
        print(f"Shots: {shots:4d} | Ideal: {ideal_time:.3f}s | "
              f"Noisy: {noisy_time:.3f}s | Overhead: {overhead_ratio:.2f}x")


def main():
    """Run all tests."""
    print("🧪 Running Quantum Noise Model System Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestNoiseModels,
        TestNoiseLibrary,
        TestNoisySimulator,
        TestNoisySimulatorBackend,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split(chr(10))[-2]}")
    
    # Run performance benchmarks if tests passed
    if not result.failures and not result.errors:
        run_performance_benchmarks()
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 