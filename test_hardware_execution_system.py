#!/usr/bin/env python3
"""
Test Suite for Hardware Execution System

This test suite validates the Real Hardware Execution features including
Hardware Abstraction Layer (HAL) and job management.
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch

from quantum_platform import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, Y, Z, CNOT, RX, RY, RZ, measure
from quantum_platform.hardware import (
    LocalSimulatorBackend, 
    get_backend_registry,
    register_backend,
    get_job_manager,
    JobPriority,
    JobStatus,
    DeviceType
)
from quantum_platform.hardware.hal import DeviceInfo, QuantumHardwareBackend
from quantum_platform.hardware.transpilation import CircuitTranspiler, QubitMapping
from quantum_platform.errors import HardwareError, CompilationError


class TestHardwareAbstractionLayer(unittest.TestCase):
    """Test the Hardware Abstraction Layer (HAL)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = LocalSimulatorBackend("test_sim", max_qubits=5)
        self.backend.initialize()
    
    def test_backend_initialization(self):
        """Test backend initialization."""
        self.assertTrue(self.backend._initialized)
        self.assertEqual(self.backend.name, "test_sim")
        self.assertEqual(self.backend.provider, "local")
    
    def test_device_info(self):
        """Test device information retrieval."""
        device_info = self.backend.get_device_info()
        
        self.assertIsInstance(device_info, DeviceInfo)
        self.assertEqual(device_info.name, "test_sim")
        self.assertEqual(device_info.provider, "local")
        self.assertEqual(device_info.device_type, DeviceType.SIMULATOR)
        self.assertEqual(device_info.num_qubits, 5)
        self.assertTrue(device_info.simulator)
        self.assertTrue(device_info.operational)
        self.assertIn("h", device_info.basis_gates)
        self.assertIn("cnot", device_info.basis_gates)
    
    def test_circuit_validation(self):
        """Test circuit validation."""
        # Create valid circuit
        with QuantumProgram() as qp:
            qubits = qp.allocate(2)
            H(qubits[0])
            CNOT(qubits[0], qubits[1])
            measure(qubits)
        
        # Should pass validation
        self.assertTrue(self.backend.validate_circuit(qp.circuit))
        
        # Create invalid circuit (too many qubits)
        with QuantumProgram() as qp_invalid:
            qubits = qp_invalid.allocate(10)  # More than max_qubits=5
            H(qubits[0])
            measure(qubits)
        
        # Should fail validation
        with self.assertRaises(CompilationError):
            self.backend.validate_circuit(qp_invalid.circuit)
    
    def test_circuit_submission(self):
        """Test circuit submission."""
        with QuantumProgram() as qp:
            qubits = qp.allocate(2)
            H(qubits[0])
            CNOT(qubits[0], qubits[1])
            measure(qubits)
        
        job_handle = self.backend.submit_circuit(qp.circuit, shots=100)
        
        self.assertIsNotNone(job_handle)
        self.assertEqual(job_handle.backend_name, "test_sim")
        self.assertEqual(job_handle.shots, 100)
        self.assertIsNotNone(job_handle.job_id)
    
    def test_job_status_monitoring(self):
        """Test job status monitoring."""
        with QuantumProgram() as qp:
            qubits = qp.allocate(1)
            H(qubits[0])
            measure(qubits)
        
        job_handle = self.backend.submit_circuit(qp.circuit, shots=10)
        
        # Wait for completion
        max_wait = 5.0  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = self.backend.get_job_status(job_handle)
            if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                break
            time.sleep(0.1)
        
        final_status = self.backend.get_job_status(job_handle)
        self.assertIn(final_status, [JobStatus.COMPLETED, JobStatus.FAILED])
    
    def test_result_retrieval(self):
        """Test result retrieval."""
        with QuantumProgram() as qp:
            qubits = qp.allocate(1)
            H(qubits[0])
            measure(qubits)
        
        result = self.backend.submit_and_wait(qp.circuit, shots=100, timeout=10)
        
        self.assertEqual(result.status, JobStatus.COMPLETED)
        self.assertIsInstance(result.counts, dict)
        self.assertEqual(result.shots, 100)
        self.assertIsNotNone(result.execution_time)
        
        # Check that we have measurement results
        total_counts = sum(result.counts.values())
        self.assertEqual(total_counts, 100)


class TestBackendRegistry(unittest.TestCase):
    """Test the backend registry system."""
    
    def test_backend_registration(self):
        """Test backend registration and retrieval."""
        registry = get_backend_registry()
        
        # Register backend type
        register_backend(LocalSimulatorBackend, "TestSimulator")
        
        # Check registration
        backend_types = registry.list_backend_types()
        self.assertIn("TestSimulator", backend_types)
        
        # Create backend instance
        backend = registry.create_backend(
            "TestSimulator", "my_simulator", max_qubits=10
        )
        
        self.assertIsInstance(backend, LocalSimulatorBackend)
        self.assertEqual(backend.name, "my_simulator")
        self.assertTrue(backend._initialized)
        
        # Retrieve backend
        retrieved = registry.get_backend("my_simulator")
        self.assertEqual(backend, retrieved)
    
    def test_backend_listing(self):
        """Test backend listing functionality."""
        registry = get_backend_registry()
        
        # Create a few backends
        registry.create_backend("TestSimulator", "sim1", max_qubits=5)
        registry.create_backend("TestSimulator", "sim2", max_qubits=10)
        
        backends = registry.list_backends()
        self.assertIn("sim1", backends)
        self.assertIn("sim2", backends)


class TestJobManager(unittest.TestCase):
    """Test the job manager system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.job_manager = get_job_manager()
        self.backend = LocalSimulatorBackend("test_backend")
        self.backend.initialize()
        self.job_manager.register_backend("test_backend", self.backend)
        
        # Create test circuit
        with QuantumProgram() as qp:
            qubits = qp.allocate(1)
            H(qubits[0])
            measure(qubits)
        self.test_circuit = qp.circuit
    
    def test_job_submission(self):
        """Test job submission through job manager."""
        self.job_manager.start()
        
        try:
            job_id = self.job_manager.submit_job(
                self.test_circuit, "test_backend", shots=100,
                priority=JobPriority.HIGH
            )
            
            self.assertIsNotNone(job_id)
            
            # Check job status
            status = self.job_manager.get_job_status(job_id)
            self.assertIsInstance(status, JobStatus)
            
        finally:
            self.job_manager.stop()
    
    def test_job_priority_handling(self):
        """Test job priority handling."""
        self.job_manager.start()
        
        try:
            # Submit jobs with different priorities
            low_job = self.job_manager.submit_job(
                self.test_circuit, "test_backend", 
                priority=JobPriority.LOW
            )
            high_job = self.job_manager.submit_job(
                self.test_circuit, "test_backend",
                priority=JobPriority.HIGH
            )
            
            jobs = self.job_manager.list_jobs()
            self.assertEqual(len(jobs), 2)
            
        finally:
            self.job_manager.stop()
    
    def test_queue_statistics(self):
        """Test queue statistics."""
        stats = self.job_manager.get_queue_stats()
        
        self.assertIn("active_jobs", stats)
        self.assertIn("pending_jobs", stats)
        self.assertIn("completed_jobs", stats)
        self.assertIn("available_backends", stats)
        self.assertIn("test_backend", stats["available_backends"])


class TestErrorHandling(unittest.TestCase):
    """Test error handling in hardware execution."""
    
    def test_invalid_backend_error(self):
        """Test error handling for invalid backends."""
        job_manager = get_job_manager()
        
        with QuantumProgram() as qp:
            qubits = qp.allocate(1)
            H(qubits[0])
            measure(qubits)
        
        with self.assertRaises(HardwareError):
            job_manager.submit_job(qp.circuit, "nonexistent_backend")
    
    def test_circuit_validation_error(self):
        """Test circuit validation errors."""
        backend = LocalSimulatorBackend("limited", max_qubits=2)
        backend.initialize()
        
        # Create circuit that's too large
        with QuantumProgram() as qp:
            qubits = qp.allocate(5)  # More than max_qubits=2
            H(qubits[0])
            measure(qubits)
        
        with self.assertRaises(CompilationError):
            backend.validate_circuit(qp.circuit)


class TestIntegration(unittest.TestCase):
    """Integration tests for the hardware execution system."""
    
    def test_end_to_end_execution(self):
        """Test complete end-to-end execution workflow."""
        # Set up backend and job manager
        backend = LocalSimulatorBackend("integration_test")
        backend.initialize()
        
        job_manager = get_job_manager()
        job_manager.register_backend("integration_test", backend)
        job_manager.start()
        
        try:
            # Create circuit
            with QuantumProgram() as qp:
                qubits = qp.allocate(2)
                H(qubits[0])
                CNOT(qubits[0], qubits[1])
                measure(qubits)
            
            # Submit job
            job_id = job_manager.submit_job(
                qp.circuit, "integration_test", shots=100
            )
            
            # Wait for completion
            max_wait = 10.0
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status = job_manager.get_job_status(job_id)
                if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    break
                time.sleep(0.1)
            
            # Get result
            result = job_manager.get_job_result(job_id)
            
            self.assertIsNotNone(result)
            self.assertEqual(result.status, JobStatus.COMPLETED)
            self.assertIsInstance(result.counts, dict)
            self.assertEqual(sum(result.counts.values()), 100)
            
        finally:
            job_manager.stop()


def run_hardware_tests():
    """Run all hardware execution tests."""
    print("🧪 Running Hardware Execution System Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHardwareAbstractionLayer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBackendRegistry))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestJobManager))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🧪 Hardware Execution Tests Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n✅ Success rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_hardware_tests()
    exit(0 if success else 1) 