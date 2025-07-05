"""
Test Suite for Hybrid Quantum-Classical Orchestration System

This test suite verifies the functionality of the hybrid orchestration system
including parameter binding, optimization workflows, and multi-provider support.
"""

import unittest
import numpy as np
import time
from typing import Dict, Any

from quantum_platform.compiler.ir.circuit import QuantumCircuit
from quantum_platform.compiler.language.dsl import QuantumProgram
from quantum_platform.compiler.language.operations import H, RX, RY, RZ, CNOT, measure
from quantum_platform.hardware.backends import LocalSimulatorBackend
from quantum_platform.orchestration.hybrid_executor import (
    HybridExecutor, ExecutionContext, ExecutionMode, ParameterBinder,
    OptimizationLoop, execute_hybrid_algorithm
)
from quantum_platform.orchestration.optimizers import (
    ScipyOptimizer, GradientDescentOptimizer, OptimizerCallback
)
from quantum_platform.providers.provider_manager import (
    ProviderManager, get_provider_manager, switch_provider,
    get_active_backend, list_available_providers
)


class TestParameterBinding(unittest.TestCase):
    """Test parameter binding functionality."""
    
    def setUp(self):
        """Set up test circuit with parameters."""
        with QuantumProgram() as qp:
            q0 = qp.allocate_qubit("q0")
            q1 = qp.allocate_qubit("q1")
            
            RY(q0, "theta")
            RX(q1, "phi")
            CNOT(q0, q1)
            RZ(q1, "lambda")
            
            measure(q0)
            measure(q1)
        
        self.circuit = qp.circuit
        self.parameter_binder = ParameterBinder(self.circuit)
    
    def test_parameter_extraction(self):
        """Test parameter name extraction from circuit."""
        expected_params = ["lambda", "phi", "theta"]  # Sorted
        self.assertEqual(self.parameter_binder.parameter_names, expected_params)
    
    def test_parameter_binding(self):
        """Test parameter binding to circuit."""
        parameters = {"theta": np.pi/4, "phi": np.pi/3, "lambda": np.pi/2}
        
        bound_circuit = self.parameter_binder.bind_parameters(parameters)
        
        # Check that circuit is different from original
        self.assertIsNot(bound_circuit, self.circuit)
        self.assertEqual(bound_circuit.num_qubits, self.circuit.num_qubits)
        self.assertEqual(bound_circuit.num_operations, self.circuit.num_operations)
    
    def test_parameter_binding_with_missing_parameters(self):
        """Test parameter binding with missing parameters."""
        parameters = {"theta": np.pi/4, "phi": np.pi/3}  # Missing lambda
        
        with self.assertRaises(Exception):
            self.parameter_binder.bind_parameters(parameters)
    
    def test_parameter_binding_cache(self):
        """Test parameter binding caching."""
        parameters = {"theta": np.pi/4, "phi": np.pi/3, "lambda": np.pi/2}
        
        # First binding
        bound_circuit1 = self.parameter_binder.bind_parameters(parameters)
        
        # Second binding with same parameters should use cache
        bound_circuit2 = self.parameter_binder.bind_parameters(parameters)
        
        # Should be the same cached circuit
        self.assertIs(bound_circuit1, bound_circuit2)
    
    def test_cache_clear(self):
        """Test clearing parameter binding cache."""
        parameters = {"theta": np.pi/4, "phi": np.pi/3, "lambda": np.pi/2}
        
        # Bind parameters
        bound_circuit1 = self.parameter_binder.bind_parameters(parameters)
        
        # Clear cache
        self.parameter_binder.clear_cache()
        
        # Bind same parameters again
        bound_circuit2 = self.parameter_binder.bind_parameters(parameters)
        
        # Should be different objects (not cached)
        self.assertIsNot(bound_circuit1, bound_circuit2)


class TestHybridExecutor(unittest.TestCase):
    """Test hybrid executor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test circuit
        with QuantumProgram() as qp:
            q0 = qp.allocate_qubit("q0")
            q1 = qp.allocate_qubit("q1")
            
            RY(q0, "theta")
            CNOT(q0, q1)
            RZ(q1, "phi")
            
            measure(q0)
            measure(q1)
        
        self.circuit = qp.circuit
        
        # Create backend
        self.backend = LocalSimulatorBackend("test_backend", "local")
        self.backend.initialize()
        
        # Create execution context
        self.context = ExecutionContext(
            backend=self.backend,
            shots=100,  # Small number for faster tests
            enable_caching=True
        )
        
        # Create executor
        self.executor = HybridExecutor(self.context)
        self.executor.set_circuit(self.circuit)
    
    def test_executor_initialization(self):
        """Test executor initialization."""
        self.assertIsNotNone(self.executor.parameter_binder)
        self.assertEqual(self.executor.execution_count, 0)
        self.assertEqual(self.executor.total_time, 0.0)
    
    def test_circuit_execution(self):
        """Test basic circuit execution."""
        parameters = {"theta": np.pi/4, "phi": np.pi/6}
        
        result = self.executor.execute(parameters)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.parameters, parameters)
        self.assertIsInstance(result.expectation_value, float)
        self.assertFalse(result.cache_hit)
        self.assertEqual(result.shots_used, 100)
        self.assertEqual(result.backend_used, "test_backend")
    
    def test_execution_caching(self):
        """Test execution result caching."""
        parameters = {"theta": np.pi/4, "phi": np.pi/6}
        
        # First execution
        result1 = self.executor.execute(parameters)
        
        # Second execution with same parameters
        result2 = self.executor.execute(parameters)
        
        # Second should be cached
        self.assertFalse(result1.cache_hit)
        self.assertTrue(result2.cache_hit)
    
    def test_custom_expectation_operator(self):
        """Test custom expectation value operator."""
        parameters = {"theta": np.pi/4, "phi": np.pi/6}
        
        def custom_operator(counts):
            """Custom expectation operator that returns fixed value."""
            return 42.0
        
        result = self.executor.execute(parameters, custom_operator)
        
        self.assertEqual(result.expectation_value, 42.0)
    
    def test_execution_statistics(self):
        """Test execution statistics tracking."""
        parameters = {"theta": np.pi/4, "phi": np.pi/6}
        
        # Execute multiple times
        for i in range(3):
            self.executor.execute(parameters)
        
        stats = self.executor.get_statistics()
        
        self.assertEqual(stats['execution_count'], 3)
        self.assertGreater(stats['total_time'], 0)
        self.assertGreater(stats['average_time'], 0)
    
    def test_cache_clearing(self):
        """Test cache clearing."""
        parameters = {"theta": np.pi/4, "phi": np.pi/6}
        
        # Execute to populate cache
        result1 = self.executor.execute(parameters)
        
        # Clear cache
        self.executor.clear_cache()
        
        # Execute again
        result2 = self.executor.execute(parameters)
        
        # Both should be fresh executions
        self.assertFalse(result1.cache_hit)
        self.assertFalse(result2.cache_hit)


class TestOptimizers(unittest.TestCase):
    """Test classical optimizers."""
    
    def test_gradient_descent_optimizer(self):
        """Test gradient descent optimizer."""
        optimizer = GradientDescentOptimizer()
        optimizer.set_options(max_iterations=10, tolerance=1e-6)
        
        # Simple quadratic function
        def objective(params):
            x = params['x']
            return (x - 2)**2 + 1
        
        initial_params = {'x': 0.0}
        
        result = optimizer.minimize(objective, initial_params)
        
        self.assertIsNotNone(result)
        self.assertIn('x', result.optimal_parameters)
        self.assertLess(result.optimal_value, 2.0)  # Should improve from initial
    
    def test_scipy_optimizer(self):
        """Test SciPy optimizer (with fallback)."""
        optimizer = ScipyOptimizer(method="COBYLA")
        optimizer.set_options(max_iterations=20, tolerance=1e-6)
        
        # Simple quadratic function
        def objective(params):
            x = params['x']
            y = params['y']
            return (x - 1)**2 + (y - 2)**2
        
        initial_params = {'x': 0.0, 'y': 0.0}
        
        result = optimizer.minimize(objective, initial_params)
        
        self.assertIsNotNone(result)
        self.assertIn('x', result.optimal_parameters)
        self.assertIn('y', result.optimal_parameters)
        self.assertLess(result.optimal_value, 2.0)  # Should improve
    
    def test_optimizer_callback(self):
        """Test optimizer callback functionality."""
        call_count = 0
        
        class TestCallback(OptimizerCallback):
            def __call__(self, iteration, parameters, value, **kwargs):
                nonlocal call_count
                call_count += 1
                return iteration < 5  # Stop after 5 iterations
        
        optimizer = GradientDescentOptimizer()
        optimizer.set_callback(TestCallback())
        optimizer.set_options(max_iterations=20)
        
        def objective(params):
            return params['x']**2
        
        result = optimizer.minimize(objective, {'x': 10.0})
        
        self.assertGreater(call_count, 0)
        self.assertLessEqual(result.iterations, 6)  # Should stop early


class TestOptimizationLoop(unittest.TestCase):
    """Test optimization loop functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create simple test circuit
        with QuantumProgram() as qp:
            q0 = qp.allocate_qubit("q0")
            RY(q0, "theta")
            measure(q0)
        
        self.circuit = qp.circuit
        
        # Create backend and executor
        backend = LocalSimulatorBackend("test_backend", "local")
        backend.initialize()
        
        context = ExecutionContext(
            backend=backend,
            shots=100,
            enable_caching=True
        )
        
        self.executor = HybridExecutor(context)
        self.executor.set_circuit(self.circuit)
    
    def test_optimization_loop_creation(self):
        """Test optimization loop creation."""
        def objective(params):
            result = self.executor.execute(params)
            return result.expectation_value
        
        loop = self.executor.create_optimization_loop(objective)
        
        self.assertIsNotNone(loop)
        self.assertEqual(loop.executor, self.executor)
        self.assertEqual(loop.iteration, 0)
    
    def test_optimization_loop_execution(self):
        """Test optimization loop execution."""
        def objective(params):
            result = self.executor.execute(params)
            return result.expectation_value
        
        loop = self.executor.create_optimization_loop(objective)
        
        initial_params = {"theta": np.pi/4}
        
        result = loop.optimize(
            initial_params,
            max_iterations=5,
            tolerance=1e-6
        )
        
        self.assertIsNotNone(result)
        self.assertIn('best_parameters', result)
        self.assertIn('best_value', result)
        self.assertIn('iterations', result)
        self.assertIn('history', result)
        self.assertGreater(len(result['history']), 0)


class TestProviderManager(unittest.TestCase):
    """Test provider manager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary provider manager
        self.provider_manager = ProviderManager()
    
    def test_provider_manager_initialization(self):
        """Test provider manager initialization."""
        self.assertIsNotNone(self.provider_manager.providers)
        self.assertIn("local", self.provider_manager.providers)
        self.assertEqual(self.provider_manager.active_provider, "local")
    
    def test_list_providers(self):
        """Test listing available providers."""
        providers = self.provider_manager.list_providers()
        
        self.assertIsInstance(providers, dict)
        self.assertIn("local", providers)
        self.assertIn("ibm", providers)
        self.assertIn("aws", providers)
    
    def test_list_devices(self):
        """Test listing available devices."""
        devices = self.provider_manager.list_devices()
        
        self.assertIsInstance(devices, list)
        self.assertGreater(len(devices), 0)
        
        # Check local simulator is available
        local_devices = [d for d in devices if d.provider == "local"]
        self.assertGreater(len(local_devices), 0)
    
    def test_get_provider_info(self):
        """Test getting provider information."""
        local_info = self.provider_manager.get_provider_info("local")
        
        self.assertIsNotNone(local_info)
        self.assertEqual(local_info.name, "local")
        self.assertEqual(local_info.provider_type, "simulator")
    
    def test_set_active_provider(self):
        """Test setting active provider."""
        # Should start with local
        self.assertEqual(self.provider_manager.active_provider, "local")
        
        # Try to set to local again (should work)
        self.provider_manager.set_active_provider("local")
        self.assertEqual(self.provider_manager.active_provider, "local")
        
        # Try to set to invalid provider (should fail)
        with self.assertRaises(Exception):
            self.provider_manager.set_active_provider("invalid_provider")
    
    def test_get_active_backend(self):
        """Test getting active backend."""
        backend = self.provider_manager.get_active_backend()
        
        self.assertIsNotNone(backend)
        self.assertEqual(backend.name, "local_simulator")
        self.assertEqual(backend.provider, "local")
    
    def test_status_summary(self):
        """Test getting status summary."""
        summary = self.provider_manager.get_status_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn("active_provider", summary)
        self.assertIn("active_device", summary)
        self.assertIn("providers", summary)
        self.assertIn("total_devices", summary)
        self.assertIn("available_devices", summary)
        
        self.assertEqual(summary["active_provider"], "local")


class TestIntegration(unittest.TestCase):
    """Integration tests for hybrid orchestration system."""
    
    def test_end_to_end_vqe_workflow(self):
        """Test complete VQE workflow."""
        # Create parameterized circuit
        with QuantumProgram() as qp:
            q0 = qp.allocate_qubit("q0")
            q1 = qp.allocate_qubit("q1")
            
            RY(q0, "theta1")
            RY(q1, "theta2")
            CNOT(q0, q1)
            RZ(q1, "phi")
            
            measure(q0)
            measure(q1)
        
        circuit = qp.circuit
        
        # Set up provider
        provider_manager = get_provider_manager()
        backend = provider_manager.get_active_backend()
        
        # Create execution context
        context = ExecutionContext(
            backend=backend,
            shots=200,
            enable_caching=True
        )
        
        # Create executor
        executor = HybridExecutor(context)
        executor.set_circuit(circuit)
        
        # Define objective function
        def objective(params):
            result = executor.execute(params)
            return result.expectation_value
        
        # Create optimizer
        optimizer = GradientDescentOptimizer()
        optimizer.set_options(max_iterations=10, tolerance=1e-6)
        
        # Initial parameters
        initial_params = {
            "theta1": np.pi/4,
            "theta2": np.pi/3,
            "phi": np.pi/6
        }
        
        # Run optimization
        result = optimizer.minimize(objective, initial_params)
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIn("theta1", result.optimal_parameters)
        self.assertIn("theta2", result.optimal_parameters)
        self.assertIn("phi", result.optimal_parameters)
        self.assertGreater(result.function_evaluations, 0)
    
    def test_provider_switching_workflow(self):
        """Test switching between providers."""
        # Create simple test circuit
        with QuantumProgram() as qp:
            q0 = qp.allocate_qubit("q0")
            RY(q0, "theta")
            measure(q0)
        
        circuit = qp.circuit
        parameters = {"theta": np.pi/4}
        
        # Test local provider
        provider_manager = get_provider_manager()
        provider_manager.set_active_provider("local")
        
        backend = provider_manager.get_active_backend()
        self.assertIsNotNone(backend)
        
        # Execute circuit
        result = execute_hybrid_algorithm(
            circuit=circuit,
            parameters=parameters,
            backend=backend,
            shots=100
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.backend_used, "local_simulator")
        self.assertEqual(result.shots_used, 100)
    
    def test_multi_parameter_optimization(self):
        """Test optimization with multiple parameters."""
        # Create circuit with many parameters
        with QuantumProgram() as qp:
            qubits = [qp.allocate_qubit(f"q{i}") for i in range(3)]
            
            # Layer 1
            for i, qubit in enumerate(qubits):
                RY(qubit, f"theta1_{i}")
            
            # Entangling layer
            CNOT(qubits[0], qubits[1])
            CNOT(qubits[1], qubits[2])
            
            # Layer 2
            for i, qubit in enumerate(qubits):
                RZ(qubit, f"phi1_{i}")
            
            # Measurements
            for qubit in qubits:
                measure(qubit)
        
        circuit = qp.circuit
        
        # Set up executor
        backend = get_provider_manager().get_active_backend()
        context = ExecutionContext(backend=backend, shots=100)
        executor = HybridExecutor(context)
        executor.set_circuit(circuit)
        
        # Create parameters
        initial_params = {}
        for i in range(3):
            initial_params[f"theta1_{i}"] = np.random.uniform(0, 2*np.pi)
            initial_params[f"phi1_{i}"] = np.random.uniform(0, 2*np.pi)
        
        # Test single execution
        result = executor.execute(initial_params)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result.parameters), 6)  # 3 theta + 3 phi
        self.assertEqual(result.shots_used, 100)


class TestPerformance(unittest.TestCase):
    """Performance tests for hybrid orchestration."""
    
    def test_execution_performance(self):
        """Test execution performance and caching."""
        # Create test circuit
        with QuantumProgram() as qp:
            q0 = qp.allocate_qubit("q0")
            RY(q0, "theta")
            measure(q0)
        
        circuit = qp.circuit
        
        # Set up executor
        backend = get_provider_manager().get_active_backend()
        context = ExecutionContext(
            backend=backend,
            shots=50,  # Small for speed
            enable_caching=True
        )
        executor = HybridExecutor(context)
        executor.set_circuit(circuit)
        
        # Time first execution
        parameters = {"theta": np.pi/4}
        
        start_time = time.time()
        result1 = executor.execute(parameters)
        first_time = time.time() - start_time
        
        # Time second execution (should be cached)
        start_time = time.time()
        result2 = executor.execute(parameters)
        second_time = time.time() - start_time
        
        # Verify caching worked
        self.assertFalse(result1.cache_hit)
        self.assertTrue(result2.cache_hit)
        
        # Cached execution should be much faster
        self.assertLess(second_time, first_time / 2)
    
    def test_parameter_binding_performance(self):
        """Test parameter binding performance."""
        # Create circuit with many parameters
        with QuantumProgram() as qp:
            qubits = [qp.allocate_qubit(f"q{i}") for i in range(4)]
            
            for layer in range(3):
                for i, qubit in enumerate(qubits):
                    RY(qubit, f"theta_{layer}_{i}")
                    RZ(qubit, f"phi_{layer}_{i}")
                
                for i in range(len(qubits) - 1):
                    CNOT(qubits[i], qubits[i + 1])
            
            for qubit in qubits:
                measure(qubit)
        
        circuit = qp.circuit
        binder = ParameterBinder(circuit)
        
        # Create parameters
        parameters = {}
        for layer in range(3):
            for i in range(4):
                parameters[f"theta_{layer}_{i}"] = np.random.uniform(0, 2*np.pi)
                parameters[f"phi_{layer}_{i}"] = np.random.uniform(0, 2*np.pi)
        
        # Time parameter binding
        start_time = time.time()
        bound_circuit = binder.bind_parameters(parameters)
        binding_time = time.time() - start_time
        
        # Verify binding worked
        self.assertIsNotNone(bound_circuit)
        self.assertEqual(bound_circuit.num_qubits, circuit.num_qubits)
        
        # Binding should be reasonably fast
        self.assertLess(binding_time, 1.0)  # Should complete in under 1 second


def run_all_tests():
    """Run all tests."""
    print("Running Hybrid Orchestration System Tests...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestParameterBinding,
        TestHybridExecutor,
        TestOptimizers,
        TestOptimizationLoop,
        TestProviderManager,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!") 