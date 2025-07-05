#!/usr/bin/env python3
"""
Test Suite for Multi-Shot Execution and Result Aggregation

This test suite validates the multi-shot execution capabilities including
result aggregation, analysis, storage, and statistical functionality.
"""

import unittest
import tempfile
import os
import numpy as np
from collections import Counter

from quantum_platform import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, Y, Z, CNOT, RX, measure
from quantum_platform.hardware.results import (
    AggregatedResult, ShotResult, ResultAggregator, ResultAnalyzer,
    ResultStorage, MultiShotExecutor
)
from quantum_platform.hardware.hal import HardwareResult, JobHandle, JobStatus
from quantum_platform.simulation.statevector import StateVectorSimulator
from quantum_platform.errors import ExecutionError


class TestShotResult(unittest.TestCase):
    """Test ShotResult data structure."""
    
    def test_shot_result_creation(self):
        """Test ShotResult creation and attributes."""
        shot = ShotResult(
            shot_id=1,
            outcome="01",
            execution_time=150.5,
            metadata={"qubit_count": 2}
        )
        
        self.assertEqual(shot.shot_id, 1)
        self.assertEqual(shot.outcome, "01")
        self.assertEqual(shot.execution_time, 150.5)
        self.assertEqual(shot.metadata["qubit_count"], 2)
    
    def test_shot_result_defaults(self):
        """Test ShotResult with default values."""
        shot = ShotResult(shot_id=0, outcome="00")
        
        self.assertIsNone(shot.measurement_data)
        self.assertIsNone(shot.execution_time)
        self.assertEqual(shot.metadata, {})


class TestAggregatedResult(unittest.TestCase):
    """Test AggregatedResult data structure and methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_counts = {"00": 400, "01": 100, "10": 100, "11": 400}
        self.sample_result = AggregatedResult(
            total_shots=1000,
            successful_shots=1000,
            counts=self.sample_counts,
            circuit_id="test_circuit",
            backend_name="test_backend"
        )
    
    def test_aggregated_result_creation(self):
        """Test AggregatedResult creation and statistics calculation."""
        result = self.sample_result
        
        self.assertEqual(result.total_shots, 1000)
        self.assertEqual(result.successful_shots, 1000)
        self.assertEqual(result.failed_shots, 0)
        self.assertEqual(result.unique_outcomes, 4)
        
        # Check statistics were calculated
        self.assertIsNotNone(result.probabilities)
        self.assertIsNotNone(result.most_frequent)
        self.assertIsNotNone(result.entropy)
        
        # Check probabilities
        self.assertAlmostEqual(result.probabilities["00"], 0.4)
        self.assertAlmostEqual(result.probabilities["11"], 0.4)
        self.assertAlmostEqual(result.probabilities["01"], 0.1)
        
        # Check most/least frequent
        self.assertIn(result.most_frequent, ["00", "11"])  # Both have max count
        self.assertIn(result.least_frequent, ["01", "10"])  # Both have min count
    
    def test_get_top_outcomes(self):
        """Test getting top outcomes."""
        top_outcomes = self.sample_result.get_top_outcomes(3)
        
        self.assertEqual(len(top_outcomes), 3)
        
        # Check format: (outcome, count, probability)
        outcome, count, prob = top_outcomes[0]
        self.assertIn(outcome, ["00", "11"])  # Should be one of the most frequent
        self.assertIn(count, [400])  # Count should be 400
        self.assertAlmostEqual(prob, 0.4)  # Probability should be 0.4
    
    def test_filter_by_probability(self):
        """Test filtering by minimum probability."""
        filtered = self.sample_result.filter_by_probability(min_prob=0.15)
        
        # Should only include outcomes with >= 15% probability
        self.assertIn("00", filtered)
        self.assertIn("11", filtered)
        self.assertNotIn("01", filtered)  # 10% < 15%
        self.assertNotIn("10", filtered)  # 10% < 15%
    
    def test_serialization(self):
        """Test to_dict and from_dict methods."""
        result_dict = self.sample_result.to_dict()
        
        # Check required fields
        self.assertIn("total_shots", result_dict)
        self.assertIn("counts", result_dict)
        self.assertIn("probabilities", result_dict)
        
        # Reconstruct and check
        reconstructed = AggregatedResult.from_dict(result_dict)
        self.assertEqual(reconstructed.total_shots, self.sample_result.total_shots)
        self.assertEqual(reconstructed.counts, self.sample_result.counts)


class TestResultAggregator(unittest.TestCase):
    """Test ResultAggregator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = ResultAggregator()
        
        # Create sample hardware results
        self.hw_results = []
        for i in range(3):
            job_handle = JobHandle(f"job_{i}", "test_backend", shots=100)
            hw_result = HardwareResult(
                job_handle=job_handle,
                status=JobStatus.COMPLETED,
                counts={"00": 50, "11": 50},
                execution_time=100.0,
                shots=100
            )
            self.hw_results.append(hw_result)
    
    def test_aggregate_hardware_results(self):
        """Test aggregating hardware results."""
        result = self.aggregator.aggregate_hardware_results(
            self.hw_results, circuit_id="test_circuit"
        )
        
        self.assertEqual(result.total_shots, 300)  # 3 * 100
        self.assertEqual(result.successful_shots, 300)
        self.assertEqual(result.counts["00"], 150)  # 3 * 50
        self.assertEqual(result.counts["11"], 150)  # 3 * 50
        self.assertEqual(result.circuit_id, "test_circuit")
    
    def test_aggregate_shot_results(self):
        """Test aggregating individual shot results."""
        shot_results = [
            ShotResult(0, "00", execution_time=10.0),
            ShotResult(1, "00", execution_time=12.0),
            ShotResult(2, "11", execution_time=11.0),
            ShotResult(3, "11", execution_time=9.0)
        ]
        
        result = self.aggregator.aggregate_shot_results(
            shot_results, circuit_id="shot_test"
        )
        
        self.assertEqual(result.total_shots, 4)
        self.assertEqual(result.successful_shots, 4)
        self.assertEqual(result.counts["00"], 2)
        self.assertEqual(result.counts["11"], 2)
        self.assertAlmostEqual(result.average_shot_time, 10.5)  # (10+12+11+9)/4
    
    def test_merge_aggregated_results(self):
        """Test merging multiple aggregated results."""
        result1 = AggregatedResult(
            total_shots=100,
            successful_shots=100,
            counts={"00": 60, "11": 40}
        )
        
        result2 = AggregatedResult(
            total_shots=200,
            successful_shots=200,
            counts={"00": 80, "11": 120}
        )
        
        merged = self.aggregator.merge_aggregated_results([result1, result2])
        
        self.assertEqual(merged.total_shots, 300)
        self.assertEqual(merged.counts["00"], 140)  # 60 + 80
        self.assertEqual(merged.counts["11"], 160)  # 40 + 120
    
    def test_empty_results_error(self):
        """Test error handling for empty results."""
        with self.assertRaises(ExecutionError):
            self.aggregator.aggregate_hardware_results([])
        
        with self.assertRaises(ExecutionError):
            self.aggregator.aggregate_shot_results([])


class TestResultAnalyzer(unittest.TestCase):
    """Test ResultAnalyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ResultAnalyzer()
        
        # Create two different results for comparison
        self.result1 = AggregatedResult(
            total_shots=1000,
            successful_shots=1000,
            counts={"00": 500, "11": 500}  # Balanced
        )
        
        self.result2 = AggregatedResult(
            total_shots=1000,
            successful_shots=1000,
            counts={"00": 900, "11": 100}  # Biased
        )
    
    def test_compare_results(self):
        """Test result comparison."""
        comparison = self.analyzer.compare_results(self.result1, self.result2)
        
        self.assertIn("total_variation_distance", comparison)
        self.assertIn("overlap", comparison)
        self.assertIn("common_outcomes", comparison)
        
        # Should have some distance due to different distributions
        self.assertGreater(comparison["total_variation_distance"], 0)
        self.assertEqual(comparison["common_outcomes"], 2)  # Both have "00" and "11"
    
    def test_calculate_expectation_value(self):
        """Test expectation value calculation."""
        # Observable: +1 for |00⟩, -1 for |11⟩
        observable = {"00": 1.0, "11": -1.0}
        
        expectation1 = self.analyzer.calculate_expectation_value(self.result1, observable)
        expectation2 = self.analyzer.calculate_expectation_value(self.result2, observable)
        
        # Result 1 is balanced: 0.5 * 1 + 0.5 * (-1) = 0
        self.assertAlmostEqual(expectation1, 0.0, places=6)
        
        # Result 2 is biased: 0.9 * 1 + 0.1 * (-1) = 0.8
        self.assertAlmostEqual(expectation2, 0.8, places=6)
    
    def test_estimate_sampling_error(self):
        """Test sampling error estimation."""
        error = self.analyzer.estimate_sampling_error(self.result1, "00")
        
        # For p=0.5, n=1000: error = sqrt(0.5 * 0.5 / 1000) = sqrt(0.00025) ≈ 0.0158
        expected_error = np.sqrt(0.5 * 0.5 / 1000)
        self.assertAlmostEqual(error, expected_error, places=4)
    
    def test_detect_bias(self):
        """Test bias detection."""
        bias_info = self.analyzer.detect_bias(self.result1, expected_uniform=True)
        
        self.assertIn("n_qubits", bias_info)
        self.assertIn("expected_outcomes", bias_info)
        self.assertIn("observed_outcomes", bias_info)
        self.assertIn("coverage", bias_info)
        
        # 2 qubits -> 4 expected outcomes, 2 observed
        self.assertEqual(bias_info["expected_outcomes"], 4)
        self.assertEqual(bias_info["observed_outcomes"], 2)
        self.assertEqual(bias_info["coverage"], 0.5)


class TestResultStorage(unittest.TestCase):
    """Test ResultStorage functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ResultStorage(storage_dir=self.temp_dir)
        
        self.sample_result = AggregatedResult(
            total_shots=1000,
            successful_shots=1000,
            counts={"00": 600, "11": 400},
            circuit_id="storage_test",
            backend_name="test_backend"
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_result(self):
        """Test saving and loading results."""
        result_id = self.storage.save_result(self.sample_result)
        
        self.assertIsNotNone(result_id)
        self.assertIsInstance(result_id, str)
        
        # Load result back
        loaded_result = self.storage.load_result(result_id)
        
        self.assertEqual(loaded_result.total_shots, self.sample_result.total_shots)
        self.assertEqual(loaded_result.counts, self.sample_result.counts)
        self.assertEqual(loaded_result.circuit_id, self.sample_result.circuit_id)
    
    def test_list_results(self):
        """Test listing stored results."""
        # Save multiple results
        result_ids = []
        for i in range(3):
            result = AggregatedResult(
                total_shots=100 * (i + 1),
                successful_shots=100 * (i + 1),
                counts={"0": 50 * (i + 1), "1": 50 * (i + 1)},
                circuit_id=f"test_{i}"
            )
            result_id = self.storage.save_result(result)
            result_ids.append(result_id)
        
        # List results
        results_list = self.storage.list_results()
        
        self.assertEqual(len(results_list), 3)
        
        # Check result info structure
        result_info = results_list[0]
        self.assertIn("result_id", result_info)
        self.assertIn("saved_at", result_info)
        self.assertIn("circuit_id", result_info)
        self.assertIn("total_shots", result_info)
    
    def test_delete_result(self):
        """Test deleting stored results."""
        result_id = self.storage.save_result(self.sample_result)
        
        # Verify file exists
        filename = f"{self.temp_dir}/{result_id}.json"
        self.assertTrue(os.path.exists(filename))
        
        # Delete result
        success = self.storage.delete_result(result_id)
        self.assertTrue(success)
        
        # Verify file is gone
        self.assertFalse(os.path.exists(filename))


class TestStateVectorSimulatorMultiShot(unittest.TestCase):
    """Test enhanced StateVectorSimulator with multi-shot support."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = StateVectorSimulator(seed=42)
    
    def test_multi_shot_execution(self):
        """Test multi-shot execution with proper sampling."""
        # Create Bell state circuit
        with QuantumProgram() as qp:
            qubits = qp.allocate(2)
            H(qubits[0])
            CNOT(qubits[0], qubits[1])
            measure(qubits)
        
        result = self.simulator.run(qp.circuit, shots=1000)
        
        self.assertEqual(result.shots, 1000)
        self.assertIsInstance(result.counts, dict)
        self.assertEqual(sum(result.counts.values()), 1000)
        
        # Bell state should only produce |00⟩ and |11⟩
        valid_outcomes = {"00", "11"}
        for outcome in result.counts.keys():
            self.assertIn(outcome, valid_outcomes)
        
        # Check probabilities
        self.assertIsNotNone(result.probabilities)
        prob_00 = result.probabilities.get("00", 0)
        prob_11 = result.probabilities.get("11", 0)
        self.assertAlmostEqual(prob_00 + prob_11, 1.0, places=6)
    
    def test_individual_shot_results(self):
        """Test returning individual shot results."""
        with QuantumProgram() as qp:
            qubits = qp.allocate(1)
            H(qubits[0])
            measure(qubits)
        
        result = self.simulator.run(
            qp.circuit, 
            shots=100, 
            return_individual_shots=True
        )
        
        self.assertIsNotNone(result.shot_results)
        self.assertEqual(len(result.shot_results), 100)
        
        # All shots should be either "0" or "1"
        for shot in result.shot_results:
            self.assertIn(shot, ["0", "1"])
    
    def test_statevector_return(self):
        """Test returning statevector."""
        with QuantumProgram() as qp:
            qubits = qp.allocate(2)
            H(qubits[0])
            measure(qubits)
        
        result = self.simulator.run(
            qp.circuit, 
            shots=100, 
            return_statevector=True
        )
        
        self.assertIsNotNone(result.statevector)
        self.assertEqual(len(result.statevector), 4)  # 2^2 = 4
        
        # Check normalization
        norm = np.sum(np.abs(result.statevector) ** 2)
        self.assertAlmostEqual(norm, 1.0, places=6)
    
    def test_deterministic_results_with_seed(self):
        """Test reproducible results with seed."""
        with QuantumProgram() as qp:
            qubits = qp.allocate(2)
            H(qubits[0])
            H(qubits[1])
            measure(qubits)
        
        # Run same circuit twice with same seed
        sim1 = StateVectorSimulator(seed=123)
        sim2 = StateVectorSimulator(seed=123)
        
        result1 = sim1.run(qp.circuit, shots=1000)
        result2 = sim2.run(qp.circuit, shots=1000)
        
        # Results should be identical with same seed
        self.assertEqual(result1.counts, result2.counts)
    
    def test_simulation_metadata(self):
        """Test simulation result metadata."""
        with QuantumProgram() as qp:
            qubits = qp.allocate(3)
            H(qubits[0])
            CNOT(qubits[0], qubits[1])
            CNOT(qubits[1], qubits[2])
            measure(qubits)
        
        result = self.simulator.run(qp.circuit, shots=500)
        
        self.assertEqual(result.num_qubits, 3)
        self.assertEqual(result.gate_count, 4)  # H + 2*CNOT + measure
        self.assertGreater(result.circuit_depth, 0)
        self.assertIsNotNone(result.execution_time)


class TestIntegration(unittest.TestCase):
    """Integration tests for multi-shot execution system."""
    
    def test_end_to_end_multi_shot_workflow(self):
        """Test complete multi-shot workflow."""
        # Create circuit
        with QuantumProgram(name="integration_test") as qp:
            qubits = qp.allocate(2)
            H(qubits[0])
            CNOT(qubits[0], qubits[1])
            measure(qubits)
        
        # Execute simulation
        simulator = StateVectorSimulator(seed=999)
        sim_result = simulator.run(qp.circuit, shots=2000)
        
        # Convert to aggregated result
        agg_result = AggregatedResult(
            total_shots=2000,
            successful_shots=2000,
            counts=sim_result.counts,
            total_execution_time=sim_result.execution_time,
            circuit_id="integration_test",
            backend_name="simulator"
        )
        
        # Analyze result
        analyzer = ResultAnalyzer()
        bias_info = analyzer.detect_bias(agg_result)
        
        # Store result
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = ResultStorage(temp_dir)
            result_id = storage.save_result(agg_result)
            
            # Load and verify
            loaded_result = storage.load_result(result_id)
            self.assertEqual(loaded_result.total_shots, agg_result.total_shots)
            self.assertEqual(loaded_result.counts, agg_result.counts)
        
        # Check that everything worked
        self.assertEqual(agg_result.total_shots, 2000)
        self.assertIn("coverage", bias_info)
        self.assertIsNotNone(result_id)


def run_multi_shot_tests():
    """Run all multi-shot execution tests."""
    print("🧪 Running Multi-Shot Execution Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestShotResult))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAggregatedResult))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestResultAggregator))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestResultAnalyzer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestResultStorage))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStateVectorSimulatorMultiShot))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🧪 Multi-Shot Execution Tests Summary")
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
    success = run_multi_shot_tests()
    exit(0 if success else 1) 