#!/usr/bin/env python3
"""
Simplified Multi-Shot Test Suite

This test suite validates the core multi-shot execution capabilities
without the complex decorator and integration issues.
"""

import unittest
import numpy as np
from collections import Counter

from quantum_platform import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, Y, Z, CNOT, RX, measure
from quantum_platform.simulation.statevector import StateVectorSimulator


class TestMultiShotCore(unittest.TestCase):
    """Test core multi-shot functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = StateVectorSimulator(seed=42)
    
    def test_bell_state_distribution(self):
        """Test Bell state produces expected 50/50 distribution."""
        # Create Bell state
        with QuantumProgram() as qp:
            qubits = qp.allocate(2)
            H(qubits[0])
            CNOT(qubits[0], qubits[1])
            measure(qubits[0])
            measure(qubits[1])
        
        # Execute with many shots
        result = self.simulator.run(qp.circuit, shots=10000)
        
        # Check we get only |00⟩ and |11⟩
        self.assertEqual(set(result.counts.keys()), {'00', '11'})
        
        # Check distribution is approximately 50/50
        total = sum(result.counts.values())
        prob_00 = result.counts.get('00', 0) / total
        prob_11 = result.counts.get('11', 0) / total
        
        # Should be close to 0.5 each (within 5% tolerance)
        self.assertAlmostEqual(prob_00, 0.5, delta=0.05)
        self.assertAlmostEqual(prob_11, 0.5, delta=0.05)
        self.assertAlmostEqual(prob_00 + prob_11, 1.0, places=6)
    
    def test_single_qubit_superposition(self):
        """Test single qubit superposition."""
        with QuantumProgram() as qp:
            qubit = qp.allocate(1)
            H(qubit[0])
            measure(qubit[0])
        
        result = self.simulator.run(qp.circuit, shots=10000)
        
        # Should get both |0⟩ and |1⟩
        self.assertEqual(set(result.counts.keys()), {'0', '1'})
        
        # Should be approximately 50/50
        total = sum(result.counts.values())
        prob_0 = result.counts.get('0', 0) / total
        prob_1 = result.counts.get('1', 0) / total
        
        self.assertAlmostEqual(prob_0, 0.5, delta=0.05)
        self.assertAlmostEqual(prob_1, 0.5, delta=0.05)
    
    def test_deterministic_circuit(self):
        """Test circuit with deterministic outcome."""
        with QuantumProgram() as qp:
            qubits = qp.allocate(2)
            # Apply X to first qubit - should always be |10⟩
            X(qubits[0])
            measure(qubits[0])
            measure(qubits[1])
        
        result = self.simulator.run(qp.circuit, shots=1000)
        
        # Should always get |10⟩
        self.assertEqual(result.counts, {'10': 1000})
    
    def test_uniform_distribution(self):
        """Test uniform distribution over all outcomes."""
        with QuantumProgram() as qp:
            qubits = qp.allocate(2)
            H(qubits[0])
            H(qubits[1])
            measure(qubits[0])
            measure(qubits[1])
        
        result = self.simulator.run(qp.circuit, shots=10000)
        
        # Should get all 4 outcomes
        self.assertEqual(set(result.counts.keys()), {'00', '01', '10', '11'})
        
        # Each should be approximately 25%
        total = sum(result.counts.values())
        for outcome in ['00', '01', '10', '11']:
            prob = result.counts.get(outcome, 0) / total
            self.assertAlmostEqual(prob, 0.25, delta=0.05)
    
    def test_individual_shots(self):
        """Test individual shot results."""
        with QuantumProgram() as qp:
            qubit = qp.allocate(1)
            H(qubit[0])
            measure(qubit[0])
        
        result = self.simulator.run(qp.circuit, shots=100, return_individual_shots=True)
        
        # Should have individual shots
        self.assertIsNotNone(result.shot_results)
        self.assertEqual(len(result.shot_results), 100)
        
        # All shots should be '0' or '1'
        for shot in result.shot_results:
            self.assertIn(shot, ['0', '1'])
        
        # Counts should match individual shots
        counted = Counter(result.shot_results)
        self.assertEqual(dict(counted), result.counts)
    
    def test_statevector_return(self):
        """Test statevector return."""
        with QuantumProgram() as qp:
            qubit = qp.allocate(1)
            H(qubit[0])
        
        result = self.simulator.run(qp.circuit, shots=1, return_statevector=True)
        
        # Should have statevector
        self.assertIsNotNone(result.statevector)
        self.assertEqual(len(result.statevector), 2)  # 2^1 = 2
        
        # Should be |+⟩ = (|0⟩ + |1⟩)/√2
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(result.statevector, expected)
    
    def test_large_shot_count(self):
        """Test large shot count execution."""
        with QuantumProgram() as qp:
            qubit = qp.allocate(1)
            H(qubit[0])
            measure(qubit[0])
        
        # Execute with 100k shots
        result = self.simulator.run(qp.circuit, shots=100000)
        
        self.assertEqual(result.shots, 100000)
        self.assertEqual(sum(result.counts.values()), 100000)
        
        # Should be very close to 50/50
        total = sum(result.counts.values())
        prob_0 = result.counts.get('0', 0) / total
        prob_1 = result.counts.get('1', 0) / total
        
        self.assertAlmostEqual(prob_0, 0.5, delta=0.01)  # 1% tolerance
        self.assertAlmostEqual(prob_1, 0.5, delta=0.01)
    
    def test_execution_time_measurement(self):
        """Test that execution time is measured."""
        with QuantumProgram() as qp:
            qubits = qp.allocate(3)
            for qubit in qubits:
                H(qubit)
                measure(qubit)
        
        result = self.simulator.run(qp.circuit, shots=1000)
        
        # Should have execution time
        self.assertIsNotNone(result.execution_time)
        self.assertGreater(result.execution_time, 0)
    
    def test_circuit_metadata(self):
        """Test circuit metadata in results."""
        with QuantumProgram() as qp:
            qubits = qp.allocate(3)
            H(qubits[0])
            CNOT(qubits[0], qubits[1])
            CNOT(qubits[1], qubits[2])
            measure(qubits[0])
            measure(qubits[1])
            measure(qubits[2])
        
        result = self.simulator.run(qp.circuit, shots=1000)
        
        # Check metadata
        self.assertEqual(result.num_qubits, 3)
        self.assertEqual(result.gate_count, 6)  # 3 gates + 3 measurements
        self.assertGreater(result.circuit_depth, 0)
    
    def test_probability_calculation(self):
        """Test probability calculation from counts."""
        with QuantumProgram() as qp:
            qubit = qp.allocate(1)
            H(qubit[0])
            measure(qubit[0])
        
        result = self.simulator.run(qp.circuit, shots=1000)
        
        # Should have probabilities
        self.assertIsNotNone(result.probabilities)
        
        # Probabilities should sum to 1
        total_prob = sum(result.probabilities.values())
        self.assertAlmostEqual(total_prob, 1.0, places=6)
        
        # Should match counts
        for outcome, prob in result.probabilities.items():
            expected_prob = result.counts[outcome] / result.shots
            self.assertAlmostEqual(prob, expected_prob, places=6)


def run_simplified_tests():
    """Run the simplified multi-shot tests."""
    print("🧪 Running Simplified Multi-Shot Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMultiShotCore))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("🧪 Test Results Summary")
    print("=" * 50)
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
    success = run_simplified_tests()
    exit(0 if success else 1) 