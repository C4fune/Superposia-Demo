#!/usr/bin/env python3
"""
Test Suite for Quantum State Visualization and Circuit Debugging

This test suite validates the functionality of the visualization and debugging
modules, ensuring all components work correctly and integrate properly.
"""

import unittest
import numpy as np
import time
from typing import Dict, Any, List

# Import quantum platform components
from quantum_platform import (
    QuantumCircuit, StateVectorSimulator, StateVisualizer, QuantumDebugger,
    VisualizationConfig, StepMode, DebuggerState
)
from quantum_platform.visualization import VisualizationMode
from quantum_platform.visualization.state_utils import (
    compute_bloch_coordinates, analyze_state_structure, get_state_probabilities,
    BlochCoordinates, calculate_entanglement_measures
)


class TestStateVisualizationUtils(unittest.TestCase):
    """Test quantum state utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple test states
        self.zero_state = np.array([1.0, 0.0], dtype=complex)  # |0⟩
        self.one_state = np.array([0.0, 1.0], dtype=complex)   # |1⟩
        self.plus_state = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)  # |+⟩
        self.bell_state = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)  # |Φ+⟩
    
    def test_bloch_coordinates_basic_states(self):
        """Test Bloch sphere coordinates for basic states."""
        # Test |0⟩ state
        coords_zero = compute_bloch_coordinates(self.zero_state, 0)
        self.assertAlmostEqual(coords_zero.z, 1.0, places=6)
        self.assertAlmostEqual(coords_zero.x, 0.0, places=6)
        self.assertAlmostEqual(coords_zero.y, 0.0, places=6)
        
        # Test |1⟩ state  
        coords_one = compute_bloch_coordinates(self.one_state, 0)
        self.assertAlmostEqual(coords_one.z, -1.0, places=6)
        self.assertAlmostEqual(coords_one.x, 0.0, places=6)
        self.assertAlmostEqual(coords_one.y, 0.0, places=6)
        
        # Test |+⟩ state
        coords_plus = compute_bloch_coordinates(self.plus_state, 0)
        self.assertAlmostEqual(coords_plus.x, 1.0, places=6)
        self.assertAlmostEqual(coords_plus.y, 0.0, places=6)
        self.assertAlmostEqual(coords_plus.z, 0.0, places=6)
    
    def test_state_probabilities(self):
        """Test probability calculation."""
        probs_zero = get_state_probabilities(self.zero_state)
        self.assertAlmostEqual(probs_zero['0'], 1.0)
        self.assertEqual(len(probs_zero), 1)
        
        probs_plus = get_state_probabilities(self.plus_state)
        self.assertAlmostEqual(probs_plus['0'], 0.5, places=6)
        self.assertAlmostEqual(probs_plus['1'], 0.5, places=6)
        
        probs_bell = get_state_probabilities(self.bell_state)
        self.assertAlmostEqual(probs_bell['00'], 0.5, places=6)
        self.assertAlmostEqual(probs_bell['11'], 0.5, places=6)
        self.assertEqual(len(probs_bell), 2)
    
    def test_state_structure_analysis(self):
        """Test state structure analysis."""
        # Test single qubit state
        structure_plus = analyze_state_structure(self.plus_state)
        self.assertEqual(structure_plus.num_qubits, 1)
        self.assertEqual(structure_plus.state_dimension, 2)
        self.assertTrue(structure_plus.is_pure)
        self.assertTrue(structure_plus.is_separable)
        
        # Test Bell state (entangled)
        structure_bell = analyze_state_structure(self.bell_state)
        self.assertEqual(structure_bell.num_qubits, 2)
        self.assertEqual(structure_bell.state_dimension, 4)
        self.assertTrue(structure_bell.is_pure)
        self.assertFalse(structure_bell.is_separable)
        
        # Check entanglement measures for Bell state
        if structure_bell.entanglement_structure:
            self.assertGreater(structure_bell.entanglement_structure.von_neumann_entropy, 0.5)
            self.assertEqual(structure_bell.entanglement_structure.schmidt_rank, 2)
    
    def test_entanglement_measures(self):
        """Test entanglement measure calculations."""
        # Bell state should be maximally entangled
        entanglement = calculate_entanglement_measures(self.bell_state)
        
        self.assertAlmostEqual(entanglement.von_neumann_entropy, 1.0, places=6)
        self.assertAlmostEqual(entanglement.concurrence, 1.0, places=5)  # Perfect entanglement
        self.assertEqual(entanglement.schmidt_rank, 2)
        
        # Product state should have no entanglement
        product_state = np.kron(self.zero_state, self.one_state)  # |01⟩
        entanglement_product = calculate_entanglement_measures(product_state)
        
        self.assertAlmostEqual(entanglement_product.von_neumann_entropy, 0.0, places=6)
        self.assertAlmostEqual(entanglement_product.concurrence, 0.0, places=6)


class TestStateVisualizer(unittest.TestCase):
    """Test state visualization components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = VisualizationConfig(
            max_qubits_for_full_display=4,
            max_basis_states_displayed=8
        )
        self.visualizer = StateVisualizer(self.config)
        self.simulator = StateVectorSimulator()
    
    def test_single_qubit_visualization(self):
        """Test visualization of single qubit states."""
        # Create superposition state
        circuit = QuantumCircuit("test", 1)
        q0 = circuit.allocate_qubit()
        circuit.add_gate("h", [q0])
        
        result = self.simulator.run(circuit, shots=1, return_statevector=True)
        
        visualization = self.visualizer.visualize_state(
            result.final_state,
            modes=[VisualizationMode.BLOCH_SPHERE, VisualizationMode.PROBABILITY_HISTOGRAM]
        )
        
        # Check structure
        self.assertIn('bloch_spheres', visualization)
        self.assertIn('probability_histogram', visualization)
        self.assertIn('state_info', visualization)
        
        # Check Bloch sphere
        self.assertEqual(len(visualization['bloch_spheres']), 1)
        bloch = visualization['bloch_spheres'][0]
        self.assertEqual(bloch.qubit_index, 0)
        self.assertIsInstance(bloch.coordinates, BlochCoordinates)
        
        # Check probability histogram
        histogram = visualization['probability_histogram']
        sorted_probs = histogram.get_sorted_probabilities()
        self.assertEqual(len(sorted_probs), 2)  # |0⟩ and |1⟩
        
        # Both should have ~0.5 probability
        for state, prob in sorted_probs:
            self.assertAlmostEqual(prob, 0.5, places=5)
    
    def test_two_qubit_visualization(self):
        """Test visualization of two-qubit entangled states."""
        # Create Bell state
        circuit = QuantumCircuit("bell", 2)
        q0, q1 = circuit.allocate_qubits(2)
        circuit.add_gate("h", [q0])
        circuit.add_gate("cnot", [q0], controls=[q1])
        
        result = self.simulator.run(circuit, shots=1, return_statevector=True)
        
        visualization = self.visualizer.visualize_state(
            result.final_state,
            modes=[
                VisualizationMode.BLOCH_SPHERE,
                VisualizationMode.PROBABILITY_HISTOGRAM,
                VisualizationMode.ENTANGLEMENT_ANALYSIS
            ]
        )
        
        # Check Bloch spheres for both qubits
        self.assertEqual(len(visualization['bloch_spheres']), 2)
        
        # Check entanglement analysis
        if 'entanglement_analysis' in visualization:
            entanglement = visualization['entanglement_analysis']
            self.assertGreater(entanglement.von_neumann_entropy, 0.9)  # High entanglement
        
        # Check probability distribution
        histogram = visualization['probability_histogram']
        sorted_probs = histogram.get_sorted_probabilities()
        
        # Should have two dominant states: |00⟩ and |11⟩
        dominant_probs = [prob for state, prob in sorted_probs if prob > 0.1]
        self.assertEqual(len(dominant_probs), 2)
    
    def test_state_comparison(self):
        """Test state comparison functionality."""
        # Create different states
        circuit1 = QuantumCircuit("state1", 1)
        q1 = circuit1.allocate_qubit()
        circuit1.add_gate("h", [q1])
        
        circuit2 = QuantumCircuit("state2", 1)
        q2 = circuit2.allocate_qubit()
        circuit2.add_gate("x", [q2])  # |1⟩ state
        
        result1 = self.simulator.run(circuit1, shots=1, return_statevector=True)
        result2 = self.simulator.run(circuit2, shots=1, return_statevector=True)
        
        states = [
            ("Superposition", result1.final_state),
            ("Excited", result2.final_state)
        ]
        
        comparison = self.visualizer.create_comparison_visualization(states)
        
        self.assertIn('states', comparison)
        self.assertEqual(comparison['num_states'], 2)
        self.assertIn('Superposition', comparison['states'])
        self.assertIn('Excited', comparison['states'])
    
    def test_export_functionality(self):
        """Test visualization export."""
        circuit = QuantumCircuit("export_test", 1)
        q = circuit.allocate_qubit()
        circuit.add_gate("h", [q])
        
        result = self.simulator.run(circuit, shots=1, return_statevector=True)
        visualization = self.visualizer.visualize_state(result.final_state)
        
        # Test JSON export
        json_export = self.visualizer.export_visualization(visualization, 'json')
        self.assertIsInstance(json_export, str)
        self.assertGreater(len(json_export), 100)  # Should be substantial
        
        # Verify it's valid JSON
        import json
        parsed = json.loads(json_export)
        self.assertIsInstance(parsed, dict)


class TestQuantumDebugger(unittest.TestCase):
    """Test quantum circuit debugging functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = StateVectorSimulator()
        self.debugger = QuantumDebugger(self.simulator)
    
    def test_debug_session_creation(self):
        """Test creating and managing debug sessions."""
        circuit = QuantumCircuit("debug_test", 2)
        q0, q1 = circuit.allocate_qubits(2)
        circuit.add_gate("h", [q0])
        circuit.add_gate("cnot", [q0], controls=[q1])
        
        # Start session
        session_id = self.debugger.start_debug_session(circuit)
        self.assertIsInstance(session_id, str)
        self.assertIn(session_id, self.debugger.active_sessions)
        
        # Check session state
        session_state = self.debugger.get_session_state(session_id)
        self.assertIsNotNone(session_state)
        self.assertEqual(session_state['circuit_name'], 'debug_test')
        self.assertEqual(session_state['total_operations'], 2)
        self.assertEqual(session_state['current_operation_index'], 0)
        self.assertEqual(session_state['state'], DebuggerState.IDLE.value)
        
        # End session
        success = self.debugger.end_session(session_id)
        self.assertTrue(success)
        self.assertNotIn(session_id, self.debugger.active_sessions)
    
    def test_step_by_step_execution(self):
        """Test step-by-step circuit execution."""
        circuit = QuantumCircuit("step_test", 1)
        q = circuit.allocate_qubit()
        circuit.add_gate("h", [q])
        circuit.add_gate("x", [q])
        
        session_id = self.debugger.start_debug_session(circuit)
        
        # Execute first step
        step1_result = self.debugger.step_next(session_id)
        self.assertTrue(step1_result['success'])
        self.assertIn('H', step1_result['operation_executed'])
        
        session_state = self.debugger.get_session_state(session_id)
        self.assertEqual(session_state['current_operation_index'], 1)
        self.assertEqual(session_state['state'], DebuggerState.PAUSED.value)
        
        # Execute second step
        step2_result = self.debugger.step_next(session_id)
        self.assertTrue(step2_result['success'])
        self.assertIn('X', step2_result['operation_executed'])
        
        session_state = self.debugger.get_session_state(session_id)
        self.assertEqual(session_state['current_operation_index'], 2)
        
        # Try to execute beyond end
        step3_result = self.debugger.step_next(session_id)
        self.assertTrue(step3_result['completed'])
        
        self.debugger.end_session(session_id)
    
    def test_breakpoint_management(self):
        """Test breakpoint functionality."""
        circuit = QuantumCircuit("breakpoint_test", 2)
        q0, q1 = circuit.allocate_qubits(2)
        circuit.add_gate("h", [q0])
        circuit.add_gate("cnot", [q0], controls=[q1])
        circuit.add_gate("h", [q1])
        
        session_id = self.debugger.start_debug_session(circuit)
        session = self.debugger.active_sessions[session_id]
        
        # Add breakpoint
        bp_id = session.breakpoint_manager.add_breakpoint(1, description="Before CNOT")
        self.assertIsInstance(bp_id, str)
        
        # Check breakpoint exists
        breakpoints = session.breakpoint_manager.get_breakpoints()
        self.assertIn(1, breakpoints)
        self.assertEqual(breakpoints[1].description, "Before CNOT")
        self.assertTrue(breakpoints[1].enabled)
        
        # Test breakpoint condition checking
        debug_context = {'operation_index': 1, 'num_qubits': 2}
        should_break = session.breakpoint_manager.should_break_at(1, debug_context)
        self.assertTrue(should_break)
        self.assertEqual(breakpoints[1].hit_count, 1)
        
        # Remove breakpoint
        removed = session.breakpoint_manager.remove_breakpoint(1)
        self.assertTrue(removed)
        self.assertNotIn(1, session.breakpoint_manager.get_breakpoints())
        
        self.debugger.end_session(session_id)
    
    def test_state_inspection(self):
        """Test state inspection at different points."""
        circuit = QuantumCircuit("inspect_test", 1)
        q = circuit.allocate_qubit()
        circuit.add_gate("h", [q])
        circuit.add_gate("z", [q])
        
        session_id = self.debugger.start_debug_session(circuit)
        
        # Inspect state at beginning
        state_info_0 = self.debugger.inspect_state_at_operation(session_id, 0)
        self.assertIsNotNone(state_info_0)
        self.assertEqual(state_info_0['operation_index'], 0)
        
        # Inspect state after first operation
        state_info_1 = self.debugger.inspect_state_at_operation(session_id, 1)
        self.assertIsNotNone(state_info_1)
        self.assertEqual(state_info_1['operation_index'], 1)
        
        # Inspect state at end
        state_info_end = self.debugger.inspect_state_at_operation(session_id, len(circuit.operations))
        self.assertIsNotNone(state_info_end)
        
        self.debugger.end_session(session_id)
    
    def test_session_restart(self):
        """Test restarting debug sessions."""
        circuit = QuantumCircuit("restart_test", 1)
        q = circuit.allocate_qubit()
        circuit.add_gate("x", [q])
        
        session_id = self.debugger.start_debug_session(circuit)
        
        # Execute a step
        self.debugger.step_next(session_id)
        session_state = self.debugger.get_session_state(session_id)
        self.assertEqual(session_state['current_operation_index'], 1)
        
        # Restart session
        restarted = self.debugger.restart_session(session_id)
        self.assertTrue(restarted)
        
        # Check session was reset
        session_state = self.debugger.get_session_state(session_id)
        self.assertEqual(session_state['current_operation_index'], 0)
        self.assertEqual(session_state['state'], DebuggerState.IDLE.value)
        
        self.debugger.end_session(session_id)


class TestIntegration(unittest.TestCase):
    """Test integration between visualization and debugging components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = StateVectorSimulator()
        self.visualizer = StateVisualizer()
        self.debugger = QuantumDebugger(self.simulator, self.visualizer)
    
    def test_simulator_visualization_integration(self):
        """Test simulator integration with visualization."""
        circuit = QuantumCircuit("integration", 2)
        q0, q1 = circuit.allocate_qubits(2)
        circuit.add_gate("h", [q0])
        circuit.add_gate("cnot", [q0], controls=[q1])
        
        # Run simulation
        result = self.simulator.run(circuit, shots=1, return_statevector=True)
        
        # Test simulator's built-in visualization methods
        state_viz = self.simulator.get_state_visualization(['bloch_sphere', 'probability_histogram'])
        self.assertIsInstance(state_viz, dict)
        self.assertNotIn('error', state_viz)
        
        bloch_coords = self.simulator.get_bloch_coordinates(0)
        self.assertIn('x', bloch_coords)
        self.assertIn('y', bloch_coords)
        self.assertIn('z', bloch_coords)
        
        state_probs = self.simulator.get_state_probabilities()
        self.assertIsInstance(state_probs, dict)
        self.assertGreater(len(state_probs), 0)
        
        state_analysis = self.simulator.analyze_state_structure()
        self.assertIn('num_qubits', state_analysis)
        self.assertEqual(state_analysis['num_qubits'], 2)
    
    def test_debugger_visualization_integration(self):
        """Test debugger integration with visualization."""
        circuit = QuantumCircuit("debug_viz", 2)
        q0, q1 = circuit.allocate_qubits(2)
        circuit.add_gate("h", [q0])
        circuit.add_gate("cnot", [q0], controls=[q1])
        
        session_id = self.debugger.start_debug_session(circuit)
        
        # Execute one step and check state visualization
        self.debugger.step_next(session_id)
        
        session_state = self.debugger.get_session_state(session_id)
        self.assertIsNotNone(session_state)
        
        # The debugger should use the visualizer for state inspection
        state_info = self.debugger.inspect_state_at_operation(session_id, 1)
        if state_info and 'quantum_state_visualization' in state_info:
            viz = state_info['quantum_state_visualization']
            self.assertIsInstance(viz, dict)
        
        self.debugger.end_session(session_id)


def run_performance_tests():
    """Run performance tests for visualization and debugging."""
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTS")
    print("=" * 60)
    
    simulator = StateVectorSimulator()
    visualizer = StateVisualizer()
    
    # Test visualization performance with different qubit counts
    qubit_counts = [1, 2, 3, 4, 5]
    
    for n_qubits in qubit_counts:
        circuit = QuantumCircuit(f"perf_test_{n_qubits}", n_qubits)
        qubits = circuit.allocate_qubits(n_qubits)
        
        # Create random circuit
        for i, q in enumerate(qubits):
            circuit.add_gate("h", [q])
            if i > 0:
                circuit.add_gate("cnot", [qubits[i-1]], controls=[q])
        
        # Time simulation and visualization
        start_time = time.time()
        result = simulator.run(circuit, shots=1, return_statevector=True)
        sim_time = time.time() - start_time
        
        start_time = time.time()
        visualization = visualizer.visualize_state(result.final_state)
        viz_time = time.time() - start_time
        
        print(f"{n_qubits} qubits: Simulation {sim_time:.4f}s, Visualization {viz_time:.4f}s")


def main():
    """Run all tests."""
    print("🧪 Running Quantum Visualization and Debugging Tests")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestStateVisualizationUtils,
        TestStateVisualizer, 
        TestQuantumDebugger,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance tests if all unit tests pass
    if result.wasSuccessful():
        run_performance_tests()
        print("\n✅ All tests passed! Visualization and debugging systems are working correctly.")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s) occurred.")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 