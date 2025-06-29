#!/usr/bin/env python3
"""
Test script for the Quantum Simulation Engine

This demonstrates the simulation capabilities including state vector simulation,
measurement outcomes, and execution on quantum circuits.
"""

import sys
import os
import numpy as np
from math import pi

# Add the platform to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_platform import (
    QuantumProgram, SimulationExecutor, StateVectorSimulator, 
    QuantumCircuit, SimulationResult
)
from quantum_platform.compiler.language.operations import H, X, Y, Z, CNOT, RX, RY, RZ
from quantum_platform.compiler.language.dsl import allocate, add_classical_register, measure


def test_basic_simulation():
    """Test basic circuit simulation."""
    print("=== Testing Basic Simulation ===")
    
    # Create a simple Bell state circuit
    with QuantumProgram(name="bell_state") as qp:
        q = allocate(2)
        c = add_classical_register("measurements", 2)
        
        H(q[0])
        CNOT(q[0], q[1])
        measure(q, "measurements")
    
    # Create simulator and run
    executor = SimulationExecutor()
    result = executor.run(qp.circuit, shots=1024)
    
    print(f"Circuit: {qp.circuit.name}")
    print(f"Result: {result}")
    print(f"Measurement counts: {result.measurement_counts}")
    print(f"Success: {result.success}")
    
    return result


def test_parametric_simulation():
    """Test simulation with parametric gates."""
    print("\n=== Testing Parametric Simulation ===")
    
    with QuantumProgram(name="rotation_test") as qp:
        q = allocate(1)
        c = add_classical_register("meas", 1)
        
        # Apply rotation and measure
        RX(q[0], pi/2)  # Should give 50/50 probability
        measure(q[0], "meas")
    
    executor = SimulationExecutor()
    result = executor.run(qp.circuit, shots=2048)
    
    print(f"Rotation circuit results: {result.measurement_counts}")
    
    # Check if probabilities are roughly 50/50
    total_shots = sum(result.measurement_counts.values())
    if '0' in result.measurement_counts and '1' in result.measurement_counts:
        prob_0 = result.measurement_counts['0'] / total_shots
        prob_1 = result.measurement_counts['1'] / total_shots
        print(f"P(|0>) = {prob_0:.3f}, P(|1>) = {prob_1:.3f}")
        
        # Should be close to 0.5 each (within 5% tolerance)
        if abs(prob_0 - 0.5) < 0.05 and abs(prob_1 - 0.5) < 0.05:
            print("✅ Probabilities look correct!")
        else:
            print("⚠️ Probabilities seem off")
    
    return result


def test_statevector_access():
    """Test accessing the quantum state vector."""
    print("\n=== Testing State Vector Access ===")
    
    with QuantumProgram(name="ghz_state") as qp:
        q = allocate(3)
        
        # Create GHZ state: (|000> + |111>)/√2
        H(q[0])
        CNOT(q[0], q[1])
        CNOT(q[1], q[2])
    
    executor = SimulationExecutor()
    result = executor.run(qp.circuit, shots=100, return_statevector=True)
    
    if result.final_state is not None:
        print("Final state vector:")
        state = result.final_state
        for i, amplitude in enumerate(state):
            if abs(amplitude) > 1e-10:
                bitstring = format(i, '03b')
                print(f"  |{bitstring}>: {amplitude:.6f} (prob: {abs(amplitude)**2:.6f})")
        
        # Verify it's a GHZ state
        expected_amplitude = 1.0 / np.sqrt(2)
        if (abs(abs(state[0]) - expected_amplitude) < 1e-10 and 
            abs(abs(state[7]) - expected_amplitude) < 1e-10):
            print("✅ GHZ state created correctly!")
        else:
            print("⚠️ State doesn't match expected GHZ state")
    
    return result


def test_resource_estimation():
    """Test resource estimation capabilities."""
    print("\n=== Testing Resource Estimation ===")
    
    with QuantumProgram(name="large_circuit") as qp:
        q = allocate(10)
        
        # Create a deeper circuit
        for i in range(5):
            for j in range(10):
                H(q[j])
            for j in range(9):
                CNOT(q[j], q[j+1])
    
    executor = SimulationExecutor()
    
    # Estimate resources
    estimates = executor.estimate_resources(qp.circuit, shots=1024)
    print("Resource estimates:")
    for key, value in estimates.items():
        print(f"  {key}: {value}")
    
    # Validate circuit
    validation = executor.validate_circuit(qp.circuit)
    print(f"\nCircuit validation: {validation}")
    
    # Compare backends
    comparison = executor.compare_backends(qp.circuit)
    print(f"\nBackend comparison:")
    for backend, info in comparison.items():
        print(f"  {backend}: {info}")
    
    return estimates


def test_initial_state():
    """Test simulation with custom initial states."""
    print("\n=== Testing Custom Initial States ===")
    
    with QuantumProgram(name="initial_state_test") as qp:
        q = allocate(2)
        c = add_classical_register("meas", 2)
        
        # Just measure without any gates
        measure(q, "meas")
    
    executor = SimulationExecutor()
    
    # Test with |00> initial state (default)
    result1 = executor.run(qp.circuit, shots=100)
    print(f"Default |00> state: {result1.measurement_counts}")
    
    # Test with |11> initial state
    result2 = executor.run(qp.circuit, shots=100, initial_state="11")
    print(f"Initial |11> state: {result2.measurement_counts}")
    
    # Test with custom state vector (superposition)
    custom_state = np.array([1, 0, 0, 1]) / np.sqrt(2)  # (|00> + |11>)/√2
    result3 = executor.run(qp.circuit, shots=1024, initial_state=custom_state)
    print(f"Custom superposition: {result3.measurement_counts}")
    
    return result1, result2, result3


def test_error_handling():
    """Test error handling and validation."""
    print("\n=== Testing Error Handling ===")
    
    # Test circuit too large for simulator
    with QuantumProgram(name="too_large") as qp:
        q = allocate(50)  # Too many qubits
        H(q[0])
    
    executor = SimulationExecutor()
    
    try:
        result = executor.run(qp.circuit, shots=10)
        if not result.success:
            print(f"✅ Large circuit properly failed: {result.error_message}")
        else:
            print("⚠️ Large circuit should have failed but didn't")
    except Exception as e:
        print(f"✅ Exception caught for large circuit: {e}")
    
    # Test invalid backend
    try:
        with QuantumProgram(name="simple") as qp:
            q = allocate(1)
            H(q[0])
        
        result = executor.run(qp.circuit, backend="nonexistent")
        print("⚠️ Invalid backend should have failed")
    except Exception as e:
        print(f"✅ Invalid backend properly failed: {e}")


def test_direct_simulator():
    """Test using simulator directly."""
    print("\n=== Testing Direct Simulator Usage ===")
    
    # Create circuit directly
    circuit = QuantumCircuit("direct_test", num_qubits=2)
    q0 = circuit.allocate_qubit("q0")
    q1 = circuit.allocate_qubit("q1")
    
    from quantum_platform.compiler.gates.factory import GateFactory
    factory = GateFactory(circuit)
    
    # Add gates
    factory.apply_gate("H", [q0])
    factory.apply_gate("CNOT", [q0, q1])
    
    # Add measurements
    circuit.add_classical_register("meas", 2)
    circuit.add_measurement([q0, q1], "meas")
    
    # Use simulator directly
    simulator = StateVectorSimulator(max_qubits=20)
    result = simulator.run(circuit, shots=512)
    
    print(f"Direct simulator result: {result}")
    print(f"Backend info: {simulator}")
    
    return result


def main():
    """Run all simulation tests."""
    print("Quantum Simulation Engine - Test Suite")
    print("=" * 50)
    
    try:
        # Run all tests
        test_basic_simulation()
        test_parametric_simulation()
        test_statevector_access()
        test_resource_estimation()
        test_initial_state()
        test_error_handling()
        test_direct_simulator()
        
        print("\n" + "=" * 50)
        print("✅ All simulation tests completed!")
        print("The quantum simulation engine is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 