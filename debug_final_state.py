#!/usr/bin/env python3
import numpy as np
from quantum_platform import QuantumCircuit, StateVectorSimulator

# Test simple circuit with H gate
print("=== Testing Final State Return ===")

circuit = QuantumCircuit("test_h")
q0 = circuit.allocate_qubit("q0")
circuit.add_gate("h", [q0])

print(f"Circuit has {circuit.num_qubits} qubits")
print(f"Circuit has {len(circuit.operations)} operations")
print(f"Operations: {[str(op) for op in circuit.operations]}")

simulator = StateVectorSimulator()

# Test with return_statevector=True
print("\n--- With return_statevector=True ---")
result = simulator.run(circuit, shots=1, return_statevector=True)
print(f"Success: {result.success}")
print(f"Error message: {result.error_message}")
print(f"Final state: {result.final_state}")
print(f"Final state type: {type(result.final_state)}")
if result.final_state is not None:
    print(f"Final state shape: {result.final_state.shape}")
    print(f"Final state values: {result.final_state}")

print(f"Current state: {simulator.get_current_state()}")
print(f"Current state type: {type(simulator.get_current_state())}")

# Test with return_statevector=False
print("\n--- With return_statevector=False ---")
result2 = simulator.run(circuit, shots=1, return_statevector=False)
print(f"Success: {result2.success}")
print(f"Error message: {result2.error_message}")
print(f"Final state (should be None): {result2.final_state}")

# Test expected H gate result
print("\n--- Expected vs Actual ---")
expected_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
print(f"Expected H gate result: {expected_state}")
current = simulator.get_current_state()
if current is not None:
    print(f"Actual current state: {current}")
    print(f"States match: {np.allclose(expected_state, current, atol=1e-10)}")

# Test the circuit operations
print(f"\n--- Circuit Details ---")
for i, op in enumerate(circuit.operations):
    print(f"Operation {i}: {op}")
    print(f"  Type: {type(op)}")
    if hasattr(op, 'name'):
        print(f"  Name: {op.name}")
    if hasattr(op, 'targets'):
        print(f"  Targets: {op.targets}")

# Test state before simulation
print(f"\n--- Initial State ---")
print(f"Simulator state before run: {simulator.get_current_state()}") 