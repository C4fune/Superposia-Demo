#!/usr/bin/env python3
from quantum_platform import QuantumCircuit, StateVectorSimulator
import numpy as np

# Test simple circuit
circuit = QuantumCircuit("test", 1)
q0 = circuit.allocate_qubit("q0")
circuit.add_gate("h", [q0])

print(f"Circuit qubits: {circuit.num_qubits}")
print(f"Circuit operations: {len(circuit.operations)}")

simulator = StateVectorSimulator()

# Check state before running
print(f"Initial simulator state: {simulator._state}")

result = simulator.run(circuit, shots=1, return_statevector=True)

print(f"After run - simulator state: {simulator._state}")
print(f"After run - num qubits: {simulator._num_qubits}")
print(f"After run - final state: {result.final_state}")

# Check if state is copied correctly
if simulator._state is not None:
    manual_copy = simulator._state.copy()
    print(f"Manual copy: {manual_copy}")
