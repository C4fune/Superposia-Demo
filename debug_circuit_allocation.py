#!/usr/bin/env python3
from quantum_platform import QuantumCircuit, StateVectorSimulator
import numpy as np

# Test simple circuit
circuit = QuantumCircuit("test", 1)  # We specify 1 qubit here
print(f"After creation - circuit qubits: {circuit.num_qubits}")

q0 = circuit.allocate_qubit("q0")
print(f"After allocate_qubit - circuit qubits: {circuit.num_qubits}")
print(f"Qubit q0: {q0}")
print(f"Qubit q0 ID: {q0.id}")

circuit.add_gate("h", [q0])
print(f"After add_gate - circuit qubits: {circuit.num_qubits}")

# Check qubit list
print(f"Circuit qubits list: {circuit.qubits}")
print(f"Circuit qubits length: {len(circuit.qubits)}")

# Now test with explicit num_qubits=0
circuit2 = QuantumCircuit("test2", 0)
print(f"Circuit2 after creation - qubits: {circuit2.num_qubits}")
q0_2 = circuit2.allocate_qubit("q0_2")
print(f"Circuit2 after allocate - qubits: {circuit2.num_qubits}")
