#!/usr/bin/env python3
from quantum_platform import QuantumCircuit, StateVectorSimulator

# Test simple circuit
circuit = QuantumCircuit("test", 1)
q0 = circuit.allocate_qubit("q0")
circuit.add_gate("h", [q0])

simulator = StateVectorSimulator()
result = simulator.run(circuit, shots=1, return_statevector=True)

print(f"Final state: {result.final_state}")
print(f"Current state: {simulator.get_current_state()}")
print(f"State is None: {result.final_state is None}")

# Test without return_statevector
result2 = simulator.run(circuit, shots=1, return_statevector=False)
print(f"Final state (no flag): {result2.final_state}")
