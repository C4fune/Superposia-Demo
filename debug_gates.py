#!/usr/bin/env python3

print("=== Gate Registry Debug ===")

# Import gate registry functions
from quantum_platform.compiler.gates.registry import get_gate, list_gates, gate_exists

# Test what gates are available
print(f"Available gates: {list_gates()}")
print(f"Total gates registered: {len(list_gates())}")

# Test specific gate lookups
test_gates = ["H", "h", "hadamard", "X", "x", "CNOT", "cnot"]
for gate_name in test_gates:
    exists = gate_exists(gate_name)
    gate = get_gate(gate_name)
    print(f"Gate '{gate_name}': exists={exists}, gate={gate}")

# Test importing standard gates directly
print("\n=== Direct Gate Import ===")
try:
    from quantum_platform.compiler.gates.standard import H, X, Y, Z
    print(f"H gate imported: {H}")
    print(f"H gate name: {H.name}")
    print(f"H gate matrix: {H.matrix.evaluate()}")
except ImportError as e:
    print(f"Import error: {e}")

# Test circuit creation
print("\n=== Circuit Creation ===")
from quantum_platform import QuantumCircuit

circuit = QuantumCircuit("test")
q0 = circuit.allocate_qubit("q0")

# Try different gate names
gate_variants = ["H", "h", "hadamard"]
for gate_name in gate_variants:
    try:
        # Create a new circuit for each test
        test_circuit = QuantumCircuit("test")
        test_q = test_circuit.allocate_qubit("q")
        test_circuit.add_gate(gate_name, [test_q])
        print(f"Successfully added gate '{gate_name}' to circuit")
    except Exception as e:
        print(f"Failed to add gate '{gate_name}': {e}")

# Check if gates module is being imported
print("\n=== Module Import Check ===")
import sys
if 'quantum_platform.compiler.gates.standard' in sys.modules:
    print("Standard gates module is imported")
else:
    print("Standard gates module is NOT imported")
    # Try to import it
    try:
        import quantum_platform.compiler.gates.standard
        print("Successfully imported standard gates module")
        print(f"Available gates after import: {list_gates()}")
    except ImportError as e:
        print(f"Failed to import standard gates: {e}") 