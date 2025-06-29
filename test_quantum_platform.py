#!/usr/bin/env python3
"""
Test script for the Next-Generation Quantum Computing Platform

This demonstrates the core IR and compiler features implemented
including the high-level DSL, gate operations, and circuit building.
"""

import sys
import os
import numpy as np
from math import pi

# Add the platform to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_platform import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, Y, Z, CNOT, RX
from quantum_platform.compiler.language.dsl import allocate, add_classical_register, measure
from quantum_platform.compiler.gates.registry import get_registry_stats
from quantum_platform.compiler.ir.types import ParameterValue


def test_basic_circuit():
    """Test basic circuit creation and gate operations."""
    print("=== Testing Basic Circuit Creation ===")
    
    with QuantumProgram(name="bell_state") as qp:
        # Allocate qubits
        q = allocate(2, names=["qubit0", "qubit1"])
        
        # Add classical register for measurements
        c = add_classical_register("measurements", 2)
        
        # Create Bell state circuit
        H(q[0])              # Apply Hadamard to first qubit
        CNOT(q[0], q[1])     # Apply CNOT with q[0] as control
        
        # Measure qubits
        measure(q, "measurements")
        
        # Print circuit information
        circuit_info = qp.get_circuit_info()
        print(f"Circuit: {circuit_info}")
        
        # Print the full circuit representation
        print(f"\nCircuit details:\n{qp.circuit}")
        
        return qp.circuit


def test_parametric_circuit():
    """Test parametric gates and symbolic parameters."""
    print("\n=== Testing Parametric Circuit ===")
    
    with QuantumProgram(name="parametric_rotation") as qp:
        # Allocate a qubit
        q = allocate(1)
        
        # Apply parametric rotation gates
        RX(q[0], pi/4)       # Concrete parameter
        RX(q[0], ParameterValue(pi/2))  # Using ParameterValue
        
        # Print circuit
        print(f"Parametric circuit:\n{qp.circuit}")
        
        return qp.circuit


def test_gate_registry():
    """Test the gate registry functionality."""
    print("\n=== Testing Gate Registry ===")
    
    stats = get_registry_stats()
    print(f"Registry statistics: {stats}")
    
    from quantum_platform.compiler.gates.registry import get_gate, list_gates
    
    # Test getting specific gates
    h_gate = get_gate("H")
    print(f"Hadamard gate: {h_gate}")
    
    cnot_gate = get_gate("CNOT")
    print(f"CNOT gate: {cnot_gate}")
    
    # List some available gates
    all_gates = list_gates()
    print(f"Available gates ({len(all_gates)}): {all_gates[:10]}...")  # Show first 10


def test_complex_circuit():
    """Test a more complex quantum circuit."""
    print("\n=== Testing Complex Circuit ===")
    
    with QuantumProgram(name="quantum_teleportation_prep") as qp:
        # Allocate qubits for quantum teleportation setup
        qubits = allocate(3, names=["message", "alice", "bob"])
        
        # Prepare entangled pair (Alice and Bob)
        H(qubits[1])         # Alice's qubit
        CNOT(qubits[1], qubits[2])  # Entangle Alice and Bob
        
        # Prepare message qubit in some state
        RX(qubits[0], pi/3)  # Rotate message qubit
        
        # Bell measurement (Alice)
        CNOT(qubits[0], qubits[1])
        H(qubits[0])
        
        # Add measurements
        c = add_classical_register("bell_measurement", 2)
        measure([qubits[0], qubits[1]], "bell_measurement")
        
        print(f"Quantum teleportation circuit:\n{qp.circuit}")
        print(f"Circuit depth: {qp.circuit.depth}")
        print(f"Is parameterized: {qp.circuit.is_parameterized}")
        
        return qp.circuit


def test_circuit_serialization():
    """Test circuit serialization to JSON."""
    print("\n=== Testing Circuit Serialization ===")
    
    with QuantumProgram(name="serialization_test") as qp:
        q = allocate(2)
        H(q[0])
        CNOT(q[0], q[1])
        
        # Test JSON export
        json_data = qp.to_json()
        print("Circuit as JSON (first 200 chars):")
        print(json_data[:200] + "..." if len(json_data) > 200 else json_data)
        
        # Test dict export
        circuit_dict = qp.to_dict()
        print(f"Circuit dict keys: {list(circuit_dict.keys())}")


def test_qubit_management():
    """Test qubit allocation and lifetime management."""
    print("\n=== Testing Qubit Management ===")
    
    with QuantumProgram(name="qubit_management") as qp:
        # Test individual allocation
        q1 = allocate(1)
        print(f"Allocated qubit: {q1}")
        
        # Test batch allocation
        q_batch = allocate(3, names=["a", "b", "c"])
        print(f"Allocated batch: {q_batch}")
        
        # Test register allocation
        from quantum_platform.compiler.language.dsl import allocate_register
        reg = allocate_register("test_reg", 2)
        print(f"Allocated register: {reg}")
        
        # Apply some gates to test qubit usage
        H(q1[0])
        X(q_batch[0])
        CNOT(q_batch[1], q_batch[2])
        
        print(f"Final circuit:\n{qp.circuit}")


def main():
    """Run all tests."""
    print("Next-Generation Quantum Computing Platform - Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        test_basic_circuit()
        test_parametric_circuit()
        test_gate_registry()
        test_complex_circuit()
        test_circuit_serialization()
        test_qubit_management()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("The quantum computing platform core IR and compiler features are working.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 