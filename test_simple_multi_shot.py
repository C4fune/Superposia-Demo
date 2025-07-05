#!/usr/bin/env python3
"""
Simple Multi-Shot Test

Test the basic functionality of the multi-shot simulator with Bell states.
"""

import numpy as np
from quantum_platform import QuantumProgram
from quantum_platform.compiler.language.operations import H, CNOT, measure
from quantum_platform.simulation.statevector import StateVectorSimulator


def test_bell_state():
    """Test Bell state creation and measurement."""
    print("🧪 Testing Bell State Creation")
    print("=" * 40)
    
    # Create Bell state circuit
    with QuantumProgram(name="bell_test") as qp:
        qubits = qp.allocate(2)
        H(qubits[0])
        CNOT(qubits[0], qubits[1])
        # Don't add explicit measure - let simulator handle it
    
    print(f"Circuit operations: {len(qp.circuit.operations)}")
    for i, op in enumerate(qp.circuit.operations):
        print(f"  {i}: {op.__class__.__name__} on {[q.id for q in op.targets]}")
    
    # Test without measurements first
    simulator = StateVectorSimulator(seed=42)
    result = simulator.run(qp.circuit, shots=1, return_statevector=True)
    
    print(f"\nStatevector (no measurements):")
    if result.statevector is not None:
        for i, amp in enumerate(result.statevector):
            if abs(amp) > 1e-10:
                bitstring = format(i, '02b')
                print(f"  |{bitstring}⟩: {amp:.6f} (prob: {abs(amp)**2:.6f})")
    
    # Test with measurements
    print(f"\nTesting with measurements:")
    
    # Add measurements
    with QuantumProgram(name="bell_measured") as qp2:
        qubits = qp2.allocate(2)
        H(qubits[0])
        CNOT(qubits[0], qubits[1])
        measure(qubits[0])
        measure(qubits[1])
    
    # Test multi-shot
    result = simulator.run(qp2.circuit, shots=1000)
    
    print(f"Multi-shot results (1000 shots):")
    total = sum(result.counts.values())
    for outcome, count in sorted(result.counts.items()):
        prob = count / total * 100
        print(f"  |{outcome}⟩: {count} ({prob:.1f}%)")
    
    return result


def test_single_qubit():
    """Test single qubit in superposition."""
    print("\n🧪 Testing Single Qubit Superposition")
    print("=" * 40)
    
    # Create superposition
    with QuantumProgram(name="single_qubit") as qp:
        qubit = qp.allocate(1)
        H(qubit[0])
        measure(qubit[0])
    
    simulator = StateVectorSimulator(seed=123)
    result = simulator.run(qp.circuit, shots=1000)
    
    print(f"Single qubit results (1000 shots):")
    total = sum(result.counts.values())
    for outcome, count in sorted(result.counts.items()):
        prob = count / total * 100
        print(f"  |{outcome}⟩: {count} ({prob:.1f}%)")
    
    return result


def test_deterministic():
    """Test deterministic circuit (no randomness)."""
    print("\n🧪 Testing Deterministic Circuit")
    print("=" * 40)
    
    # Create deterministic circuit
    with QuantumProgram(name="deterministic") as qp:
        qubits = qp.allocate(2)
        # No operations - should always be |00⟩
        measure(qubits[0])
        measure(qubits[1])
    
    simulator = StateVectorSimulator()
    result = simulator.run(qp.circuit, shots=100)
    
    print(f"Deterministic results (100 shots):")
    for outcome, count in result.counts.items():
        print(f"  |{outcome}⟩: {count} (100%)")
    
    return result


def main():
    """Run all tests."""
    print("🎯 Simple Multi-Shot Simulator Tests")
    print("=" * 50)
    
    try:
        bell_result = test_bell_state()
        single_result = test_single_qubit()
        det_result = test_deterministic()
        
        print("\n✅ All tests completed!")
        print("Expected results:")
        print("  • Bell state: ~50% |00⟩ and ~50% |11⟩")
        print("  • Single qubit: ~50% |0⟩ and ~50% |1⟩")
        print("  • Deterministic: 100% |00⟩")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 