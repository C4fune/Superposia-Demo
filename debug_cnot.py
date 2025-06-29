#!/usr/bin/env python3
"""Debug CNOT gate behavior"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_platform import QuantumProgram, SimulationExecutor
from quantum_platform.compiler.language.operations import H, X, CNOT
from quantum_platform.compiler.language.dsl import allocate, add_classical_register, measure

def test_cnot_behavior():
    """Test CNOT with different control/target configurations."""
    
    print("=== Testing CNOT Behavior ===")
    
    # Test 1: |10> -> |11> (control=0, target=1)
    print("\nTest 1: Control qubit 0 in |1>, target qubit 1 in |0>")
    with QuantumProgram(name="cnot_test1") as qp:
        q = allocate(2)
        X(q[0])  # Put control in |1>
        CNOT(q[0], q[1])  # CNOT with control=0, target=1
        measure(q, "result")
        add_classical_register("result", 2)
    
    executor = SimulationExecutor()
    result = executor.run(qp.circuit, shots=100, return_statevector=True)
    print(f"Result: {result.measurement_counts}")
    if result.final_state is not None:
        for i, amp in enumerate(result.final_state):
            if abs(amp) > 1e-10:
                bitstring = format(i, '02b')
                print(f"  |{bitstring}>: {amp:.6f}")
    
    # Test 2: Bell state creation
    print("\nTest 2: Bell state |00> + |11>")
    with QuantumProgram(name="bell") as qp:
        q = allocate(2)
        H(q[0])  # Put control in superposition
        CNOT(q[0], q[1])  # CNOT with control=0, target=1
        measure(q, "result")
        add_classical_register("result", 2)
    
    result = executor.run(qp.circuit, shots=1000, return_statevector=True)
    print(f"Result: {result.measurement_counts}")
    if result.final_state is not None:
        for i, amp in enumerate(result.final_state):
            if abs(amp) > 1e-10:
                bitstring = format(i, '02b')
                print(f"  |{bitstring}>: {amp:.6f}")
    
    # Test 3: Check CNOT matrix
    from quantum_platform.compiler.gates.registry import get_gate
    cnot_gate = get_gate("CNOT")
    if cnot_gate:
        matrix = cnot_gate.get_matrix()
        print(f"\nCNOT Matrix:")
        print(matrix)

if __name__ == "__main__":
    test_cnot_behavior() 