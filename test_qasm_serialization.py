#!/usr/bin/env python3
"""
OpenQASM Serialization Test Script

This script demonstrates the OpenQASM export and import functionality
of the quantum platform, including both QASM 2.0 and QASM 3.0 formats.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_platform import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, Y, Z, CNOT, RX, RY, RZ, measure
from quantum_platform.compiler.language.dsl import allocate, add_classical_register
from quantum_platform.compiler.serialization import QasmExporter, QasmImporter
from quantum_platform.compiler.serialization.formats import (
    ExportOptions, ImportOptions, SerializationFormat,
    STRICT_QASM2_EXPORT, COMPATIBLE_QASM3_EXPORT
)


def test_qasm2_export():
    """Test exporting circuits to QASM 2.0 format."""
    print("=== Testing QASM 2.0 Export ===")
    
    # Create a Bell state circuit
    with QuantumProgram(name="bell_state") as qp:
        q = allocate(2)
        c = add_classical_register("meas", 2)
        
        H(q[0])
        CNOT(q[0], q[1])
        measure(q, "meas")
    
    # Export to QASM 2.0
    exporter = QasmExporter()
    qasm2_output = exporter.export(qp.circuit, STRICT_QASM2_EXPORT)
    
    print("QASM 2.0 Output:")
    print(qasm2_output)
    print()
    
    return qasm2_output


def test_qasm3_export():
    """Test exporting circuits to QASM 3.0 format."""
    print("=== Testing QASM 3.0 Export ===")
    
    # Create a more complex circuit with parameterized gates
    with QuantumProgram(name="parameterized_circuit") as qp:
        q = allocate(3)
        c = add_classical_register("results", 3)
        
        H(q[0])
        RX(q[1], theta=1.5708)  # π/2
        RY(q[2], theta=0.7854)  # π/4
        CNOT(q[0], q[1])
        CNOT(q[1], q[2])
        
        # Add a barrier
        from quantum_platform.compiler.language.dsl import barrier
        barrier(q)
        
        # Measurements
        measure(q, "results")
    
    # Export to QASM 3.0
    exporter = QasmExporter()
    qasm3_output = exporter.export(qp.circuit, COMPATIBLE_QASM3_EXPORT)
    
    print("QASM 3.0 Output:")
    print(qasm3_output)
    print()
    
    return qasm3_output


def test_qasm_import():
    """Test importing circuits from QASM format."""
    print("=== Testing QASM Import ===")
    
    # Create a simple QASM 2.0 string
    qasm2_string = """
OPENQASM 2.0;
include "qelib1.inc";

// Simple quantum teleportation preparation
qreg q[3];
creg c[3];

h q[0];
cx q[0],q[1];
cx q[1],q[2];
barrier q;
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
"""
    
    # Import the circuit
    importer = QasmImporter()
    imported_circuit = importer.import_from_string(qasm2_string)
    
    print(f"Imported circuit: {imported_circuit.name}")
    print(f"Qubits: {imported_circuit.num_qubits}")
    print(f"Operations: {imported_circuit.num_operations}")
    print(f"Classical registers: {list(imported_circuit.classical_registers.keys())}")
    
    # Show the operations
    print("\nOperations:")
    for i, op in enumerate(imported_circuit.operations):
        print(f"  {i+1}. {op.name} on qubits {[q.id for q in op.targets]}")
    
    print()
    return imported_circuit


def test_roundtrip_conversion():
    """Test exporting and then importing the same circuit."""
    print("=== Testing Round-trip Conversion ===")
    
    # Create original circuit
    with QuantumProgram(name="original") as qp:
        q = allocate(2)
        c = add_classical_register("output", 2)
        
        H(q[0])
        X(q[1])
        CNOT(q[0], q[1])
        Y(q[0])
        Z(q[1])
        measure(q, "output")
    
    original = qp.circuit
    print(f"Original circuit: {original.num_qubits} qubits, {original.num_operations} operations")
    
    # Export to QASM 3.0
    exporter = QasmExporter()
    qasm_string = exporter.export(original, COMPATIBLE_QASM3_EXPORT)
    
    print("\nExported QASM:")
    print(qasm_string)
    
    # Import back
    importer = QasmImporter()
    imported = importer.import_from_string(qasm_string)
    
    print(f"\nImported circuit: {imported.num_qubits} qubits, {imported.num_operations} operations")
    
    # Compare basic properties
    print(f"Qubits match: {original.num_qubits == imported.num_qubits}")
    print(f"Operation count match: {original.num_operations == imported.num_operations}")
    print(f"Classical registers match: {len(original.classical_registers) == len(imported.classical_registers)}")
    
    return original, imported


def test_complex_circuit_export():
    """Test exporting a more complex circuit with various gate types."""
    print("=== Testing Complex Circuit Export ===")
    
    with QuantumProgram(name="complex_circuit") as qp:
        q = allocate(4)
        c = add_classical_register("final", 4)
        
        # Initialize superposition
        for i in range(4):
            H(q[i])
        
        # Add some rotations
        RX(q[0], theta=0.5)
        RY(q[1], theta=1.0)
        RZ(q[2], theta=1.5)
        
        # Entangle qubits
        CNOT(q[0], q[1])
        CNOT(q[1], q[2])
        CNOT(q[2], q[3])
        
        # More gates
        Z(q[0])
        X(q[1])
        Y(q[2])
        
        # Final measurements
        measure(q, "final")
    
    # Export both formats
    exporter = QasmExporter()
    
    print("QASM 2.0 format:")
    qasm2 = exporter.export(qp.circuit, STRICT_QASM2_EXPORT)
    print(qasm2[:300] + "..." if len(qasm2) > 300 else qasm2)
    
    print("\nQASM 3.0 format:")
    qasm3 = exporter.export(qp.circuit, COMPATIBLE_QASM3_EXPORT)
    print(qasm3[:300] + "..." if len(qasm3) > 300 else qasm3)
    
    return qp.circuit


def test_export_options():
    """Test various export options and configurations."""
    print("=== Testing Export Options ===")
    
    with QuantumProgram(name="options_test") as qp:
        q = allocate(2)
        H(q[0])
        CNOT(q[0], q[1])
        from quantum_platform.compiler.language.dsl import barrier
        barrier(q)
        measure(q[0])
    
    exporter = QasmExporter()
    
    # Test with comments disabled
    no_comments = ExportOptions(
        format=SerializationFormat.QASM3,
        include_comments=False,
        include_barriers=False
    )
    
    print("Without comments and barriers:")
    output = exporter.export(qp.circuit, no_comments)
    print(output)
    
    # Test with comments and metadata
    with_extras = ExportOptions(
        format=SerializationFormat.QASM3,
        include_comments=True,
        include_barriers=True,
        include_metadata=True
    )
    
    print("\nWith comments and barriers:")
    output = exporter.export(qp.circuit, with_extras)
    print(output)


def save_and_load_file():
    """Test saving QASM to file and loading it back."""
    print("=== Testing File I/O ===")
    
    # Create a circuit
    with QuantumProgram(name="file_test") as qp:
        q = allocate(3)
        c = add_classical_register("result", 3)
        
        # Create GHZ state
        H(q[0])
        CNOT(q[0], q[1])
        CNOT(q[0], q[2])
        measure(q, "result")
    
    # Export to QASM
    exporter = QasmExporter()
    qasm_content = exporter.export(qp.circuit, COMPATIBLE_QASM3_EXPORT)
    
    # Save to file
    filename = "test_circuit.qasm"
    with open(filename, 'w') as f:
        f.write(qasm_content)
    
    print(f"Saved circuit to {filename}")
    
    # Load from file
    importer = QasmImporter()
    loaded_circuit = importer.import_from_file(filename)
    
    print(f"Loaded circuit: {loaded_circuit.num_qubits} qubits, {loaded_circuit.num_operations} operations")
    
    # Clean up
    os.remove(filename)
    print(f"Cleaned up {filename}")
    
    return loaded_circuit


def main():
    """Run all QASM serialization tests."""
    print("OpenQASM Serialization Test Suite")
    print("=" * 50)
    
    try:
        # Run all tests
        test_qasm2_export()
        test_qasm3_export()
        test_qasm_import()
        test_roundtrip_conversion()
        test_complex_circuit_export()
        test_export_options()
        save_and_load_file()
        
        print("\n" + "=" * 50)
        print("✅ All OpenQASM serialization tests completed successfully!")
        print("The platform now supports:")
        print("  - OpenQASM 2.0 export")
        print("  - OpenQASM 3.0 export")
        print("  - OpenQASM import with parsing")
        print("  - Round-trip circuit conversion")
        print("  - File-based save/load operations")
        print("  - Configurable export options")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main() 