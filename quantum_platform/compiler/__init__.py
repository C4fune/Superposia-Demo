"""
Quantum Compiler and Intermediate Representation (IR) Subsystem

This subsystem is the core engine that translates human-written quantum programs
into executable forms for simulation or hardware execution.
"""

from quantum_platform.compiler.ir import QuantumCircuit, Qubit, Operation
from quantum_platform.compiler.language import QuantumProgram
from quantum_platform.compiler.gates import GATE_SET
from quantum_platform.compiler.serialization import QasmExporter, QasmImporter
# TODO: Implement these modules
# from quantum_platform.compiler.optimization import PassManager
# from quantum_platform.compiler.allocation import QubitAllocator

__all__ = [
    "QuantumCircuit",
    "Qubit", 
    "Operation",
    "QuantumProgram",
    "GATE_SET",
    "QasmExporter",
    "QasmImporter",
    # TODO: Add these when implemented
    # "PassManager",
    # "QubitAllocator",
] 