"""
High-Level Quantum Programming Language

This package provides a Python-based DSL for writing quantum programs
in an intuitive way, with support for classical control flow and
quantum-classical hybrid algorithms.
"""

from quantum_platform.compiler.language.dsl import QuantumProgram, QuantumContext
from quantum_platform.compiler.language.operations import *
from quantum_platform.compiler.language.control_flow import if_statement, loop, while_loop

__all__ = [
    "QuantumProgram",
    "QuantumContext", 
    "if_statement",
    "loop",
    "while_loop",
    # Gate operations (imported via *)
    "H", "X", "Y", "Z", "S", "T", "I",
    "RX", "RY", "RZ", "U", "U1", "U2", "U3",
    "CNOT", "CX", "CY", "CZ", "SWAP",
    "TOFFOLI", "FREDKIN", "measure", "reset", "barrier"
] 