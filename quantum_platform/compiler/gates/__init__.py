"""
Quantum Gate Set and Extensibility

This package defines the flexible gate set for the platform, including
common quantum gates and extensibility mechanisms for custom gates.
"""

from quantum_platform.compiler.gates.base import Gate, GateMatrix
from quantum_platform.compiler.gates.registry import GATE_SET, GateRegistry, register_gate
from quantum_platform.compiler.gates.standard import *  # Import all standard gates
from quantum_platform.compiler.gates.factory import GateFactory

__all__ = [
    "Gate",
    "GateMatrix", 
    "GATE_SET",
    "GateRegistry",
    "register_gate",
    "GateFactory",
    # Standard gates (imported via *)
    "H", "X", "Y", "Z", "S", "T", "I",
    "RX", "RY", "RZ", "U", "U1", "U2", "U3",
    "CNOT", "CX", "CY", "CZ", "CRX", "CRY", "CRZ",
    "SWAP", "CSWAP", "TOFFOLI", "FREDKIN",
    "measure", "reset", "barrier"
] 