"""
Quantum Intermediate Representation (IR)

The IR defines the core data structures for representing quantum circuits,
operations, and classical control flow in a hardware-agnostic way.
"""

from quantum_platform.compiler.ir.qubit import Qubit
from quantum_platform.compiler.ir.operation import Operation, MeasurementOperation, IfOperation, LoopOperation
from quantum_platform.compiler.ir.circuit import QuantumCircuit, ClassicalRegister
from quantum_platform.compiler.ir.types import ParameterValue, SymbolicParameter

__all__ = [
    "Qubit",
    "Operation",
    "MeasurementOperation", 
    "IfOperation",
    "LoopOperation",
    "QuantumCircuit",
    "ClassicalRegister",
    "ParameterValue",
    "SymbolicParameter",
] 