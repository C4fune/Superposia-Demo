"""
Quantum Circuit Compilation and Intermediate Representation

This module provides the basic infrastructure for quantum circuit compilation
and intermediate representation (IR) used throughout the quantum platform.
"""

from .ir import QuantumCircuit, QuantumGate, QuantumRegister, ClassicalRegister

__all__ = [
    'QuantumCircuit',
    'QuantumGate', 
    'QuantumRegister',
    'ClassicalRegister'
] 