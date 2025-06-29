"""
Gate Operations for the High-Level Language DSL

This module provides convenient functions for applying quantum gates
within the quantum programming context.
"""

from typing import Union, List, Optional
from quantum_platform.compiler.ir.qubit import Qubit
from quantum_platform.compiler.ir.operation import Operation
from quantum_platform.compiler.ir.types import Parameter
from quantum_platform.compiler.language.dsl import get_current_context


def _apply_gate(gate_name: str, targets: Union[Qubit, List[Qubit]], 
               controls: Optional[Union[Qubit, List[Qubit]]] = None,
               **params) -> Operation:
    """Helper function to apply a gate in the current context."""
    context = get_current_context()
    if context is None:
        raise RuntimeError("No active quantum context. Use 'with QuantumProgram():' first.")
    
    return context.apply_gate(gate_name, targets, controls, **params)


# Single-qubit Pauli gates
def I(qubit: Qubit) -> Operation:
    """Apply Identity gate."""
    return _apply_gate("I", qubit)


def X(qubit: Qubit) -> Operation:
    """Apply Pauli-X (NOT) gate."""
    return _apply_gate("X", qubit)


def Y(qubit: Qubit) -> Operation:
    """Apply Pauli-Y gate.""" 
    return _apply_gate("Y", qubit)


def Z(qubit: Qubit) -> Operation:
    """Apply Pauli-Z gate."""
    return _apply_gate("Z", qubit)


def H(qubit: Qubit) -> Operation:
    """Apply Hadamard gate."""
    return _apply_gate("H", qubit)


def S(qubit: Qubit) -> Operation:
    """Apply S gate (phase gate)."""
    return _apply_gate("S", qubit)


def T(qubit: Qubit) -> Operation:
    """Apply T gate (π/8 gate)."""
    return _apply_gate("T", qubit)


# Parametric single-qubit gates
def RX(qubit: Qubit, theta: Parameter) -> Operation:
    """Apply X-rotation gate."""
    return _apply_gate("RX", qubit, theta=theta)


def RY(qubit: Qubit, theta: Parameter) -> Operation:
    """Apply Y-rotation gate."""
    return _apply_gate("RY", qubit, theta=theta)


def RZ(qubit: Qubit, theta: Parameter) -> Operation:
    """Apply Z-rotation gate."""
    return _apply_gate("RZ", qubit, theta=theta)


def U(qubit: Qubit, theta: Parameter, phi: Parameter, lambda_param: Parameter) -> Operation:
    """Apply universal single-qubit gate."""
    return _apply_gate("U", qubit, theta=theta, phi=phi, **{"lambda": lambda_param})


def U1(qubit: Qubit, lambda_param: Parameter) -> Operation:
    """Apply U1 gate (phase gate)."""
    return _apply_gate("U1", qubit, **{"lambda": lambda_param})


def U2(qubit: Qubit, phi: Parameter, lambda_param: Parameter) -> Operation:
    """Apply U2 gate."""
    return _apply_gate("U2", qubit, phi=phi, **{"lambda": lambda_param})


def U3(qubit: Qubit, theta: Parameter, phi: Parameter, lambda_param: Parameter) -> Operation:
    """Apply U3 gate (alias for U)."""
    return _apply_gate("U3", qubit, theta=theta, phi=phi, **{"lambda": lambda_param})


# Two-qubit gates
def CNOT(control: Qubit, target: Qubit) -> Operation:
    """Apply CNOT (Controlled-NOT) gate."""
    return _apply_gate("CNOT", target, control)


def CX(control: Qubit, target: Qubit) -> Operation:
    """Apply CX gate (alias for CNOT)."""
    return _apply_gate("CX", target, control)


def CY(control: Qubit, target: Qubit) -> Operation:
    """Apply Controlled-Y gate."""
    return _apply_gate("CY", target, control)


def CZ(control: Qubit, target: Qubit) -> Operation:
    """Apply Controlled-Z gate."""
    return _apply_gate("CZ", target, control)


def SWAP(qubit1: Qubit, qubit2: Qubit) -> Operation:
    """Apply SWAP gate."""
    return _apply_gate("SWAP", [qubit1, qubit2])


# Parametric two-qubit gates
def CRX(control: Qubit, target: Qubit, theta: Parameter) -> Operation:
    """Apply Controlled X-rotation gate."""
    return _apply_gate("CRX", target, control, theta=theta)


def CRY(control: Qubit, target: Qubit, theta: Parameter) -> Operation:
    """Apply Controlled Y-rotation gate."""
    return _apply_gate("CRY", target, control, theta=theta)


def CRZ(control: Qubit, target: Qubit, theta: Parameter) -> Operation:
    """Apply Controlled Z-rotation gate."""
    return _apply_gate("CRZ", target, control, theta=theta)


# Multi-qubit gates
def TOFFOLI(control1: Qubit, control2: Qubit, target: Qubit) -> Operation:
    """Apply Toffoli (CCX) gate."""
    return _apply_gate("TOFFOLI", target, [control1, control2])


def FREDKIN(control: Qubit, target1: Qubit, target2: Qubit) -> Operation:
    """Apply Fredkin (CSWAP) gate."""
    return _apply_gate("FREDKIN", [target1, target2], control)


# Measurement and special operations
def measure(qubits: Union[Qubit, List[Qubit]], 
           classical_register: Optional[str] = None) -> Operation:
    """
    Measure qubits and store results in classical register.
    
    Args:
        qubits: Qubit or list of qubits to measure
        classical_register: Name of classical register to store results
    
    Returns:
        Measurement operation
    """
    context = get_current_context()
    if context is None:
        raise RuntimeError("No active quantum context. Use 'with QuantumProgram():' first.")
    
    if isinstance(qubits, Qubit):
        qubits = [qubits]
    
    return context.measure(qubits, classical_register)


def reset(qubit: Qubit) -> Operation:
    """Reset a qubit to |0> state."""
    context = get_current_context()
    if context is None:
        raise RuntimeError("No active quantum context. Use 'with QuantumProgram():' first.")
    
    return context.reset_qubit(qubit)


def barrier(*qubits: Qubit) -> Operation:
    """Add a barrier operation."""
    context = get_current_context()
    if context is None:
        raise RuntimeError("No active quantum context. Use 'with QuantumProgram():' first.")
    
    return context.barrier(list(qubits) if qubits else None)


# Export all gate functions
__all__ = [
    "I", "X", "Y", "Z", "H", "S", "T",
    "RX", "RY", "RZ", "U", "U1", "U2", "U3",
    "CNOT", "CX", "CY", "CZ", "SWAP",
    "CRX", "CRY", "CRZ",
    "TOFFOLI", "FREDKIN",
    "measure", "reset", "barrier"
] 