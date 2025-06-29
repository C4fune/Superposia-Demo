"""
Standard Quantum Gates

This module defines the standard set of quantum gates that are
commonly used in quantum computing, including Pauli gates,
rotation gates, controlled gates, and multi-qubit gates.
"""

import numpy as np
import sympy as sp
from typing import Dict, Any, Optional, List
from math import pi, sqrt, cos, sin

from quantum_platform.compiler.gates.base import Gate, GateMatrix, GateType
from quantum_platform.compiler.gates.registry import register_gate


# Define mathematical constants
SQRT2 = sqrt(2)
INV_SQRT2 = 1 / SQRT2

# Complex number shortcuts
j = 1j


def create_rotation_matrix(axis: str):
    """
    Create a parameterized rotation matrix function.
    
    Args:
        axis: Rotation axis ('x', 'y', or 'z')
        
    Returns:
        Function that creates rotation matrix given angle
    """
    def rotation_matrix(theta: float) -> np.ndarray:
        c = cos(theta / 2)
        s = sin(theta / 2)
        
        if axis == 'x':
            return np.array([
                [c, -j * s],
                [-j * s, c]
            ], dtype=complex)
        elif axis == 'y':
            return np.array([
                [c, -s],
                [s, c]
            ], dtype=complex)
        elif axis == 'z':
            return np.array([
                [np.exp(-j * theta / 2), 0],
                [0, np.exp(j * theta / 2)]
            ], dtype=complex)
        else:
            raise ValueError(f"Invalid rotation axis: {axis}")
    
    return rotation_matrix


def create_universal_gate_matrix(theta: float, phi: float, lambda_: float) -> np.ndarray:
    """
    Create a universal single-qubit gate matrix.
    
    U(θ,φ,λ) = [cos(θ/2), -e^(iλ)sin(θ/2)]
               [e^(iφ)sin(θ/2), e^(i(φ+λ))cos(θ/2)]
    """
    c = cos(theta / 2)
    s = sin(theta / 2)
    
    return np.array([
        [c, -np.exp(j * lambda_) * s],
        [np.exp(j * phi) * s, np.exp(j * (phi + lambda_)) * c]
    ], dtype=complex)


# =============================================================================
# Single-Qubit Pauli Gates
# =============================================================================

# Identity Gate
I_matrix = GateMatrix(
    matrix=np.array([
        [1, 0],
        [0, 1]
    ], dtype=complex)
)

I = Gate(
    name="I",
    num_qubits=1,
    gate_type=GateType.SINGLE_QUBIT,
    matrix=I_matrix,
    description="Identity gate (no-op)"
)

# Pauli-X Gate (NOT gate)
X_matrix = GateMatrix(
    matrix=np.array([
        [0, 1],
        [1, 0]
    ], dtype=complex)
)

X = Gate(
    name="X",
    num_qubits=1,
    gate_type=GateType.SINGLE_QUBIT,
    matrix=X_matrix,
    description="Pauli-X gate (NOT gate)"
)

# Pauli-Y Gate  
Y_matrix = GateMatrix(
    matrix=np.array([
        [0, -j],
        [j, 0]
    ], dtype=complex)
)

Y = Gate(
    name="Y",
    num_qubits=1,
    gate_type=GateType.SINGLE_QUBIT,
    matrix=Y_matrix,
    description="Pauli-Y gate"
)

# Pauli-Z Gate
Z_matrix = GateMatrix(
    matrix=np.array([
        [1, 0],
        [0, -1]
    ], dtype=complex)
)

Z = Gate(
    name="Z", 
    num_qubits=1,
    gate_type=GateType.SINGLE_QUBIT,
    matrix=Z_matrix,
    description="Pauli-Z gate"
)

# Hadamard Gate
H_matrix = GateMatrix(
    matrix=np.array([
        [1, 1],
        [1, -1]
    ], dtype=complex) * INV_SQRT2
)

H = Gate(
    name="H",
    num_qubits=1,
    gate_type=GateType.SINGLE_QUBIT,
    matrix=H_matrix,
    description="Hadamard gate (creates superposition)"
)

# S Gate (Phase gate)
S_matrix = GateMatrix(
    matrix=np.array([
        [1, 0],
        [0, j]
    ], dtype=complex)
)

S = Gate(
    name="S",
    num_qubits=1,
    gate_type=GateType.SINGLE_QUBIT,
    matrix=S_matrix,
    description="S gate (phase gate, √Z)"
)

# T Gate (π/8 gate)
T_matrix = GateMatrix(
    matrix=np.array([
        [1, 0],
        [0, np.exp(j * pi / 4)]
    ], dtype=complex)
)

T = Gate(
    name="T",
    num_qubits=1,
    gate_type=GateType.SINGLE_QUBIT,
    matrix=T_matrix,
    description="T gate (π/8 gate, √S)"
)

# =============================================================================
# Parametric Single-Qubit Gates
# =============================================================================

# Rotation Gates
RX_matrix = GateMatrix(
    matrix=create_rotation_matrix('x'),
    is_parametric=True,
    parameter_names=["theta"]
)

RX = Gate(
    name="RX",
    num_qubits=1,
    gate_type=GateType.PARAMETRIC,
    parameters=["theta"],
    matrix=RX_matrix,
    description="X-axis rotation gate"
)

RY_matrix = GateMatrix(
    matrix=create_rotation_matrix('y'),
    is_parametric=True,
    parameter_names=["theta"]
)

RY = Gate(
    name="RY",
    num_qubits=1,
    gate_type=GateType.PARAMETRIC,
    parameters=["theta"],
    matrix=RY_matrix,
    description="Y-axis rotation gate"
)

RZ_matrix = GateMatrix(
    matrix=create_rotation_matrix('z'),
    is_parametric=True,
    parameter_names=["theta"]
)

RZ = Gate(
    name="RZ",
    num_qubits=1,
    gate_type=GateType.PARAMETRIC,
    parameters=["theta"],
    matrix=RZ_matrix,
    description="Z-axis rotation gate"
)

# Universal Gates
U_matrix = GateMatrix(
    matrix=create_universal_gate_matrix,
    is_parametric=True,
    parameter_names=["theta", "phi", "lambda"]
)

U = Gate(
    name="U",
    num_qubits=1,
    gate_type=GateType.PARAMETRIC,
    parameters=["theta", "phi", "lambda"],
    matrix=U_matrix,
    description="Universal single-qubit gate U(θ,φ,λ)"
)

# U1 Gate (phase gate with arbitrary phase)
def create_u1_matrix(lambda_: float) -> np.ndarray:
    """U1(λ) = diag(1, e^(iλ))"""
    return np.array([
        [1, 0],
        [0, np.exp(j * lambda_)]
    ], dtype=complex)

U1_matrix = GateMatrix(
    matrix=create_u1_matrix,
    is_parametric=True,
    parameter_names=["lambda"]
)

U1 = Gate(
    name="U1",
    num_qubits=1,
    gate_type=GateType.PARAMETRIC,
    parameters=["lambda"],
    matrix=U1_matrix,
    description="Single-parameter phase gate U1(λ)"
)

# U2 Gate 
def create_u2_matrix(phi: float, lambda_: float) -> np.ndarray:
    """U2(φ,λ) = U(π/2, φ, λ)"""
    return create_universal_gate_matrix(pi/2, phi, lambda_)

U2_matrix = GateMatrix(
    matrix=create_u2_matrix,
    is_parametric=True,
    parameter_names=["phi", "lambda"]
)

U2 = Gate(
    name="U2",
    num_qubits=1,
    gate_type=GateType.PARAMETRIC,
    parameters=["phi", "lambda"],
    matrix=U2_matrix,
    description="Two-parameter gate U2(φ,λ)"
)

# U3 Gate (alias for U)
U3 = Gate(
    name="U3",
    num_qubits=1,
    gate_type=GateType.PARAMETRIC,
    parameters=["theta", "phi", "lambda"],
    matrix=U_matrix,
    description="Three-parameter gate U3(θ,φ,λ) = U(θ,φ,λ)"
)

# =============================================================================
# Two-Qubit Gates
# =============================================================================

# CNOT Gate (Controlled-X)
CNOT_matrix = GateMatrix(
    matrix=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
)

CNOT = Gate(
    name="CNOT",
    num_qubits=2,
    gate_type=GateType.TWO_QUBIT,
    matrix=CNOT_matrix,
    description="Controlled-NOT gate"
)

# CX (alias for CNOT)
CX = Gate(
    name="CX",
    num_qubits=2,
    gate_type=GateType.TWO_QUBIT,
    matrix=CNOT_matrix,
    description="Controlled-X gate (alias for CNOT)"
)

# Controlled-Y Gate
CY_matrix = GateMatrix(
    matrix=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -j],
        [0, 0, j, 0]
    ], dtype=complex)
)

CY = Gate(
    name="CY",
    num_qubits=2,
    gate_type=GateType.TWO_QUBIT,
    matrix=CY_matrix,
    description="Controlled-Y gate"
)

# Controlled-Z Gate
CZ_matrix = GateMatrix(
    matrix=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=complex)
)

CZ = Gate(
    name="CZ",
    num_qubits=2,
    gate_type=GateType.TWO_QUBIT,
    matrix=CZ_matrix,
    description="Controlled-Z gate"
)

# SWAP Gate
SWAP_matrix = GateMatrix(
    matrix=np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)
)

SWAP = Gate(
    name="SWAP",
    num_qubits=2,
    gate_type=GateType.TWO_QUBIT,
    matrix=SWAP_matrix,
    description="SWAP gate (exchanges qubit states)"
)

# =============================================================================
# Parametric Two-Qubit Gates
# =============================================================================

def create_controlled_rotation_matrix(axis: str):
    """Create controlled rotation matrix function."""
    def controlled_rotation(theta: float) -> np.ndarray:
        # Identity on |00>, |01>, |10> and rotation on |11>
        rotation = create_rotation_matrix(axis)(theta)
        
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, rotation[0, 0], rotation[0, 1]],
            [0, 0, rotation[1, 0], rotation[1, 1]]
        ], dtype=complex)
    
    return controlled_rotation

# Controlled Rotation Gates
CRX_matrix = GateMatrix(
    matrix=create_controlled_rotation_matrix('x'),
    is_parametric=True,
    parameter_names=["theta"]
)

CRX = Gate(
    name="CRX",
    num_qubits=2,
    gate_type=GateType.PARAMETRIC,
    parameters=["theta"],
    matrix=CRX_matrix,
    description="Controlled X-rotation gate"
)

CRY_matrix = GateMatrix(
    matrix=create_controlled_rotation_matrix('y'),
    is_parametric=True,
    parameter_names=["theta"]
)

CRY = Gate(
    name="CRY",
    num_qubits=2,
    gate_type=GateType.PARAMETRIC,
    parameters=["theta"],
    matrix=CRY_matrix,
    description="Controlled Y-rotation gate"
)

CRZ_matrix = GateMatrix(
    matrix=create_controlled_rotation_matrix('z'),
    is_parametric=True,
    parameter_names=["theta"]
)

CRZ = Gate(
    name="CRZ",
    num_qubits=2,
    gate_type=GateType.PARAMETRIC,
    parameters=["theta"],
    matrix=CRZ_matrix,
    description="Controlled Z-rotation gate"
)

# =============================================================================
# Three-Qubit Gates
# =============================================================================

# Toffoli Gate (Controlled-Controlled-X)
TOFFOLI_matrix = GateMatrix(
    matrix=np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]
    ], dtype=complex)
)

TOFFOLI = Gate(
    name="TOFFOLI",
    num_qubits=3,
    gate_type=GateType.MULTI_QUBIT,
    matrix=TOFFOLI_matrix,
    description="Toffoli gate (Controlled-Controlled-X)"
)

# Fredkin Gate (Controlled-SWAP)  
FREDKIN_matrix = GateMatrix(
    matrix=np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=complex)
)

FREDKIN = Gate(
    name="FREDKIN",
    num_qubits=3,
    gate_type=GateType.MULTI_QUBIT,
    matrix=FREDKIN_matrix,
    description="Fredkin gate (Controlled-SWAP)"
)

# Controlled-SWAP (alias for Fredkin)
CSWAP = Gate(
    name="CSWAP",
    num_qubits=3,
    gate_type=GateType.MULTI_QUBIT,
    matrix=FREDKIN_matrix,
    description="Controlled-SWAP gate (alias for Fredkin)"
)

# =============================================================================
# Special Operations
# =============================================================================

# Measurement (not a unitary gate, but included for completeness)
measure = Gate(
    name="measure",
    num_qubits=1,  # Can measure multiple qubits, but this is per-qubit
    gate_type=GateType.MEASUREMENT,
    description="Measurement operation"
)

# Reset (not unitary)
reset = Gate(
    name="reset",
    num_qubits=1,
    gate_type=GateType.CLASSICAL,
    description="Reset qubit to |0> state"
)

# Barrier (synchronization)
barrier = Gate(
    name="barrier",
    num_qubits=0,  # Can span multiple qubits
    gate_type=GateType.CLASSICAL,
    description="Synchronization barrier"
)

# =============================================================================
# Register all standard gates
# =============================================================================

def register_standard_gates():
    """Register all standard gates in the global registry."""
    gates_to_register = [
        # Single-qubit Pauli gates
        I, X, Y, Z, H, S, T,
        
        # Parametric single-qubit gates
        RX, RY, RZ, U, U1, U2, U3,
        
        # Two-qubit gates
        CNOT, CX, CY, CZ, SWAP,
        
        # Parametric two-qubit gates
        CRX, CRY, CRZ,
        
        # Multi-qubit gates
        TOFFOLI, FREDKIN, CSWAP,
        
        # Special operations
        measure, reset, barrier
    ]
    
    for gate in gates_to_register:
        register_gate(gate)

# Auto-register standard gates when module is imported
register_standard_gates()

# Export all gates for * import
__all__ = [
    "I", "X", "Y", "Z", "H", "S", "T",
    "RX", "RY", "RZ", "U", "U1", "U2", "U3", 
    "CNOT", "CX", "CY", "CZ", "SWAP",
    "CRX", "CRY", "CRZ",
    "TOFFOLI", "FREDKIN", "CSWAP",
    "measure", "reset", "barrier",
    "register_standard_gates"
] 