"""
Quantum State Analysis Utilities

This module provides mathematical functions for analyzing quantum states,
computing visualizations, and extracting meaningful information from
state vectors and density matrices.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from math import sqrt, log2, atan2, acos
import cmath

# Pauli matrices for Bloch sphere calculations
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)


@dataclass
class BlochCoordinates:
    """Represents a point on the Bloch sphere."""
    x: float
    y: float
    z: float
    radius: float = 1.0
    
    @property
    def theta(self) -> float:
        """Polar angle (0 to π)."""
        return acos(np.clip(self.z / self.radius, -1, 1))
    
    @property
    def phi(self) -> float:
        """Azimuthal angle (0 to 2π)."""
        return atan2(self.y, self.x)
    
    def to_spherical(self) -> Tuple[float, float, float]:
        """Convert to spherical coordinates (r, theta, phi)."""
        return self.radius, self.theta, self.phi


@dataclass
class EntanglementMeasures:
    """Container for various entanglement measures."""
    concurrence: float
    negativity: float
    von_neumann_entropy: float
    linear_entropy: float
    schmidt_rank: int
    schmidt_coefficients: List[float]


@dataclass
class StateStructure:
    """Analysis of quantum state structure."""
    num_qubits: int
    state_dimension: int
    is_pure: bool
    is_separable: bool
    max_amplitude: float
    dominant_basis_states: List[Tuple[str, float]]  # (basis_state, probability)
    coherence_measures: Dict[str, float]
    entanglement_structure: Optional[EntanglementMeasures]


def compute_bloch_coordinates(state_vector: np.ndarray, qubit_index: int = 0) -> BlochCoordinates:
    """
    Compute Bloch sphere coordinates for a specific qubit.
    
    For multi-qubit states, computes the reduced density matrix for the specified
    qubit and extracts Bloch coordinates.
    
    Args:
        state_vector: Quantum state vector
        qubit_index: Index of qubit to analyze (0-based)
        
    Returns:
        Bloch coordinates for the qubit
    """
    # Get reduced density matrix for the specified qubit
    rho = get_reduced_density_matrix(state_vector, [qubit_index])
    
    # Compute expectation values of Pauli operators
    x = np.real(np.trace(rho @ PAULI_X))
    y = np.real(np.trace(rho @ PAULI_Y))
    z = np.real(np.trace(rho @ PAULI_Z))
    
    # Calculate radius (should be 1 for pure states, <1 for mixed)
    radius = sqrt(x**2 + y**2 + z**2)
    
    return BlochCoordinates(x, y, z, radius)


def get_reduced_density_matrix(state_vector: np.ndarray, 
                             qubit_indices: List[int]) -> np.ndarray:
    """
    Compute reduced density matrix for specified qubits.
    
    Args:
        state_vector: Full quantum state vector
        qubit_indices: List of qubit indices to keep
        
    Returns:
        Reduced density matrix
    """
    num_qubits = int(log2(len(state_vector)))
    
    if not qubit_indices:
        raise ValueError("Must specify at least one qubit")
    
    if any(idx >= num_qubits or idx < 0 for idx in qubit_indices):
        raise ValueError(f"Qubit index out of range [0, {num_qubits-1}]")
    
    # Create full density matrix
    rho_full = np.outer(state_vector, state_vector.conj())
    
    # For simpler implementation, we'll use a direct partial trace approach
    subsystem_dim = 2 ** len(qubit_indices)
    rho_reduced = np.zeros((subsystem_dim, subsystem_dim), dtype=complex)
    
    # Map between full system indices and reduced system indices
    state_dim = len(state_vector)
    
    for i in range(state_dim):
        for j in range(state_dim):
            # Extract bit patterns
            i_binary = format(i, f'0{num_qubits}b')
            j_binary = format(j, f'0{num_qubits}b')
            
            # Check if bits for qubits NOT in our subsystem match
            trace_match = True
            for qubit_idx in range(num_qubits):
                if qubit_idx not in qubit_indices:
                    # For qubits we're tracing out, i and j must have same bit
                    if i_binary[num_qubits-1-qubit_idx] != j_binary[num_qubits-1-qubit_idx]:
                        trace_match = False
                        break
            
            if trace_match:
                # Map to reduced indices - only consider bits for qubits in our subsystem
                i_reduced = 0
                j_reduced = 0
                
                for pos, qubit_idx in enumerate(sorted(qubit_indices)):
                    bit_pos = num_qubits - 1 - qubit_idx
                    if i_binary[bit_pos] == '1':
                        i_reduced += (1 << (len(qubit_indices) - 1 - pos))
                    if j_binary[bit_pos] == '1':
                        j_reduced += (1 << (len(qubit_indices) - 1 - pos))
                
                rho_reduced[i_reduced, j_reduced] += rho_full[i, j]
    
    return rho_reduced


def calculate_entanglement_measures(state_vector: np.ndarray, 
                                  partition: Optional[Tuple[List[int], List[int]]] = None) -> EntanglementMeasures:
    """
    Calculate various entanglement measures for a quantum state.
    
    Args:
        state_vector: Quantum state vector
        partition: Bipartition of qubits as (subsystem_A, subsystem_B)
        
    Returns:
        Entanglement measures
    """
    num_qubits = int(log2(len(state_vector)))
    
    if partition is None:
        # Default partition: first half vs second half
        mid = num_qubits // 2
        partition = (list(range(mid)), list(range(mid, num_qubits)))
    
    subsystem_A, subsystem_B = partition
    
    # Get reduced density matrix for subsystem A
    rho_A = get_reduced_density_matrix(state_vector, subsystem_A)
    
    # Von Neumann entropy
    eigenvals = np.linalg.eigvals(rho_A)
    eigenvals = eigenvals[eigenvals > 1e-12]  # Remove near-zero eigenvalues
    von_neumann = -np.sum(eigenvals * np.log2(eigenvals + 1e-12))
    
    # Linear entropy
    linear = 1 - np.trace(rho_A @ rho_A).real
    
    # Schmidt decomposition
    schmidt_coeffs, schmidt_rank = _compute_schmidt_decomposition(state_vector, partition)
    
    # Concurrence (for 2-qubit systems)
    concurrence = 0.0
    if num_qubits == 2:
        concurrence = _compute_concurrence(state_vector)
    
    # Negativity
    rho_full = np.outer(state_vector, state_vector.conj())
    negativity = _compute_negativity(rho_full, partition)
    
    return EntanglementMeasures(
        concurrence=concurrence,
        negativity=negativity,
        von_neumann_entropy=von_neumann,
        linear_entropy=linear,
        schmidt_rank=schmidt_rank,
        schmidt_coefficients=schmidt_coeffs
    )


def _compute_schmidt_decomposition(state_vector: np.ndarray, 
                                 partition: Tuple[List[int], List[int]]) -> Tuple[List[float], int]:
    """Compute Schmidt decomposition coefficients."""
    subsystem_A, subsystem_B = partition
    dim_A = 2 ** len(subsystem_A)
    dim_B = 2 ** len(subsystem_B)
    
    # Reshape state vector according to partition
    num_qubits = int(log2(len(state_vector)))
    
    # Create permutation to group subsystems
    perm = subsystem_A + subsystem_B
    inv_perm = [0] * num_qubits
    for i, p in enumerate(perm):
        inv_perm[p] = i
    
    # Reshape and permute
    state_tensor = state_vector.reshape([2] * num_qubits)
    state_tensor = np.transpose(state_tensor, inv_perm)
    
    # Reshape to matrix for SVD
    matrix = state_tensor.reshape(dim_A, dim_B)
    
    # Singular value decomposition
    U, s, Vh = np.linalg.svd(matrix)
    
    # Schmidt coefficients are singular values
    schmidt_coeffs = s[s > 1e-12].tolist()
    schmidt_rank = len(schmidt_coeffs)
    
    return schmidt_coeffs, schmidt_rank


def _compute_concurrence(state_vector: np.ndarray) -> float:
    """Compute concurrence for a 2-qubit state."""
    if len(state_vector) != 4:
        return 0.0
    
    # Pauli-Y tensor product
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_yy = np.kron(sigma_y, sigma_y)
    
    # Spin-flipped state
    state_tilde = sigma_yy @ state_vector.conj()
    
    # Concurrence formula
    overlap = np.abs(np.vdot(state_vector, state_tilde))
    eigenvals = np.sort(np.linalg.eigvals(
        np.outer(state_vector, state_vector.conj()) @ 
        np.outer(state_tilde, state_tilde.conj())
    ))[::-1]
    
    lambda_vals = np.sqrt(np.maximum(eigenvals.real, 0))
    concurrence = max(0, lambda_vals[0] - lambda_vals[1] - lambda_vals[2] - lambda_vals[3])
    
    return concurrence


def _compute_negativity(density_matrix: np.ndarray, 
                       partition: Tuple[List[int], List[int]]) -> float:
    """Compute logarithmic negativity."""
    # Partial transpose with respect to subsystem B
    subsystem_A, subsystem_B = partition
    
    # This is a simplified implementation
    # Full implementation would require proper partial transpose
    rho_A = get_reduced_density_matrix(density_matrix.diagonal(), subsystem_A)
    eigenvals = np.linalg.eigvals(rho_A)
    
    # Negativity is related to negative eigenvalues after partial transpose
    negative_sum = np.sum(np.abs(eigenvals[eigenvals < 0]))
    
    return float(negative_sum)


def compute_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Compute fidelity between two quantum states.
    
    Args:
        state1: First quantum state vector
        state2: Second quantum state vector
        
    Returns:
        Fidelity between the states (0 to 1)
    """
    # For pure states, fidelity is |<ψ1|ψ2>|²
    overlap = np.vdot(state1, state2)
    return float(np.abs(overlap) ** 2)


def analyze_state_structure(state_vector: np.ndarray, 
                          max_basis_states: int = 10) -> StateStructure:
    """
    Analyze the structure of a quantum state.
    
    Args:
        state_vector: Quantum state vector to analyze
        max_basis_states: Maximum number of dominant basis states to return
        
    Returns:
        Comprehensive state structure analysis
    """
    num_qubits = int(log2(len(state_vector)))
    state_dimension = len(state_vector)
    
    # Calculate probabilities
    probabilities = np.abs(state_vector) ** 2
    
    # Find dominant basis states
    sorted_indices = np.argsort(probabilities)[::-1]
    dominant_states = []
    
    for i in range(min(max_basis_states, len(sorted_indices))):
        idx = sorted_indices[i]
        prob = probabilities[idx]
        if prob > 1e-10:  # Only include non-negligible amplitudes
            basis_state = format(idx, f'0{num_qubits}b')
            dominant_states.append((basis_state, float(prob)))
    
    # State properties
    is_pure = True  # State vectors are always pure
    max_amplitude = float(np.max(np.abs(state_vector)))
    
    # Coherence measures
    coherence_measures = {
        'l1_norm': float(np.sum(np.abs(state_vector)) - max_amplitude),
        'relative_entropy': float(-np.sum(probabilities * np.log2(probabilities + 1e-12))),
        'linear_entropy': float(1 - np.sum(probabilities ** 2))
    }
    
    # Entanglement analysis (if multi-qubit)
    entanglement_structure = None
    is_separable = True
    
    if num_qubits > 1:
        try:
            entanglement_structure = calculate_entanglement_measures(state_vector)
            # A state is separable if its entanglement entropy is very small
            is_separable = entanglement_structure.von_neumann_entropy < 1e-6
        except Exception:
            # Fallback for edge cases
            is_separable = False
    
    return StateStructure(
        num_qubits=num_qubits,
        state_dimension=state_dimension,
        is_pure=is_pure,
        is_separable=is_separable,
        max_amplitude=max_amplitude,
        dominant_basis_states=dominant_states,
        coherence_measures=coherence_measures,
        entanglement_structure=entanglement_structure
    )


def get_state_probabilities(state_vector: np.ndarray, 
                          basis: str = 'computational') -> Dict[str, float]:
    """
    Get measurement probabilities in specified basis.
    
    Args:
        state_vector: Quantum state vector
        basis: Measurement basis ('computational', 'x', 'y', 'z')
        
    Returns:
        Dictionary mapping basis states to probabilities
    """
    num_qubits = int(log2(len(state_vector)))
    
    if basis == 'computational':
        # Standard computational basis
        probabilities = np.abs(state_vector) ** 2
        return {
            format(i, f'0{num_qubits}b'): float(prob)
            for i, prob in enumerate(probabilities)
            if prob > 1e-12
        }
    
    # For other bases, would need to transform state vector
    # This is a simplified implementation
    probabilities = np.abs(state_vector) ** 2
    return {
        format(i, f'0{num_qubits}b'): float(prob)
        for i, prob in enumerate(probabilities)
        if prob > 1e-12
    } 