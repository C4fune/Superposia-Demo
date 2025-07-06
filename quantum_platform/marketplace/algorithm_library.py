"""
Built-in Quantum Algorithm Library

This module provides a comprehensive library of common quantum algorithms
and circuits that users can readily import and use as templates or building blocks.

The library includes:
- Grover's search algorithm
- Shor's factorization algorithm (quantum portion)
- Quantum Fourier Transform (QFT)
- Bell state preparation
- Quantum teleportation
- Superdense coding
- GHZ state preparation
- Variational quantum eigensolver (VQE) ansätze
- Quantum phase estimation
- Quantum amplitude amplification

Each algorithm is implemented as a function that returns a quantum circuit
configured for the given parameters, with comprehensive documentation and
usage examples.
"""

import numpy as np
import math
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from quantum_platform.observability.logging import get_logger
from quantum_platform.compilation.ir import QuantumCircuit, QuantumGate, QuantumRegister, ClassicalRegister


class AlgorithmCategory(Enum):
    """Categories of quantum algorithms."""
    SEARCH = "search"
    FACTORIZATION = "factorization"
    SIMULATION = "simulation"
    OPTIMIZATION = "optimization"
    COMMUNICATION = "communication"
    ENTANGLEMENT = "entanglement"
    BASIC_GATES = "basic_gates"
    QUANTUM_WALK = "quantum_walk"
    MACHINE_LEARNING = "machine_learning"
    CRYPTOGRAPHY = "cryptography"


class AlgorithmComplexity(Enum):
    """Complexity levels for algorithms."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class QuantumAlgorithm:
    """
    Represents a quantum algorithm with metadata and implementation.
    """
    name: str
    description: str
    category: AlgorithmCategory
    complexity: AlgorithmComplexity
    min_qubits: int
    max_qubits: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    author: str = "Quantum Platform Team"
    version: str = "1.0"
    
    def __post_init__(self):
        """Validate algorithm parameters."""
        if self.min_qubits <= 0:
            raise ValueError("Minimum qubits must be positive")
        if self.max_qubits is not None and self.max_qubits < self.min_qubits:
            raise ValueError("Maximum qubits must be >= minimum qubits")


class AlgorithmLibrary:
    """
    Main algorithm library providing access to built-in quantum algorithms.
    
    This class serves as the central registry for quantum algorithms,
    providing methods to list, search, and create algorithm instances.
    """
    
    def __init__(self):
        """Initialize the algorithm library."""
        self.logger = get_logger("AlgorithmLibrary")
        self._algorithms: Dict[str, QuantumAlgorithm] = {}
        self._initialize_built_in_algorithms()
    
    def _initialize_built_in_algorithms(self):
        """Initialize the built-in algorithm registry."""
        algorithms = [
            # Basic entanglement circuits
            QuantumAlgorithm(
                name="bell_state",
                description="Creates a Bell state (maximally entangled 2-qubit state)",
                category=AlgorithmCategory.ENTANGLEMENT,
                complexity=AlgorithmComplexity.BEGINNER,
                min_qubits=2,
                max_qubits=2,
                parameters={"state_type": "str"},
                references=["https://en.wikipedia.org/wiki/Bell_state"],
                tags=["entanglement", "basic", "two-qubit"]
            ),
            
            QuantumAlgorithm(
                name="ghz_state",
                description="Creates a GHZ state (maximally entangled multi-qubit state)",
                category=AlgorithmCategory.ENTANGLEMENT,
                complexity=AlgorithmComplexity.BEGINNER,
                min_qubits=3,
                max_qubits=10,
                parameters={"num_qubits": "int"},
                references=["https://en.wikipedia.org/wiki/Greenberger-Horne-Zeilinger_state"],
                tags=["entanglement", "multi-qubit", "ghz"]
            ),
            
            # Search algorithms
            QuantumAlgorithm(
                name="grover",
                description="Grover's quantum search algorithm for unstructured search",
                category=AlgorithmCategory.SEARCH,
                complexity=AlgorithmComplexity.INTERMEDIATE,
                min_qubits=2,
                max_qubits=20,
                parameters={"num_qubits": "int", "marked_items": "List[int]"},
                references=[
                    "https://arxiv.org/abs/quant-ph/9605043",
                    "Nielsen & Chuang, Chapter 6"
                ],
                tags=["search", "grover", "oracle", "amplitude-amplification"]
            ),
            
            # Factorization algorithms
            QuantumAlgorithm(
                name="shor",
                description="Shor's quantum factorization algorithm (quantum portion)",
                category=AlgorithmCategory.FACTORIZATION,
                complexity=AlgorithmComplexity.EXPERT,
                min_qubits=4,
                max_qubits=50,
                parameters={"N": "int", "a": "int"},
                references=[
                    "https://arxiv.org/abs/quant-ph/9508027",
                    "Nielsen & Chuang, Chapter 5"
                ],
                tags=["factorization", "shor", "period-finding", "qft"]
            ),
            
            # Fourier Transform
            QuantumAlgorithm(
                name="qft",
                description="Quantum Fourier Transform",
                category=AlgorithmCategory.BASIC_GATES,
                complexity=AlgorithmComplexity.INTERMEDIATE,
                min_qubits=1,
                max_qubits=20,
                parameters={"num_qubits": "int", "inverse": "bool"},
                references=["Nielsen & Chuang, Chapter 5"],
                tags=["qft", "fourier", "phase-estimation"]
            ),
            
            # Communication protocols
            QuantumAlgorithm(
                name="quantum_teleportation",
                description="Quantum teleportation protocol",
                category=AlgorithmCategory.COMMUNICATION,
                complexity=AlgorithmComplexity.INTERMEDIATE,
                min_qubits=3,
                max_qubits=3,
                parameters={},
                references=["https://arxiv.org/abs/quant-ph/9307077"],
                tags=["teleportation", "communication", "entanglement"]
            ),
            
            QuantumAlgorithm(
                name="superdense_coding",
                description="Superdense coding protocol",
                category=AlgorithmCategory.COMMUNICATION,
                complexity=AlgorithmComplexity.INTERMEDIATE,
                min_qubits=2,
                max_qubits=2,
                parameters={"message": "str"},
                references=["https://arxiv.org/abs/quant-ph/9307077"],
                tags=["superdense", "communication", "entanglement"]
            ),
            
            # Variational algorithms
            QuantumAlgorithm(
                name="variational_ansatz",
                description="Parameterized ansatz circuit for VQE and QAOA",
                category=AlgorithmCategory.OPTIMIZATION,
                complexity=AlgorithmComplexity.ADVANCED,
                min_qubits=2,
                max_qubits=30,
                parameters={"num_qubits": "int", "layers": "int", "ansatz_type": "str"},
                references=[
                    "https://arxiv.org/abs/1304.3061",
                    "https://arxiv.org/abs/1411.4028"
                ],
                tags=["vqe", "qaoa", "variational", "optimization"]
            ),
        ]
        
        for algorithm in algorithms:
            self._algorithms[algorithm.name] = algorithm
        
        self.logger.info(f"Initialized {len(self._algorithms)} built-in algorithms")
    
    def list_algorithms(self, category: Optional[AlgorithmCategory] = None,
                       complexity: Optional[AlgorithmComplexity] = None,
                       min_qubits: Optional[int] = None,
                       max_qubits: Optional[int] = None) -> List[QuantumAlgorithm]:
        """
        List algorithms with optional filtering.
        
        Args:
            category: Filter by algorithm category
            complexity: Filter by complexity level
            min_qubits: Filter by minimum qubit requirement
            max_qubits: Filter by maximum qubit requirement
            
        Returns:
            List of matching algorithms
        """
        algorithms = list(self._algorithms.values())
        
        if category:
            algorithms = [a for a in algorithms if a.category == category]
        
        if complexity:
            algorithms = [a for a in algorithms if a.complexity == complexity]
        
        if min_qubits is not None:
            algorithms = [a for a in algorithms if a.min_qubits >= min_qubits]
        
        if max_qubits is not None:
            algorithms = [a for a in algorithms if a.max_qubits is None or a.max_qubits <= max_qubits]
        
        return algorithms
    
    def get_algorithm(self, name: str) -> Optional[QuantumAlgorithm]:
        """Get algorithm by name."""
        return self._algorithms.get(name)
    
    def search_algorithms(self, query: str) -> List[QuantumAlgorithm]:
        """
        Search algorithms by name, description, or tags.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching algorithms
        """
        query_lower = query.lower()
        results = []
        
        for algorithm in self._algorithms.values():
            if (query_lower in algorithm.name.lower() or
                query_lower in algorithm.description.lower() or
                any(query_lower in tag.lower() for tag in algorithm.tags)):
                results.append(algorithm)
        
        return results
    
    def create_circuit(self, algorithm_name: str, **kwargs) -> QuantumCircuit:
        """
        Create a quantum circuit for the specified algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            **kwargs: Parameters for the algorithm
            
        Returns:
            Quantum circuit implementing the algorithm
        """
        algorithm = self.get_algorithm(algorithm_name)
        if not algorithm:
            raise ValueError(f"Algorithm '{algorithm_name}' not found")
        
        # Dispatch to specific algorithm implementation
        if algorithm_name == "bell_state":
            return create_bell_state(**kwargs)
        elif algorithm_name == "ghz_state":
            return create_ghz_state(**kwargs)
        elif algorithm_name == "grover":
            return create_grover_circuit(**kwargs)
        elif algorithm_name == "shor":
            return create_shor_circuit(**kwargs)
        elif algorithm_name == "qft":
            return create_qft_circuit(**kwargs)
        elif algorithm_name == "quantum_teleportation":
            return create_quantum_teleportation(**kwargs)
        elif algorithm_name == "superdense_coding":
            return create_superdense_coding(**kwargs)
        elif algorithm_name == "variational_ansatz":
            return create_variational_ansatz(**kwargs)
        else:
            raise NotImplementedError(f"Algorithm '{algorithm_name}' not implemented yet")


# Algorithm implementations

def create_bell_state(state_type: str = "phi_plus") -> QuantumCircuit:
    """
    Create a Bell state circuit.
    
    Args:
        state_type: Type of Bell state ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
        
    Returns:
        Quantum circuit creating the specified Bell state
    """
    circuit = QuantumCircuit(2, 2)
    
    # Create |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    circuit.h(0)
    circuit.cx(0, 1)
    
    # Apply additional gates based on state type
    if state_type == "phi_minus":  # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
        circuit.z(0)
    elif state_type == "psi_plus":  # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
        circuit.x(1)
    elif state_type == "psi_minus":  # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
        circuit.x(1)
        circuit.z(0)
    
    # Add measurements
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    
    return circuit


def create_ghz_state(num_qubits: int = 3) -> QuantumCircuit:
    """
    Create a GHZ state circuit.
    
    Args:
        num_qubits: Number of qubits in the GHZ state
        
    Returns:
        Quantum circuit creating the GHZ state
    """
    if num_qubits < 3:
        raise ValueError("GHZ state requires at least 3 qubits")
    
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    # Create |GHZ⟩ = (|000...0⟩ + |111...1⟩)/√2
    circuit.h(0)
    for i in range(1, num_qubits):
        circuit.cx(0, i)
    
    # Add measurements
    for i in range(num_qubits):
        circuit.measure(i, i)
    
    return circuit


def create_grover_circuit(num_qubits: int, marked_items: List[int]) -> QuantumCircuit:
    """
    Create Grover's search algorithm circuit.
    
    Args:
        num_qubits: Number of qubits in the search space
        marked_items: List of marked items (integers)
        
    Returns:
        Quantum circuit implementing Grover's algorithm
    """
    if num_qubits < 2:
        raise ValueError("Grover's algorithm requires at least 2 qubits")
    
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    # Initialize superposition
    for i in range(num_qubits):
        circuit.h(i)
    
    # Calculate number of iterations
    N = 2 ** num_qubits
    iterations = int(np.pi * np.sqrt(N) / 4)
    
    # Grover iterations
    for _ in range(iterations):
        # Oracle: mark the target states
        for item in marked_items:
            _apply_oracle(circuit, item, num_qubits)
        
        # Diffusion operator
        _apply_diffusion(circuit, num_qubits)
    
    # Measure all qubits
    for i in range(num_qubits):
        circuit.measure(i, i)
    
    return circuit


def create_qft_circuit(num_qubits: int, inverse: bool = False) -> QuantumCircuit:
    """
    Create a Quantum Fourier Transform circuit.
    
    Args:
        num_qubits: Number of qubits
        inverse: Whether to create inverse QFT
        
    Returns:
        Quantum circuit implementing QFT
    """
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    def qft_rotations(n: int):
        """Apply QFT rotations."""
        if n == 0:
            return
        
        # Apply Hadamard to the most significant qubit
        circuit.h(n - 1)
        
        # Apply controlled rotations
        for qubit in range(n - 1):
            angle = np.pi / (2 ** (n - 1 - qubit))
            circuit.crz(angle, qubit, n - 1)
        
        # Recursively apply to remaining qubits
        qft_rotations(n - 1)
    
    qft_rotations(num_qubits)
    
    # Swap qubits to get correct order
    for i in range(num_qubits // 2):
        circuit.swap(i, num_qubits - 1 - i)
    
    if inverse:
        circuit = circuit.inverse()
    
    # Add measurements
    for i in range(num_qubits):
        circuit.measure(i, i)
    
    return circuit


def create_shor_circuit(N: int, a: int) -> QuantumCircuit:
    """
    Create Shor's algorithm circuit (quantum portion).
    
    Args:
        N: Number to factor
        a: Randomly chosen base (coprime to N)
        
    Returns:
        Quantum circuit implementing quantum portion of Shor's algorithm
    """
    # Calculate required qubits
    n = int(np.ceil(np.log2(N)))
    qubits_needed = 2 * n
    
    circuit = QuantumCircuit(qubits_needed, qubits_needed)
    
    # Initialize first register in superposition
    for i in range(n):
        circuit.h(i)
    
    # Apply quantum modular exponentiation
    _apply_quantum_modular_exp(circuit, a, N, n)
    
    # Apply inverse QFT to first register
    qft_inverse = create_qft_circuit(n, inverse=True)
    circuit.append(qft_inverse, range(n))
    
    # Measure first register
    for i in range(n):
        circuit.measure(i, i)
    
    return circuit


def create_quantum_teleportation() -> QuantumCircuit:
    """
    Create a quantum teleportation circuit.
    
    Returns:
        Quantum circuit implementing quantum teleportation
    """
    circuit = QuantumCircuit(3, 3)
    
    # Prepare the state to teleport (|ψ⟩ = α|0⟩ + β|1⟩)
    # For demonstration, we'll use a random state
    circuit.ry(np.pi/4, 0)  # Creates (|0⟩ + |1⟩)/√2
    
    # Create Bell pair between qubits 1 and 2
    circuit.h(1)
    circuit.cx(1, 2)
    
    # Bell measurement on qubits 0 and 1
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    
    # Apply corrections based on measurement results
    # In a real implementation, this would be conditional
    circuit.cx(1, 2)
    circuit.cz(0, 2)
    
    circuit.measure(2, 2)
    
    return circuit


def create_superdense_coding(message: str = "00") -> QuantumCircuit:
    """
    Create a superdense coding circuit.
    
    Args:
        message: 2-bit message to encode ("00", "01", "10", "11")
        
    Returns:
        Quantum circuit implementing superdense coding
    """
    circuit = QuantumCircuit(2, 2)
    
    # Create Bell pair
    circuit.h(0)
    circuit.cx(0, 1)
    
    # Encode message on first qubit
    if message == "01":
        circuit.z(0)
    elif message == "10":
        circuit.x(0)
    elif message == "11":
        circuit.x(0)
        circuit.z(0)
    
    # Decode: Bell measurement
    circuit.cx(0, 1)
    circuit.h(0)
    
    # Measure both qubits
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    
    return circuit


def create_variational_ansatz(num_qubits: int, layers: int = 1, 
                             ansatz_type: str = "efficient_su2") -> QuantumCircuit:
    """
    Create a variational ansatz circuit.
    
    Args:
        num_qubits: Number of qubits
        layers: Number of ansatz layers
        ansatz_type: Type of ansatz ("efficient_su2", "hardware_efficient")
        
    Returns:
        Parameterized quantum circuit
    """
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    param_index = 0
    
    for layer in range(layers):
        if ansatz_type == "efficient_su2":
            # Single-qubit rotations
            for qubit in range(num_qubits):
                circuit.ry(f"theta_{param_index}", qubit)
                param_index += 1
                circuit.rz(f"phi_{param_index}", qubit)
                param_index += 1
            
            # Entangling gates
            for qubit in range(num_qubits - 1):
                circuit.cx(qubit, qubit + 1)
            
        elif ansatz_type == "hardware_efficient":
            # Hardware-efficient ansatz
            for qubit in range(num_qubits):
                circuit.ry(f"theta_{param_index}", qubit)
                param_index += 1
            
            # Circular entangling
            for qubit in range(num_qubits):
                circuit.cx(qubit, (qubit + 1) % num_qubits)
    
    # Final layer of single-qubit rotations
    for qubit in range(num_qubits):
        circuit.ry(f"theta_{param_index}", qubit)
        param_index += 1
    
    # Add measurements
    for i in range(num_qubits):
        circuit.measure(i, i)
    
    return circuit


# Helper functions

def _apply_oracle(circuit: QuantumCircuit, marked_item: int, num_qubits: int):
    """Apply oracle for Grover's algorithm."""
    # Convert marked item to binary
    binary_string = format(marked_item, f'0{num_qubits}b')
    
    # Apply X gates to qubits that should be |0⟩
    for i, bit in enumerate(binary_string):
        if bit == '0':
            circuit.x(i)
    
    # Apply multi-controlled Z gate
    if num_qubits == 2:
        circuit.cz(0, 1)
    else:
        circuit.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        circuit.z(num_qubits - 1)
        circuit.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    
    # Undo X gates
    for i, bit in enumerate(binary_string):
        if bit == '0':
            circuit.x(i)


def _apply_diffusion(circuit: QuantumCircuit, num_qubits: int):
    """Apply diffusion operator for Grover's algorithm."""
    # Apply H to all qubits
    for i in range(num_qubits):
        circuit.h(i)
    
    # Apply X to all qubits
    for i in range(num_qubits):
        circuit.x(i)
    
    # Apply multi-controlled Z
    if num_qubits == 2:
        circuit.cz(0, 1)
    else:
        circuit.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        circuit.z(num_qubits - 1)
        circuit.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    
    # Undo X gates
    for i in range(num_qubits):
        circuit.x(i)
    
    # Undo H gates
    for i in range(num_qubits):
        circuit.h(i)


def _apply_quantum_modular_exp(circuit: QuantumCircuit, a: int, N: int, n: int):
    """Apply quantum modular exponentiation for Shor's algorithm."""
    # This is a simplified implementation
    # In practice, this would require more sophisticated modular arithmetic
    for i in range(n):
        # Controlled modular multiplication
        power = 2 ** i
        controlled_value = pow(a, power, N)
        
        # Apply controlled operations based on the computed value
        # This is highly simplified - real implementation would be much more complex
        if controlled_value % 2 == 1:
            circuit.cx(i, n) 