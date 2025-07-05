"""
Quantum Circuit Transpilation Module

This module provides transpilation capabilities for converting quantum circuits
to be compatible with specific quantum hardware devices.
"""

from .transpiler import (
    CircuitTranspiler,
    TranspilationResult,
    TranspilationPass,
    transpile_for_device
)

from .qubit_mapping import (
    QubitMapping,
    QubitMapper,
    create_initial_mapping,
    optimize_mapping
)

from .gate_decomposition import (
    GateDecomposer,
    DecompositionRule,
    get_decomposition_rules,
    decompose_gate
)

from .routing import (
    QuantumRouter,
    SwapStrategy,
    route_circuit,
    insert_swaps
)

__all__ = [
    # Core Transpilation
    'CircuitTranspiler',
    'TranspilationResult', 
    'TranspilationPass',
    'transpile_for_device',
    
    # Qubit Mapping
    'QubitMapping',
    'QubitMapper',
    'create_initial_mapping',
    'optimize_mapping',
    
    # Gate Decomposition
    'GateDecomposer',
    'DecompositionRule',
    'get_decomposition_rules',
    'decompose_gate',
    
    # Routing
    'QuantumRouter',
    'SwapStrategy',
    'route_circuit',
    'insert_swaps'
] 