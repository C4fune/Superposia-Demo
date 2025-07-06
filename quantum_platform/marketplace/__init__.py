"""
Quantum Platform Marketplace and Community Features

This module provides a comprehensive marketplace system for the quantum platform,
including:

- Built-in quantum algorithm library with common algorithms
- Marketplace for sharing and acquiring extensions
- Community-driven ecosystem for collaboration
- Plugin system for custom algorithms and tools
- Educational resources and example circuits

The marketplace enables users to:
- Access pre-built quantum algorithms (Grover, Shor, QFT, etc.)
- Share custom algorithms and implementations
- Discover and install community extensions
- Access educational resources and tutorials
- Collaborate on quantum computing projects
"""

from .algorithm_library import (
    AlgorithmLibrary, 
    QuantumAlgorithm,
    create_bell_state,
    create_grover_circuit,
    create_shor_circuit,
    create_qft_circuit,
    create_quantum_teleportation,
    create_superdense_coding,
    create_ghz_state,
    create_variational_ansatz
)

from .marketplace_manager import (
    MarketplaceManager,
    Package,
    PackageType,
    PackageStatus,
    MarketplaceConfig
)

from .community import (
    CommunityManager,
    UserProfile,
    AlgorithmContribution,
    CommunityRating,
    DiscussionThread
)

from .plugin_system import (
    PluginManager,
    QuantumPlugin,
    PluginType,
    PluginInterface
)

__all__ = [
    'AlgorithmLibrary',
    'QuantumAlgorithm', 
    'create_bell_state',
    'create_grover_circuit',
    'create_shor_circuit',
    'create_qft_circuit',
    'create_quantum_teleportation',
    'create_superdense_coding',
    'create_ghz_state',
    'create_variational_ansatz',
    'MarketplaceManager',
    'Package',
    'PackageType',
    'PackageStatus',
    'MarketplaceConfig',
    'CommunityManager',
    'UserProfile',
    'AlgorithmContribution',
    'CommunityRating',
    'DiscussionThread',
    'PluginManager',
    'QuantumPlugin',
    'PluginType',
    'PluginInterface'
] 