"""
Quantum Circuit Optimization Module

This module provides comprehensive optimization passes for quantum circuits,
including gate cancellation, parallelization, constant folding, and more.
"""

from .pass_manager import (
    PassManager,
    OptimizationPass,
    PassPriority,
    PassResult,
    PassStatistics
)

from .passes import (
    # Basic optimization passes
    GateCancellationPass,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    
    # Advanced optimization passes
    CommutationAnalysisPass,
    ParallelizationPass,
    CircuitDepthReductionPass,
    
    # Utility passes
    CircuitAnalysisPass,
    StatisticsPass
)

from .registry import (
    OptimizationRegistry,
    register_pass,
    get_registered_passes,
    create_optimization_pipeline
)

__all__ = [
    # Pass management
    'PassManager',
    'OptimizationPass',
    'PassPriority',
    'PassResult',
    'PassStatistics',
    
    # Optimization passes
    'GateCancellationPass',
    'ConstantFoldingPass',
    'DeadCodeEliminationPass',
    'CommutationAnalysisPass',
    'ParallelizationPass',
    'CircuitDepthReductionPass',
    'CircuitAnalysisPass',
    'StatisticsPass',
    
    # Registry
    'OptimizationRegistry',
    'register_pass',
    'get_registered_passes',
    'create_optimization_pipeline'
] 