"""
Hybrid Quantum-Classical Orchestration Module

This module provides comprehensive support for hybrid algorithms that interleave
quantum and classical computation, such as VQE, QAOA, and adaptive algorithms.
"""

# Core orchestration will be imported when modules are created
# For now, we'll use graceful imports
try:
    from .hybrid_executor import (
        HybridExecutor,
        ParameterBinder,
        ExecutionContext,
        HybridResult,
        OptimizationLoop,
        get_hybrid_executor
    )
except ImportError:
    pass

try:
    from .optimizers import (
        ClassicalOptimizer,
        ScipyOptimizer,
        CustomOptimizer,
        OptimizerResult,
        OptimizerCallback,
        get_optimizer
    )
except ImportError:
    pass

try:
    from .caching import (
        ResultCache,
        ParameterHasher,
        CacheManager,
        get_cache_manager
    )
except ImportError:
    pass

try:
    from .workflows import (
        VQEWorkflow,
        QAOAWorkflow,
        AdaptiveWorkflow,
        WorkflowBuilder,
        WorkflowResult,
        get_workflow
    )
except ImportError:
    pass

try:
    from .monitoring import (
        OptimizationMonitor,
        ConvergenceTracker,
        LivePlotter,
        get_optimization_monitor
    )
except ImportError:
    pass

__all__ = [
    # Core orchestration
    'HybridExecutor',
    'ParameterBinder', 
    'ExecutionContext',
    'HybridResult',
    'OptimizationLoop',
    'get_hybrid_executor',
    
    # Optimizers
    'ClassicalOptimizer',
    'ScipyOptimizer',
    'CustomOptimizer',
    'OptimizerResult',
    'OptimizerCallback',
    'get_optimizer',
    
    # Caching
    'ResultCache',
    'ParameterHasher',
    'CacheManager',
    'get_cache_manager',
    
    # Workflows
    'VQEWorkflow',
    'QAOAWorkflow',
    'AdaptiveWorkflow',
    'WorkflowBuilder',
    'WorkflowResult',
    'get_workflow',
    
    # Monitoring
    'OptimizationMonitor',
    'ConvergenceTracker',
    'LivePlotter',
    'get_optimization_monitor'
] 