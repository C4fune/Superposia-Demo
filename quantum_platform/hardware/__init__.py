"""
Quantum Hardware Execution Module

This module provides hardware abstraction and execution capabilities for
running quantum circuits on real quantum hardware providers.
"""

from .hal import (
    QuantumHardwareBackend,
    JobHandle,
    JobStatus,
    DeviceInfo,
    HardwareResult,
    get_backend_registry,
    register_backend,
    DeviceType
)

from .backends import (
    IBMQBackend,
    LocalSimulatorBackend
)

from .job_manager import (
    JobManager,
    HardwareJob,
    JobQueue,
    get_job_manager,
    JobPriority
)

from .results import (
    AggregatedResult,
    ShotResult,
    ResultAggregator,
    ResultAnalyzer,
    ResultStorage,
    MultiShotExecutor,
    get_result_aggregator,
    get_result_analyzer,
    get_result_storage,
    get_multi_shot_executor
)

__all__ = [
    # HAL Core
    'QuantumHardwareBackend',
    'JobHandle',
    'JobStatus', 
    'DeviceInfo',
    'HardwareResult',
    'get_backend_registry',
    'register_backend',
    'DeviceType',
    
    # Backend Implementations
    'IBMQBackend',
    'LocalSimulatorBackend',
    
    # Job Management
    'JobManager',
    'HardwareJob',
    'JobQueue',
    'get_job_manager',
    'JobPriority',
    
    # Results and Multi-Shot Execution
    'AggregatedResult',
    'ShotResult',
    'ResultAggregator',
    'ResultAnalyzer', 
    'ResultStorage',
    'MultiShotExecutor',
    'get_result_aggregator',
    'get_result_analyzer',
    'get_result_storage',
    'get_multi_shot_executor'
] 