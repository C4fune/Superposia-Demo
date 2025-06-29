"""
Quantum Platform Performance Profiling and Benchmarking

This module provides comprehensive performance analysis tools for quantum programs,
including simulation profiling, hardware timing analysis, memory usage tracking,
compilation profiling, and benchmarking capabilities.
"""

# Import profiling components (will be created)
try:
    from quantum_platform.profiling.profiler import (
        QuantumProfiler, ProfilerConfig, ProfileReport, ProfileData, ProfilerMode,
        get_profiler, configure_profiler, profile_execution
    )
    from quantum_platform.profiling.simulation_profiler import (
        SimulationProfiler, SimulationProfile, GateProfile, ExecutionBreakdown
    )
    from quantum_platform.profiling.hardware_profiler import (
        HardwareProfiler, HardwareProfile, HardwareTiming, ProviderTiming
    )
    from quantum_platform.profiling.compiler_profiler import (
        CompilerProfiler, CompilerProfile, PassProfile, OptimizationTiming
    )
    from quantum_platform.profiling.memory_profiler import (
        MemoryProfiler, MemoryProfile, MemorySnapshot, MemoryUsage
    )
    from quantum_platform.profiling.benchmark import (
        QuantumBenchmark, BenchmarkSuite, BenchmarkResult, ScalingAnalysis
    )
    from quantum_platform.profiling.reports import (
        ProfileReportGenerator, ReportFormat, PerformanceReport, BenchmarkReport
    )
except ImportError as e:
    # Graceful fallback if components aren't available yet
    pass

__all__ = [
    # Main profiler
    'QuantumProfiler', 'ProfilerConfig', 'ProfileReport', 'ProfileData', 'ProfilerMode',
    'get_profiler', 'configure_profiler', 'profile_execution',
    
    # Simulation profiling
    'SimulationProfiler', 'SimulationProfile', 'GateProfile', 'ExecutionBreakdown',
    
    # Hardware profiling  
    'HardwareProfiler', 'HardwareProfile', 'HardwareTiming', 'ProviderTiming',
    
    # Compiler profiling
    'CompilerProfiler', 'CompilerProfile', 'PassProfile', 'OptimizationTiming',
    
    # Memory profiling
    'MemoryProfiler', 'MemoryProfile', 'MemorySnapshot', 'MemoryUsage',
    
    # Benchmarking
    'QuantumBenchmark', 'BenchmarkSuite', 'BenchmarkResult', 'ScalingAnalysis',
    
    # Reporting
    'ProfileReportGenerator', 'ReportFormat', 'PerformanceReport', 'BenchmarkReport'
] 