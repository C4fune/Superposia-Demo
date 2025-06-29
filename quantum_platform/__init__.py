"""
Next-Generation Quantum Computing Platform

A comprehensive, local-first quantum computing development environment.
"""

__version__ = "0.1.0"
__author__ = "Quantum Platform Team"
__email__ = "quantum@platform.dev"

# Core imports for easy access
from quantum_platform.compiler.ir.circuit import QuantumCircuit
from quantum_platform.compiler.ir.qubit import Qubit
from quantum_platform.compiler.ir.operation import Operation
from quantum_platform.compiler.language.dsl import QuantumProgram
from quantum_platform.compiler.gates.registry import GATE_SET
from quantum_platform.simulation import (
    QuantumSimulator, SimulationResult, StateVectorSimulator, SimulationExecutor
)

# Security and RBAC System
from quantum_platform.security import (
    Permission, UserRole, User, UserContext, QuantumPlatformSecurity,
    SecurityContext, AdminContext, get_quantum_security, initialize_security
)

# Observability and Debugging System
from quantum_platform.observability import (
    QuantumLogger, get_logger, setup_logging, configure_logging,
    LogLevel, LogFormat, LogConfig, SystemMonitor, PerformanceMetrics, ResourceUsage
)

# Visualization and Debugging Tools
from quantum_platform.visualization import (
    StateVisualizer, BlochSphere, ProbabilityHistogram, StateVectorAnalysis,
    VisualizationConfig, QuantumDebugger, DebugSession, BreakpointManager,
    StepMode, DebuggerState, DebugEvent
)

# Performance Profiling and Benchmarking System
from quantum_platform.profiling import (
    QuantumProfiler, ProfilerConfig, ProfilerMode, ProfileReport, ProfileData,
    SimulationProfiler, SimulationProfile, GateProfile, ExecutionBreakdown,
    HardwareProfiler, HardwareProfile, HardwareTiming, ProviderTiming,
    CompilerProfiler, CompilerProfile, PassProfile, OptimizationTiming,
    MemoryProfiler, MemoryProfile, MemorySnapshot, MemoryUsage,
    QuantumBenchmark, BenchmarkSuite, BenchmarkResult, ScalingAnalysis,
    ProfileReportGenerator, ReportFormat, PerformanceReport, BenchmarkReport,
    get_profiler, configure_profiler, profile_execution
)

# Execution Monitoring and Job Management  
from quantum_platform.execution import (
    ExecutionJob, JobManager, JobStatus, JobType,
    StatusMonitor, ProgressTracker, ExecutionDashboard,
    get_dashboard
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "QuantumCircuit",
    "Qubit",
    "Operation", 
    "QuantumProgram",
    "GATE_SET",
    "QuantumSimulator",
    "SimulationResult", 
    "StateVectorSimulator",
    "SimulationExecutor",
    # Security components
    "Permission",
    "UserRole", 
    "User",
    "UserContext",
    "QuantumPlatformSecurity",
    "SecurityContext",
    "AdminContext",
    "get_quantum_security",
    "initialize_security",
    # Observability components
    "QuantumLogger",
    "get_logger",
    "setup_logging", 
    "configure_logging",
    "LogLevel",
    "LogFormat",
    "LogConfig",
    "SystemMonitor",
    "PerformanceMetrics",
    "ResourceUsage",
    # Visualization and debugging components
    "StateVisualizer",
    "BlochSphere",
    "ProbabilityHistogram",
    "StateVectorAnalysis",
    "VisualizationConfig",
    "QuantumDebugger",
    "DebugSession",
    "BreakpointManager",
    "StepMode",
    "DebuggerState",
    "DebugEvent",
    # Performance profiling and benchmarking components
    "QuantumProfiler",
    "ProfilerConfig", 
    "ProfilerMode",
    "ProfileReport",
    "ProfileData",
    "SimulationProfiler",
    "SimulationProfile",
    "GateProfile",
    "ExecutionBreakdown",
    "HardwareProfiler",
    "HardwareProfile",
    "HardwareTiming",
    "ProviderTiming",
    "CompilerProfiler",
    "CompilerProfile",
    "PassProfile",
    "OptimizationTiming",
    "MemoryProfiler",
    "MemoryProfile",
    "MemorySnapshot",
    "MemoryUsage",
    "QuantumBenchmark",
    "BenchmarkSuite",
    "BenchmarkResult",
    "ScalingAnalysis",
    "ProfileReportGenerator",
    "ReportFormat",
    "PerformanceReport",
    "BenchmarkReport",
    "get_profiler",
    "configure_profiler",
    "profile_execution",
    # Execution monitoring components
    "ExecutionJob",
    "JobManager", 
    "JobStatus",
    "JobType",
    "StatusMonitor",
    "ProgressTracker",
    "ExecutionDashboard",
    "get_dashboard",
] 