"""
Main Quantum Profiler

This module provides the central profiling coordinator that manages
all performance analysis across simulation, hardware, and compilation.
"""

import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import cProfile
import pstats
from io import StringIO

from quantum_platform.observability.logging import get_logger
from quantum_platform.observability.monitor import get_monitor


class ProfilerMode(Enum):
    """Profiler operation modes."""
    DISABLED = "disabled"
    BASIC = "basic"
    DETAILED = "detailed"
    DEVELOPER = "developer"


@dataclass
class ProfilerConfig:
    """Configuration for the quantum profiler."""
    mode: ProfilerMode = ProfilerMode.BASIC
    
    # What to profile
    profile_simulation: bool = True
    profile_hardware: bool = True
    profile_compilation: bool = True
    profile_memory: bool = True
    
    # Profiling options
    track_gate_timing: bool = True
    track_memory_usage: bool = True
    track_cpu_usage: bool = True
    use_cprofile: bool = False  # For developer mode
    
    # Sampling intervals
    memory_sample_interval: float = 0.1  # seconds
    cpu_sample_interval: float = 0.1     # seconds
    
    # Report options
    include_detailed_reports: bool = True
    max_report_entries: int = 100
    
    # Performance overhead limits
    max_overhead_percent: float = 5.0  # Maximum allowed overhead


@dataclass
class ProfileData:
    """Container for profile data from a single execution."""
    profile_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Basic timing
    total_duration: float = 0.0
    compilation_time: float = 0.0
    simulation_time: float = 0.0
    postprocessing_time: float = 0.0
    
    # Detailed breakdowns
    simulation_profile: Optional[Any] = None
    hardware_profile: Optional[Any] = None
    compiler_profile: Optional[Any] = None
    memory_profile: Optional[Any] = None
    
    # System resource usage
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    
    # Metadata
    circuit_name: str = ""
    backend_name: str = ""
    shots: int = 0
    num_qubits: int = 0
    gate_count: int = 0
    
    # Developer profiling
    cprofile_stats: Optional[pstats.Stats] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile data to dictionary."""
        return {
            'profile_id': self.profile_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration': self.total_duration,
            'compilation_time': self.compilation_time,
            'simulation_time': self.simulation_time,
            'postprocessing_time': self.postprocessing_time,
            'peak_memory_mb': self.peak_memory_mb,
            'avg_cpu_percent': self.avg_cpu_percent,
            'circuit_name': self.circuit_name,
            'backend_name': self.backend_name,
            'shots': self.shots,
            'num_qubits': self.num_qubits,
            'gate_count': self.gate_count
        }


@dataclass
class ProfileReport:
    """Comprehensive profiling report."""
    profile_data: ProfileData
    summary: Dict[str, Any] = field(default_factory=dict)
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def get_formatted_summary(self) -> str:
        """Get formatted text summary of the profile."""
        lines = [
            f"Profile Report for {self.profile_data.circuit_name}",
            "=" * 50,
            f"Total Execution Time: {self.profile_data.total_duration:.3f}s",
            f"  - Compilation: {self.profile_data.compilation_time:.3f}s ({self.profile_data.compilation_time/self.profile_data.total_duration*100:.1f}%)",
            f"  - Simulation: {self.profile_data.simulation_time:.3f}s ({self.profile_data.simulation_time/self.profile_data.total_duration*100:.1f}%)",
            f"  - Post-processing: {self.profile_data.postprocessing_time:.3f}s ({self.profile_data.postprocessing_time/self.profile_data.total_duration*100:.1f}%)",
            "",
            f"Resource Usage:",
            f"  - Peak Memory: {self.profile_data.peak_memory_mb:.1f}MB",
            f"  - Average CPU: {self.profile_data.avg_cpu_percent:.1f}%",
            "",
            f"Circuit Details:",
            f"  - Qubits: {self.profile_data.num_qubits}",
            f"  - Gates: {self.profile_data.gate_count}",
            f"  - Shots: {self.profile_data.shots}",
            f"  - Backend: {self.profile_data.backend_name}",
        ]
        
        if self.recommendations:
            lines.extend([
                "",
                "Recommendations:",
                *[f"  - {rec}" for rec in self.recommendations]
            ])
        
        return "\n".join(lines)


class QuantumProfiler:
    """
    Main quantum profiler that coordinates performance analysis across
    all components of quantum program execution.
    """
    
    def __init__(self, config: Optional[ProfilerConfig] = None):
        """
        Initialize the quantum profiler.
        
        Args:
            config: Profiler configuration
        """
        self.config = config or ProfilerConfig()
        self.logger = get_logger(__name__)
        self.monitor = get_monitor()
        
        # Active profiles
        self._active_profiles: Dict[str, ProfileData] = {}
        self._profile_history: List[ProfileData] = []
        self._lock = threading.RLock()
        
        # Sub-profilers (initialized lazily)
        self._simulation_profiler = None
        self._hardware_profiler = None
        self._compiler_profiler = None
        self._memory_profiler = None
        
        # Performance monitoring
        self._profiler_overhead_start = 0.0
        
        self.logger.info(f"QuantumProfiler initialized with mode: {self.config.mode.value}")
    
    def start_profile(self, profile_id: str, **metadata) -> ProfileData:
        """
        Start a new profiling session.
        
        Args:
            profile_id: Unique identifier for this profile
            **metadata: Additional metadata for the profile
            
        Returns:
            ProfileData object for this session
        """
        if self.config.mode == ProfilerMode.DISABLED:
            return None
        
        self._profiler_overhead_start = time.time()
        
        with self._lock:
            if profile_id in self._active_profiles:
                self.logger.warning(f"Profile {profile_id} already active, stopping previous")
                self.stop_profile(profile_id)
            
            profile_data = ProfileData(
                profile_id=profile_id,
                start_time=datetime.now(),
                **metadata
            )
            
            self._active_profiles[profile_id] = profile_data
            
            # Initialize sub-profilers if needed
            self._ensure_subprofilers_initialized()
            
            # Start sub-profilers
            if self.config.profile_simulation and self._simulation_profiler:
                self._simulation_profiler.start_profile(profile_id)
            
            if self.config.profile_hardware and self._hardware_profiler:
                self._hardware_profiler.start_profile(profile_id)
            
            if self.config.profile_compilation and self._compiler_profiler:
                self._compiler_profiler.start_profile(profile_id)
            
            if self.config.profile_memory and self._memory_profiler:
                self._memory_profiler.start_profile(profile_id)
            
            # Start cProfile if in developer mode
            if self.config.mode == ProfilerMode.DEVELOPER and self.config.use_cprofile:
                self._start_cprofile(profile_id)
            
            self.logger.debug(f"Started profiling session: {profile_id}")
            return profile_data
    
    def stop_profile(self, profile_id: str) -> Optional[ProfileData]:
        """
        Stop a profiling session and collect results.
        
        Args:
            profile_id: Profile identifier to stop
            
        Returns:
            Completed ProfileData with results
        """
        if self.config.mode == ProfilerMode.DISABLED:
            return None
        
        with self._lock:
            if profile_id not in self._active_profiles:
                self.logger.warning(f"Profile {profile_id} not found")
                return None
            
            profile_data = self._active_profiles[profile_id]
            profile_data.end_time = datetime.now()
            profile_data.total_duration = (profile_data.end_time - profile_data.start_time).total_seconds()
            
            # Collect results from sub-profilers
            if self.config.profile_simulation and self._simulation_profiler:
                profile_data.simulation_profile = self._simulation_profiler.stop_profile(profile_id)
                if profile_data.simulation_profile:
                    profile_data.simulation_time = profile_data.simulation_profile.total_time
            
            if self.config.profile_hardware and self._hardware_profiler:
                profile_data.hardware_profile = self._hardware_profiler.stop_profile(profile_id)
            
            if self.config.profile_compilation and self._compiler_profiler:
                profile_data.compiler_profile = self._compiler_profiler.stop_profile(profile_id)
                if profile_data.compiler_profile:
                    profile_data.compilation_time = profile_data.compiler_profile.total_time
            
            if self.config.profile_memory and self._memory_profiler:
                profile_data.memory_profile = self._memory_profiler.stop_profile(profile_id)
                if profile_data.memory_profile:
                    profile_data.peak_memory_mb = profile_data.memory_profile.peak_memory_mb
            
            # Stop cProfile if running
            if self.config.mode == ProfilerMode.DEVELOPER and self.config.use_cprofile:
                profile_data.cprofile_stats = self._stop_cprofile(profile_id)
            
            # Calculate post-processing time
            profile_data.postprocessing_time = max(0, 
                profile_data.total_duration - profile_data.compilation_time - profile_data.simulation_time
            )
            
            # Remove from active and add to history
            del self._active_profiles[profile_id]
            self._profile_history.append(profile_data)
            
            # Log profiler overhead
            overhead = time.time() - self._profiler_overhead_start
            overhead_percent = (overhead / profile_data.total_duration) * 100 if profile_data.total_duration > 0 else 0
            
            if overhead_percent > self.config.max_overhead_percent:
                self.logger.warning(f"Profiler overhead {overhead_percent:.1f}% exceeds limit {self.config.max_overhead_percent}%")
            
            self.logger.debug(f"Completed profiling session: {profile_id} (overhead: {overhead_percent:.1f}%)")
            return profile_data
    
    def generate_report(self, profile_data: ProfileData) -> ProfileReport:
        """
        Generate a comprehensive report from profile data.
        
        Args:
            profile_data: Profile data to analyze
            
        Returns:
            ProfileReport with analysis and recommendations
        """
        report = ProfileReport(profile_data=profile_data)
        
        # Generate summary
        report.summary = {
            'total_time': profile_data.total_duration,
            'time_breakdown': {
                'compilation': profile_data.compilation_time,
                'simulation': profile_data.simulation_time,
                'postprocessing': profile_data.postprocessing_time
            },
            'resource_usage': {
                'peak_memory_mb': profile_data.peak_memory_mb,
                'avg_cpu_percent': profile_data.avg_cpu_percent
            }
        }
        
        # Generate recommendations
        recommendations = []
        
        if profile_data.compilation_time > profile_data.simulation_time:
            recommendations.append("Consider caching compiled circuits to reduce compilation overhead")
        
        if profile_data.peak_memory_mb > 1000:  # > 1GB
            recommendations.append("High memory usage detected - consider reducing circuit size or using sparse simulation")
        
        if profile_data.avg_cpu_percent < 50 and profile_data.simulation_time > 1.0:
            recommendations.append("Low CPU utilization - consider enabling parallel execution")
        
        report.recommendations = recommendations
        
        return report
    
    def get_active_profiles(self) -> List[str]:
        """Get list of currently active profile IDs."""
        with self._lock:
            return list(self._active_profiles.keys())
    
    def get_profile_history(self, limit: Optional[int] = None) -> List[ProfileData]:
        """
        Get historical profile data.
        
        Args:
            limit: Maximum number of profiles to return
            
        Returns:
            List of historical ProfileData
        """
        with self._lock:
            history = sorted(self._profile_history, key=lambda p: p.start_time, reverse=True)
            return history[:limit] if limit else history
    
    def _ensure_subprofilers_initialized(self):
        """Initialize sub-profilers lazily."""
        if self.config.profile_simulation and not self._simulation_profiler:
            from quantum_platform.profiling.simulation_profiler import SimulationProfiler
            self._simulation_profiler = SimulationProfiler(self.config)
        
        if self.config.profile_hardware and not self._hardware_profiler:
            from quantum_platform.profiling.hardware_profiler import HardwareProfiler
            self._hardware_profiler = HardwareProfiler(self.config)
        
        if self.config.profile_compilation and not self._compiler_profiler:
            from quantum_platform.profiling.compiler_profiler import CompilerProfiler
            self._compiler_profiler = CompilerProfiler(self.config)
        
        if self.config.profile_memory and not self._memory_profiler:
            from quantum_platform.profiling.memory_profiler import MemoryProfiler
            self._memory_profiler = MemoryProfiler(self.config)
    
    def _start_cprofile(self, profile_id: str):
        """Start cProfile for developer-level profiling."""
        if not hasattr(self, '_cprofiles'):
            self._cprofiles = {}
        
        profiler = cProfile.Profile()
        profiler.enable()
        self._cprofiles[profile_id] = profiler
    
    def _stop_cprofile(self, profile_id: str) -> Optional[pstats.Stats]:
        """Stop cProfile and return stats."""
        if not hasattr(self, '_cprofiles') or profile_id not in self._cprofiles:
            return None
        
        profiler = self._cprofiles[profile_id]
        profiler.disable()
        
        # Convert to stats
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s)
        del self._cprofiles[profile_id]
        
        return stats


# Global profiler instance
_global_profiler: Optional[QuantumProfiler] = None


def get_profiler(config: Optional[ProfilerConfig] = None) -> QuantumProfiler:
    """Get the global quantum profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = QuantumProfiler(config)
    return _global_profiler


def configure_profiler(config: ProfilerConfig):
    """Configure the global profiler."""
    global _global_profiler
    _global_profiler = QuantumProfiler(config)


# Context manager for easy profiling
class profile_execution:
    """Context manager for profiling quantum execution."""
    
    def __init__(self, profile_id: str, **metadata):
        self.profile_id = profile_id
        self.metadata = metadata
        self.profiler = get_profiler()
        self.profile_data = None
    
    def __enter__(self):
        self.profile_data = self.profiler.start_profile(self.profile_id, **self.metadata)
        return self.profile_data
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profile_data:
            self.profiler.stop_profile(self.profile_id) 