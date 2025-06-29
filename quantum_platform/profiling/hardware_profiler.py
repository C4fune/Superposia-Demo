"""
Hardware Performance Profiler

This module provides profiling for quantum hardware jobs, tracking queue times,
execution times, network latency, and provider-specific performance metrics.
"""

import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from quantum_platform.observability.logging import get_logger


class HardwareProvider(Enum):
    """Supported hardware providers."""
    IBM = "ibm"
    GOOGLE = "google"
    IONQ = "ionq"
    RIGETTI = "rigetti"
    AWS_BRAKET = "aws_braket"
    AZURE = "azure"
    HONEYWELL = "honeywell"
    XANADU = "xanadu"


@dataclass
class ProviderTiming:
    """Provider-specific timing information."""
    provider: HardwareProvider
    device_name: str = ""
    
    # Queue information
    initial_queue_position: Optional[int] = None
    final_queue_position: Optional[int] = None
    estimated_queue_time: Optional[timedelta] = None
    actual_queue_time: Optional[timedelta] = None
    
    # Execution timing
    job_submission_time: Optional[datetime] = None
    execution_start_time: Optional[datetime] = None
    execution_end_time: Optional[datetime] = None
    result_retrieval_time: Optional[datetime] = None
    
    # Provider-reported metrics
    provider_execution_time: Optional[float] = None  # Time reported by provider
    provider_queue_time: Optional[float] = None
    
    # Network timing
    submission_latency: Optional[float] = None
    retrieval_latency: Optional[float] = None
    
    # Device information
    device_properties: Dict[str, Any] = field(default_factory=dict)
    calibration_data: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_derived_timings(self):
        """Calculate derived timing metrics."""
        if self.job_submission_time and self.execution_start_time:
            self.actual_queue_time = self.execution_start_time - self.job_submission_time
        
        if self.execution_start_time and self.execution_end_time:
            if not self.provider_execution_time:
                self.provider_execution_time = (self.execution_end_time - self.execution_start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'provider': self.provider.value,
            'device_name': self.device_name,
            'initial_queue_position': self.initial_queue_position,
            'actual_queue_time_seconds': self.actual_queue_time.total_seconds() if self.actual_queue_time else None,
            'provider_execution_time': self.provider_execution_time,
            'provider_queue_time': self.provider_queue_time,
            'submission_latency': self.submission_latency,
            'retrieval_latency': self.retrieval_latency,
            'device_properties': self.device_properties
        }


@dataclass
class HardwareTiming:
    """Comprehensive hardware job timing breakdown."""
    
    # Overall timing
    total_wall_time: float = 0.0  # From user submission to result retrieval
    
    # Phase breakdown
    preparation_time: float = 0.0      # Circuit compilation and preparation
    submission_time: float = 0.0       # Time to submit job to provider
    queue_wait_time: float = 0.0       # Time waiting in queue
    execution_time: float = 0.0        # Actual execution on hardware
    retrieval_time: float = 0.0        # Time to retrieve results
    postprocessing_time: float = 0.0   # Result processing and formatting
    
    # Network timing
    network_overhead: float = 0.0      # Total network latency
    
    # Efficiency metrics
    utilization_ratio: float = 0.0     # execution_time / total_wall_time
    queue_accuracy: float = 0.0        # How accurate queue time estimates were
    
    def calculate_efficiency_metrics(self):
        """Calculate efficiency and utilization metrics."""
        if self.total_wall_time > 0:
            self.utilization_ratio = self.execution_time / self.total_wall_time
        
        self.network_overhead = self.submission_time + self.retrieval_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'total_wall_time': self.total_wall_time,
            'preparation_time': self.preparation_time,
            'submission_time': self.submission_time,
            'queue_wait_time': self.queue_wait_time,
            'execution_time': self.execution_time,
            'retrieval_time': self.retrieval_time,
            'postprocessing_time': self.postprocessing_time,
            'network_overhead': self.network_overhead,
            'utilization_ratio': self.utilization_ratio,
            'queue_accuracy': self.queue_accuracy
        }


@dataclass
class HardwareProfile:
    """Complete hardware job performance profile."""
    profile_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Job information
    job_id: str = ""
    circuit_name: str = ""
    backend_name: str = ""
    shots: int = 0
    
    # Provider-specific timing
    provider_timing: Optional[ProviderTiming] = None
    
    # Comprehensive timing breakdown
    hardware_timing: HardwareTiming = field(default_factory=HardwareTiming)
    
    # Performance metrics
    cost_per_shot: Optional[float] = None
    cost_per_second: Optional[float] = None
    throughput_shots_per_hour: float = 0.0
    
    # Quality metrics
    fidelity_estimate: Optional[float] = None
    error_rates: Dict[str, float] = field(default_factory=dict)
    
    # Comparison with simulation
    simulation_time_estimate: Optional[float] = None
    hardware_vs_simulation_ratio: Optional[float] = None
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def calculate_performance_metrics(self):
        """Calculate derived performance metrics."""
        if self.provider_timing:
            self.provider_timing.calculate_derived_timings()
        
        self.hardware_timing.calculate_efficiency_metrics()
        
        # Calculate throughput
        if self.hardware_timing.total_wall_time > 0:
            hours = self.hardware_timing.total_wall_time / 3600
            self.throughput_shots_per_hour = self.shots / hours if hours > 0 else 0
        
        # Calculate hardware vs simulation ratio
        if self.simulation_time_estimate and self.hardware_timing.execution_time > 0:
            self.hardware_vs_simulation_ratio = self.hardware_timing.execution_time / self.simulation_time_estimate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'profile_id': self.profile_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'job_id': self.job_id,
            'circuit_name': self.circuit_name,
            'backend_name': self.backend_name,
            'shots': self.shots,
            'provider_timing': self.provider_timing.to_dict() if self.provider_timing else None,
            'hardware_timing': self.hardware_timing.to_dict(),
            'throughput_shots_per_hour': self.throughput_shots_per_hour,
            'hardware_vs_simulation_ratio': self.hardware_vs_simulation_ratio,
            'errors': self.errors,
            'warnings': self.warnings
        }


class HardwareProfiler:
    """
    Profiler for quantum hardware job performance analysis.
    
    Tracks timing, efficiency, and cost metrics for quantum hardware executions
    across different providers and devices.
    """
    
    def __init__(self, config):
        """Initialize the hardware profiler."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Active profiles
        self._active_profiles: Dict[str, HardwareProfile] = {}
        self._profile_history: List[HardwareProfile] = []
        self._lock = threading.RLock()
        
        # Timing context
        self._phase_start_times: Dict[str, Dict[str, float]] = {}
        
        self.logger.debug("HardwareProfiler initialized")
    
    def start_profile(self, profile_id: str, job_id: str = "", backend_name: str = "", 
                     circuit_name: str = "", shots: int = 0) -> HardwareProfile:
        """Start profiling a hardware job."""
        with self._lock:
            if profile_id in self._active_profiles:
                self.logger.warning(f"Hardware profile {profile_id} already active")
                return self._active_profiles[profile_id]
            
            profile = HardwareProfile(
                profile_id=profile_id,
                start_time=datetime.now(),
                job_id=job_id,
                backend_name=backend_name,
                circuit_name=circuit_name,
                shots=shots
            )
            
            self._active_profiles[profile_id] = profile
            self._phase_start_times[profile_id] = {}
            
            self.logger.debug(f"Started hardware profiling: {profile_id}")
            return profile
    
    def stop_profile(self, profile_id: str) -> Optional[HardwareProfile]:
        """Stop profiling and return results."""
        with self._lock:
            if profile_id not in self._active_profiles:
                self.logger.warning(f"Hardware profile {profile_id} not found")
                return None
            
            profile = self._active_profiles[profile_id]
            profile.end_time = datetime.now()
            
            # Calculate total wall time
            profile.hardware_timing.total_wall_time = (
                profile.end_time - profile.start_time
            ).total_seconds()
            
            # Calculate performance metrics
            profile.calculate_performance_metrics()
            
            # Clean up and move to history
            del self._active_profiles[profile_id]
            if profile_id in self._phase_start_times:
                del self._phase_start_times[profile_id]
            
            self._profile_history.append(profile)
            
            self.logger.debug(f"Completed hardware profiling: {profile_id}")
            return profile
    
    def start_phase(self, profile_id: str, phase_name: str):
        """Start timing a hardware execution phase."""
        if profile_id not in self._active_profiles:
            return
        
        if profile_id not in self._phase_start_times:
            self._phase_start_times[profile_id] = {}
        
        self._phase_start_times[profile_id][phase_name] = time.time()
    
    def end_phase(self, profile_id: str, phase_name: str):
        """End timing a hardware execution phase."""
        if (profile_id not in self._active_profiles or 
            profile_id not in self._phase_start_times or 
            phase_name not in self._phase_start_times[profile_id]):
            return
        
        duration = time.time() - self._phase_start_times[profile_id][phase_name]
        del self._phase_start_times[profile_id][phase_name]
        
        with self._lock:
            profile = self._active_profiles[profile_id]
            timing = profile.hardware_timing
            
            if phase_name == "preparation":
                timing.preparation_time += duration
            elif phase_name == "submission":
                timing.submission_time += duration
            elif phase_name == "queue_wait":
                timing.queue_wait_time += duration
            elif phase_name == "execution":
                timing.execution_time += duration
            elif phase_name == "retrieval":
                timing.retrieval_time += duration
            elif phase_name == "postprocessing":
                timing.postprocessing_time += duration
    
    def set_provider_info(self, profile_id: str, provider: HardwareProvider, 
                         device_name: str, **provider_data):
        """Set provider-specific information."""
        if profile_id not in self._active_profiles:
            return
        
        with self._lock:
            profile = self._active_profiles[profile_id]
            if not profile.provider_timing:
                profile.provider_timing = ProviderTiming(provider=provider, device_name=device_name)
            
            # Update provider timing with additional data
            for key, value in provider_data.items():
                if hasattr(profile.provider_timing, key):
                    setattr(profile.provider_timing, key, value)
    
    def record_queue_update(self, profile_id: str, queue_position: int, 
                           estimated_time: Optional[timedelta] = None):
        """Record queue position update."""
        if profile_id not in self._active_profiles:
            return
        
        with self._lock:
            profile = self._active_profiles[profile_id]
            if profile.provider_timing:
                if profile.provider_timing.initial_queue_position is None:
                    profile.provider_timing.initial_queue_position = queue_position
                
                profile.provider_timing.final_queue_position = queue_position
                if estimated_time:
                    profile.provider_timing.estimated_queue_time = estimated_time
    
    def record_execution_timing(self, profile_id: str, 
                               execution_start: Optional[datetime] = None,
                               execution_end: Optional[datetime] = None,
                               provider_reported_time: Optional[float] = None):
        """Record execution timing information."""
        if profile_id not in self._active_profiles:
            return
        
        with self._lock:
            profile = self._active_profiles[profile_id]
            if profile.provider_timing:
                if execution_start:
                    profile.provider_timing.execution_start_time = execution_start
                if execution_end:
                    profile.provider_timing.execution_end_time = execution_end
                if provider_reported_time:
                    profile.provider_timing.provider_execution_time = provider_reported_time
    
    def record_network_latency(self, profile_id: str, 
                              submission_latency: Optional[float] = None,
                              retrieval_latency: Optional[float] = None):
        """Record network latency measurements."""
        if profile_id not in self._active_profiles:
            return
        
        with self._lock:
            profile = self._active_profiles[profile_id]
            if profile.provider_timing:
                if submission_latency:
                    profile.provider_timing.submission_latency = submission_latency
                if retrieval_latency:
                    profile.provider_timing.retrieval_latency = retrieval_latency
    
    def record_error(self, profile_id: str, error: str):
        """Record an error during hardware execution."""
        if profile_id not in self._active_profiles:
            return
        
        with self._lock:
            profile = self._active_profiles[profile_id]
            profile.errors.append(error)
            self.logger.error(f"Hardware execution error [{profile_id}]: {error}")
    
    def record_warning(self, profile_id: str, warning: str):
        """Record a warning during hardware execution."""
        if profile_id not in self._active_profiles:
            return
        
        with self._lock:
            profile = self._active_profiles[profile_id]
            profile.warnings.append(warning)
            self.logger.warning(f"Hardware execution warning [{profile_id}]: {warning}")
    
    def set_simulation_comparison(self, profile_id: str, simulation_time: float):
        """Set simulation time for comparison."""
        if profile_id not in self._active_profiles:
            return
        
        with self._lock:
            profile = self._active_profiles[profile_id]
            profile.simulation_time_estimate = simulation_time
    
    def get_current_profile(self, profile_id: str) -> Optional[HardwareProfile]:
        """Get the currently active profile."""
        with self._lock:
            return self._active_profiles.get(profile_id)
    
    def get_profile_history(self, limit: Optional[int] = None) -> List[HardwareProfile]:
        """Get historical hardware profiles."""
        with self._lock:
            history = sorted(self._profile_history, key=lambda p: p.start_time, reverse=True)
            return history[:limit] if limit else history
    
    def get_provider_statistics(self, provider: HardwareProvider) -> Dict[str, Any]:
        """Get performance statistics for a specific provider."""
        with self._lock:
            provider_profiles = [
                p for p in self._profile_history 
                if p.provider_timing and p.provider_timing.provider == provider
            ]
            
            if not provider_profiles:
                return {}
            
            total_jobs = len(provider_profiles)
            avg_queue_time = sum(
                p.hardware_timing.queue_wait_time for p in provider_profiles
            ) / total_jobs
            
            avg_execution_time = sum(
                p.hardware_timing.execution_time for p in provider_profiles
            ) / total_jobs
            
            avg_utilization = sum(
                p.hardware_timing.utilization_ratio for p in provider_profiles
            ) / total_jobs
            
            return {
                'provider': provider.value,
                'total_jobs': total_jobs,
                'avg_queue_time_seconds': avg_queue_time,
                'avg_execution_time_seconds': avg_execution_time,
                'avg_utilization_ratio': avg_utilization,
                'recent_profiles': len([p for p in provider_profiles[-10:]]),
            } 