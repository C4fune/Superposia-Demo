"""
Memory Usage Profiler

This module provides detailed memory profiling for quantum operations,
tracking memory consumption patterns and identifying memory bottlenecks.
"""

import time
import threading
import psutil
import gc
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from quantum_platform.observability.logging import get_logger


@dataclass
class MemorySnapshot:
    """A snapshot of memory usage at a point in time."""
    timestamp: datetime
    
    # System memory
    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    memory_percent: float = 0.0
    
    # Process memory
    process_memory_mb: float = 0.0
    process_memory_percent: float = 0.0
    
    # Quantum-specific memory
    state_vector_memory_mb: float = 0.0
    circuit_memory_mb: float = 0.0
    result_memory_mb: float = 0.0
    
    # Memory allocation details
    heap_objects: int = 0
    gc_collections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_memory_mb': self.total_memory_mb,
            'used_memory_mb': self.used_memory_mb,
            'available_memory_mb': self.available_memory_mb,
            'memory_percent': self.memory_percent,
            'process_memory_mb': self.process_memory_mb,
            'process_memory_percent': self.process_memory_percent,
            'state_vector_memory_mb': self.state_vector_memory_mb,
            'circuit_memory_mb': self.circuit_memory_mb,
            'result_memory_mb': self.result_memory_mb,
            'heap_objects': self.heap_objects,
            'gc_collections': self.gc_collections
        }


@dataclass
class MemoryUsage:
    """Memory usage statistics over time."""
    
    # Peak usage
    peak_total_memory_mb: float = 0.0
    peak_process_memory_mb: float = 0.0
    peak_state_vector_memory_mb: float = 0.0
    
    # Average usage
    avg_memory_mb: float = 0.0
    avg_memory_percent: float = 0.0
    
    # Memory efficiency
    memory_efficiency: float = 0.0  # Actual vs theoretical minimum
    memory_waste: float = 0.0       # Allocated but unused
    
    # Growth patterns
    memory_growth_rate: float = 0.0  # MB per second
    peak_memory_time: Optional[datetime] = None
    
    # Garbage collection impact
    gc_overhead_percent: float = 0.0
    
    def calculate_statistics(self, snapshots: List[MemorySnapshot], theoretical_min_mb: float = 0.0):
        """Calculate statistics from memory snapshots."""
        if not snapshots:
            return
        
        # Peak values
        self.peak_total_memory_mb = max(s.used_memory_mb for s in snapshots)
        self.peak_process_memory_mb = max(s.process_memory_mb for s in snapshots)
        self.peak_state_vector_memory_mb = max(s.state_vector_memory_mb for s in snapshots)
        
        # Average values
        self.avg_memory_mb = sum(s.process_memory_mb for s in snapshots) / len(snapshots)
        self.avg_memory_percent = sum(s.memory_percent for s in snapshots) / len(snapshots)
        
        # Find peak memory time
        peak_snapshot = max(snapshots, key=lambda s: s.process_memory_mb)
        self.peak_memory_time = peak_snapshot.timestamp
        
        # Calculate efficiency
        if theoretical_min_mb > 0:
            self.memory_efficiency = theoretical_min_mb / self.peak_process_memory_mb
            self.memory_waste = self.peak_process_memory_mb - theoretical_min_mb
        
        # Calculate growth rate
        if len(snapshots) > 1:
            time_span = (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds()
            if time_span > 0:
                memory_change = snapshots[-1].process_memory_mb - snapshots[0].process_memory_mb
                self.memory_growth_rate = memory_change / time_span
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'peak_total_memory_mb': self.peak_total_memory_mb,
            'peak_process_memory_mb': self.peak_process_memory_mb,
            'peak_state_vector_memory_mb': self.peak_state_vector_memory_mb,
            'avg_memory_mb': self.avg_memory_mb,
            'avg_memory_percent': self.avg_memory_percent,
            'memory_efficiency': self.memory_efficiency,
            'memory_waste': self.memory_waste,
            'memory_growth_rate': self.memory_growth_rate,
            'peak_memory_time': self.peak_memory_time.isoformat() if self.peak_memory_time else None,
            'gc_overhead_percent': self.gc_overhead_percent
        }


@dataclass
class MemoryProfile:
    """Complete memory usage profile."""
    profile_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Memory snapshots
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    
    # Summary statistics
    memory_usage: MemoryUsage = field(default_factory=MemoryUsage)
    
    # Operation context
    operation_type: str = ""  # simulation, compilation, etc.
    num_qubits: int = 0
    theoretical_memory_mb: float = 0.0
    
    # Memory events
    memory_warnings: List[str] = field(default_factory=list)
    oom_risks: List[str] = field(default_factory=list)  # Out of memory risks
    
    def calculate_summary(self):
        """Calculate summary statistics from snapshots."""
        self.memory_usage.calculate_statistics(self.snapshots, self.theoretical_memory_mb)
    
    def get_memory_timeline(self) -> List[Tuple[float, float]]:
        """Get memory usage timeline as (seconds_from_start, memory_mb) pairs."""
        if not self.snapshots:
            return []
        
        start_time = self.snapshots[0].timestamp
        return [
            ((s.timestamp - start_time).total_seconds(), s.process_memory_mb)
            for s in self.snapshots
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'profile_id': self.profile_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'operation_type': self.operation_type,
            'num_qubits': self.num_qubits,
            'theoretical_memory_mb': self.theoretical_memory_mb,
            'memory_usage': self.memory_usage.to_dict(),
            'snapshot_count': len(self.snapshots),
            'memory_warnings': self.memory_warnings,
            'oom_risks': self.oom_risks
        }


class MemoryProfiler:
    """
    Profiler for memory usage during quantum operations.
    
    Tracks memory consumption patterns, identifies bottlenecks,
    and provides memory optimization recommendations.
    """
    
    def __init__(self, config):
        """Initialize the memory profiler."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Active profiles
        self._active_profiles: Dict[str, MemoryProfile] = {}
        self._profile_history: List[MemoryProfile] = []
        self._lock = threading.RLock()
        
        # Monitoring thread
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._current_profile_id: Optional[str] = None
        
        # Process handle for memory monitoring
        self._process = psutil.Process()
        
        self.logger.debug("MemoryProfiler initialized")
    
    def start_profile(self, profile_id: str, operation_type: str = "", 
                     num_qubits: int = 0) -> MemoryProfile:
        """Start profiling memory usage."""
        with self._lock:
            if profile_id in self._active_profiles:
                self.logger.warning(f"Memory profile {profile_id} already active")
                return self._active_profiles[profile_id]
            
            # Calculate theoretical memory requirement
            theoretical_memory = 0.0
            if num_qubits > 0 and operation_type == "simulation":
                # State vector: 2^n complex numbers * 16 bytes each / 1MB
                theoretical_memory = (2 ** num_qubits) * 16 / (1024 ** 2)
            
            profile = MemoryProfile(
                profile_id=profile_id,
                start_time=datetime.now(),
                operation_type=operation_type,
                num_qubits=num_qubits,
                theoretical_memory_mb=theoretical_memory
            )
            
            self._active_profiles[profile_id] = profile
            self._current_profile_id = profile_id
            
            # Start monitoring if configured
            if self.config.track_memory_usage and not self._monitoring_thread:
                self._start_monitoring()
            
            self.logger.debug(f"Started memory profiling: {profile_id}")
            return profile
    
    def stop_profile(self, profile_id: str) -> Optional[MemoryProfile]:
        """Stop profiling and return results."""
        with self._lock:
            if profile_id not in self._active_profiles:
                self.logger.warning(f"Memory profile {profile_id} not found")
                return None
            
            profile = self._active_profiles[profile_id]
            profile.end_time = datetime.now()
            
            # Calculate summary statistics
            profile.calculate_summary()
            
            # Stop monitoring if this was the last profile
            del self._active_profiles[profile_id]
            if not self._active_profiles and self._monitoring_thread:
                self._stop_monitoring_thread()
            
            if self._current_profile_id == profile_id:
                self._current_profile_id = None
            
            self._profile_history.append(profile)
            
            self.logger.debug(f"Completed memory profiling: {profile_id}")
            return profile
    
    def _start_monitoring(self):
        """Start the memory monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.debug("Started memory monitoring thread")
    
    def _stop_monitoring_thread(self):
        """Stop the memory monitoring thread."""
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=1.0)
            self._monitoring_thread = None
            
            self.logger.debug("Stopped memory monitoring thread")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                self._take_memory_snapshot()
                self._stop_monitoring.wait(self.config.memory_sample_interval)
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
                break
    
    def _take_memory_snapshot(self):
        """Take a snapshot of current memory usage."""
        if not self._current_profile_id:
            return
        
        try:
            # System memory info
            virtual_memory = psutil.virtual_memory()
            
            # Process memory info
            process_memory = self._process.memory_info()
            
            # Garbage collection info
            gc_stats = gc.get_stats()
            total_collections = sum(stat['collections'] for stat in gc_stats)
            
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                total_memory_mb=virtual_memory.total / (1024 ** 2),
                used_memory_mb=virtual_memory.used / (1024 ** 2),
                available_memory_mb=virtual_memory.available / (1024 ** 2),
                memory_percent=virtual_memory.percent,
                process_memory_mb=process_memory.rss / (1024 ** 2),
                process_memory_percent=self._process.memory_percent(),
                heap_objects=len(gc.get_objects()),
                gc_collections=total_collections
            )
            
            with self._lock:
                if self._current_profile_id in self._active_profiles:
                    profile = self._active_profiles[self._current_profile_id]
                    profile.snapshots.append(snapshot)
                    
                    # Check for memory warnings
                    self._check_memory_warnings(profile, snapshot)
        
        except Exception as e:
            self.logger.error(f"Failed to take memory snapshot: {e}")
    
    def _check_memory_warnings(self, profile: MemoryProfile, snapshot: MemorySnapshot):
        """Check for memory-related warnings."""
        # High memory usage warning
        if snapshot.memory_percent > 85:
            warning = f"High system memory usage: {snapshot.memory_percent:.1f}%"
            if warning not in profile.memory_warnings:
                profile.memory_warnings.append(warning)
                self.logger.warning(f"Memory warning [{profile.profile_id}]: {warning}")
        
        # Out of memory risk
        if snapshot.memory_percent > 95:
            risk = f"Critical memory usage: {snapshot.memory_percent:.1f}% - OOM risk"
            if risk not in profile.oom_risks:
                profile.oom_risks.append(risk)
                self.logger.error(f"Memory critical [{profile.profile_id}]: {risk}")
        
        # Memory growth warning
        if len(profile.snapshots) > 10:
            recent_snapshots = profile.snapshots[-10:]
            growth_rate = (recent_snapshots[-1].process_memory_mb - recent_snapshots[0].process_memory_mb) / 10
            
            if growth_rate > 10:  # Growing by >10MB per snapshot
                warning = f"Rapid memory growth: {growth_rate:.1f}MB per interval"
                if warning not in profile.memory_warnings:
                    profile.memory_warnings.append(warning)
    
    def record_quantum_memory(self, state_vector_mb: float = 0.0, 
                            circuit_mb: float = 0.0, result_mb: float = 0.0):
        """Record quantum-specific memory usage."""
        if not self._current_profile_id:
            return
        
        with self._lock:
            if (self._current_profile_id in self._active_profiles and 
                self._active_profiles[self._current_profile_id].snapshots):
                
                # Update the latest snapshot
                latest_snapshot = self._active_profiles[self._current_profile_id].snapshots[-1]
                latest_snapshot.state_vector_memory_mb = state_vector_mb
                latest_snapshot.circuit_memory_mb = circuit_mb
                latest_snapshot.result_memory_mb = result_mb
    
    def get_current_profile(self) -> Optional[MemoryProfile]:
        """Get the currently active profile."""
        if not self._current_profile_id:
            return None
        
        with self._lock:
            return self._active_profiles.get(self._current_profile_id)
    
    def get_profile_history(self, limit: Optional[int] = None) -> List[MemoryProfile]:
        """Get historical memory profiles."""
        with self._lock:
            history = sorted(self._profile_history, key=lambda p: p.start_time, reverse=True)
            return history[:limit] if limit else history 