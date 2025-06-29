"""
Simulation Performance Profiler

This module provides detailed profiling of quantum simulation execution,
tracking gate application times, state manipulation performance, and
measurement sampling efficiency.
"""

import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict, Counter
import numpy as np

from quantum_platform.observability.logging import get_logger


@dataclass
class GateProfile:
    """Profile data for individual gate operations."""
    gate_name: str
    gate_type: str  # single, two_qubit, multi_qubit, controlled
    qubits: List[int]
    
    # Timing information
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Operation details
    matrix_size: int = 0
    state_vector_size: int = 0
    memory_impact: float = 0.0  # Memory delta in MB
    
    # Performance metrics
    operations_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'gate_name': self.gate_name,
            'gate_type': self.gate_type,
            'qubits': self.qubits,
            'execution_time': self.execution_time,
            'matrix_size': self.matrix_size,
            'state_vector_size': self.state_vector_size,
            'memory_impact': self.memory_impact,
            'operations_per_second': self.operations_per_second
        }


@dataclass
class ExecutionBreakdown:
    """Breakdown of simulation execution phases."""
    
    # Phase timings
    initialization_time: float = 0.0
    gate_application_time: float = 0.0
    measurement_time: float = 0.0
    state_preparation_time: float = 0.0
    finalization_time: float = 0.0
    
    # Detailed gate timing
    single_qubit_gates_time: float = 0.0
    two_qubit_gates_time: float = 0.0
    multi_qubit_gates_time: float = 0.0
    controlled_gates_time: float = 0.0
    
    # Shot execution
    total_shots: int = 0
    shot_execution_times: List[float] = field(default_factory=list)
    avg_shot_time: float = 0.0
    
    # Memory operations
    state_vector_operations: int = 0
    matrix_multiplications: int = 0
    probability_calculations: int = 0
    
    def calculate_statistics(self):
        """Calculate derived statistics."""
        if self.shot_execution_times:
            self.avg_shot_time = sum(self.shot_execution_times) / len(self.shot_execution_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'initialization_time': self.initialization_time,
            'gate_application_time': self.gate_application_time,
            'measurement_time': self.measurement_time,
            'state_preparation_time': self.state_preparation_time,
            'finalization_time': self.finalization_time,
            'single_qubit_gates_time': self.single_qubit_gates_time,
            'two_qubit_gates_time': self.two_qubit_gates_time,
            'multi_qubit_gates_time': self.multi_qubit_gates_time,
            'controlled_gates_time': self.controlled_gates_time,
            'total_shots': self.total_shots,
            'avg_shot_time': self.avg_shot_time,
            'state_vector_operations': self.state_vector_operations,
            'matrix_multiplications': self.matrix_multiplications,
            'probability_calculations': self.probability_calculations
        }


@dataclass
class SimulationProfile:
    """Complete simulation performance profile."""
    profile_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_time: float = 0.0
    
    # Circuit information
    circuit_name: str = ""
    num_qubits: int = 0
    total_gates: int = 0
    circuit_depth: int = 0
    
    # Execution breakdown
    execution_breakdown: ExecutionBreakdown = field(default_factory=ExecutionBreakdown)
    
    # Gate profiles
    gate_profiles: List[GateProfile] = field(default_factory=list)
    gate_type_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance metrics
    gates_per_second: float = 0.0
    shots_per_second: float = 0.0
    state_vector_ops_per_second: float = 0.0
    
    # Memory and efficiency
    peak_memory_mb: float = 0.0
    memory_efficiency: float = 0.0  # Actual usage vs theoretical minimum
    cpu_efficiency: float = 0.0     # CPU utilization during simulation
    
    # Error analysis
    numerical_errors: List[str] = field(default_factory=list)
    performance_warnings: List[str] = field(default_factory=list)
    
    def calculate_summary_statistics(self):
        """Calculate summary statistics from detailed profiles."""
        if self.total_time > 0:
            self.gates_per_second = self.total_gates / self.total_time
            if self.execution_breakdown.total_shots > 0:
                self.shots_per_second = self.execution_breakdown.total_shots / self.total_time
            self.state_vector_ops_per_second = self.execution_breakdown.state_vector_operations / self.total_time
        
        # Calculate gate type summary
        gate_type_counts = Counter()
        gate_type_times = defaultdict(float)
        
        for gate_profile in self.gate_profiles:
            gate_type_counts[gate_profile.gate_type] += 1
            gate_type_times[gate_profile.gate_type] += gate_profile.execution_time
        
        for gate_type in gate_type_counts:
            self.gate_type_summary[gate_type] = {
                'count': gate_type_counts[gate_type],
                'total_time': gate_type_times[gate_type],
                'avg_time': gate_type_times[gate_type] / gate_type_counts[gate_type],
                'percentage': (gate_type_times[gate_type] / self.total_time) * 100 if self.total_time > 0 else 0
            }
    
    def get_top_slowest_gates(self, limit: int = 10) -> List[GateProfile]:
        """Get the slowest gate operations."""
        return sorted(self.gate_profiles, key=lambda g: g.execution_time, reverse=True)[:limit]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'profile_id': self.profile_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_time': self.total_time,
            'circuit_name': self.circuit_name,
            'num_qubits': self.num_qubits,
            'total_gates': self.total_gates,
            'circuit_depth': self.circuit_depth,
            'gates_per_second': self.gates_per_second,
            'shots_per_second': self.shots_per_second,
            'peak_memory_mb': self.peak_memory_mb,
            'execution_breakdown': self.execution_breakdown.to_dict(),
            'gate_type_summary': self.gate_type_summary,
            'performance_warnings': self.performance_warnings
        }


class SimulationProfiler:
    """
    Profiler for quantum simulation performance analysis.
    
    Tracks detailed timing information for gate operations, state manipulations,
    measurements, and overall simulation performance.
    """
    
    def __init__(self, config):
        """Initialize the simulation profiler."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Active profiles
        self._active_profiles: Dict[str, SimulationProfile] = {}
        self._profile_history: List[SimulationProfile] = []
        self._lock = threading.RLock()
        
        # Current profiling context
        self._current_profile_id: Optional[str] = None
        self._phase_start_times: Dict[str, float] = {}
        self._gate_start_time: Optional[float] = None
        self._shot_start_time: Optional[float] = None
        
        self.logger.debug("SimulationProfiler initialized")
    
    def start_profile(self, profile_id: str) -> SimulationProfile:
        """Start profiling a simulation."""
        with self._lock:
            if profile_id in self._active_profiles:
                self.logger.warning(f"Simulation profile {profile_id} already active")
                return self._active_profiles[profile_id]
            
            profile = SimulationProfile(
                profile_id=profile_id,
                start_time=datetime.now()
            )
            
            self._active_profiles[profile_id] = profile
            self._current_profile_id = profile_id
            
            self.logger.debug(f"Started simulation profiling: {profile_id}")
            return profile
    
    def stop_profile(self, profile_id: str) -> Optional[SimulationProfile]:
        """Stop profiling and return results."""
        with self._lock:
            if profile_id not in self._active_profiles:
                self.logger.warning(f"Simulation profile {profile_id} not found")
                return None
            
            profile = self._active_profiles[profile_id]
            profile.end_time = datetime.now()
            profile.total_time = (profile.end_time - profile.start_time).total_seconds()
            
            # Calculate summary statistics
            profile.calculate_summary_statistics()
            profile.execution_breakdown.calculate_statistics()
            
            # Move to history
            del self._active_profiles[profile_id]
            self._profile_history.append(profile)
            
            if self._current_profile_id == profile_id:
                self._current_profile_id = None
            
            self.logger.debug(f"Completed simulation profiling: {profile_id}")
            return profile
    
    def set_circuit_info(self, circuit_name: str, num_qubits: int, total_gates: int, circuit_depth: int):
        """Set circuit information for the current profile."""
        if not self._current_profile_id:
            return
        
        with self._lock:
            if self._current_profile_id in self._active_profiles:
                profile = self._active_profiles[self._current_profile_id]
                profile.circuit_name = circuit_name
                profile.num_qubits = num_qubits
                profile.total_gates = total_gates
                profile.circuit_depth = circuit_depth
    
    def start_phase(self, phase_name: str):
        """Start timing a simulation phase."""
        if not self._current_profile_id or not self.config.track_gate_timing:
            return
        
        self._phase_start_times[phase_name] = time.time()
    
    def end_phase(self, phase_name: str):
        """End timing a simulation phase."""
        if not self._current_profile_id or not self.config.track_gate_timing:
            return
        
        if phase_name not in self._phase_start_times:
            return
        
        duration = time.time() - self._phase_start_times[phase_name]
        del self._phase_start_times[phase_name]
        
        with self._lock:
            if self._current_profile_id in self._active_profiles:
                profile = self._active_profiles[self._current_profile_id]
                breakdown = profile.execution_breakdown
                
                if phase_name == "initialization":
                    breakdown.initialization_time += duration
                elif phase_name == "gate_application":
                    breakdown.gate_application_time += duration
                elif phase_name == "measurement":
                    breakdown.measurement_time += duration
                elif phase_name == "state_preparation":
                    breakdown.state_preparation_time += duration
                elif phase_name == "finalization":
                    breakdown.finalization_time += duration
    
    def start_gate_timing(self, gate_name: str, gate_type: str, qubits: List[int], matrix_size: int = 0):
        """Start timing a gate operation."""
        if not self._current_profile_id or not self.config.track_gate_timing:
            return
        
        self._gate_start_time = time.time()
        self._current_gate_info = {
            'gate_name': gate_name,
            'gate_type': gate_type,
            'qubits': qubits,
            'matrix_size': matrix_size,
            'start_time': datetime.now()
        }
    
    def end_gate_timing(self, state_vector_size: int = 0, memory_impact: float = 0.0):
        """End timing a gate operation."""
        if not self._current_profile_id or not self.config.track_gate_timing or not self._gate_start_time:
            return
        
        duration = time.time() - self._gate_start_time
        
        with self._lock:
            if self._current_profile_id in self._active_profiles:
                profile = self._active_profiles[self._current_profile_id]
                
                gate_profile = GateProfile(
                    gate_name=self._current_gate_info['gate_name'],
                    gate_type=self._current_gate_info['gate_type'],
                    qubits=self._current_gate_info['qubits'],
                    execution_time=duration,
                    start_time=self._current_gate_info['start_time'],
                    end_time=datetime.now(),
                    matrix_size=self._current_gate_info['matrix_size'],
                    state_vector_size=state_vector_size,
                    memory_impact=memory_impact
                )
                
                # Calculate operations per second
                if duration > 0 and state_vector_size > 0:
                    gate_profile.operations_per_second = state_vector_size / duration
                
                profile.gate_profiles.append(gate_profile)
                
                # Update breakdown timing by gate type
                breakdown = profile.execution_breakdown
                if gate_profile.gate_type == "single":
                    breakdown.single_qubit_gates_time += duration
                elif gate_profile.gate_type == "two_qubit":
                    breakdown.two_qubit_gates_time += duration
                elif gate_profile.gate_type == "multi_qubit":
                    breakdown.multi_qubit_gates_time += duration
                elif gate_profile.gate_type == "controlled":
                    breakdown.controlled_gates_time += duration
        
        self._gate_start_time = None
        self._current_gate_info = None
    
    def record_memory_usage(self, memory_mb: float):
        """Record peak memory usage."""
        if not self._current_profile_id:
            return
        
        with self._lock:
            if self._current_profile_id in self._active_profiles:
                profile = self._active_profiles[self._current_profile_id]
                profile.peak_memory_mb = max(profile.peak_memory_mb, memory_mb)
    
    def record_performance_warning(self, warning: str):
        """Record a performance warning."""
        if not self._current_profile_id:
            return
        
        with self._lock:
            if self._current_profile_id in self._active_profiles:
                profile = self._active_profiles[self._current_profile_id]
                profile.performance_warnings.append(warning)
                self.logger.warning(f"Simulation performance warning: {warning}")
    
    def get_current_profile(self) -> Optional[SimulationProfile]:
        """Get the currently active profile."""
        if not self._current_profile_id:
            return None
        
        with self._lock:
            return self._active_profiles.get(self._current_profile_id)
    
    def get_profile_history(self, limit: Optional[int] = None) -> List[SimulationProfile]:
        """Get historical simulation profiles."""
        with self._lock:
            history = sorted(self._profile_history, key=lambda p: p.start_time, reverse=True)
            return history[:limit] if limit else history 