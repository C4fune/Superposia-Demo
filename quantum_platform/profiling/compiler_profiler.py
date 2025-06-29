"""
Compiler Performance Profiler

This module provides profiling for quantum circuit compilation,
tracking optimization pass timing and compilation efficiency.
"""

import time
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from quantum_platform.observability.logging import get_logger


class PassType(Enum):
    """Types of compilation passes."""
    OPTIMIZATION = "optimization"
    MAPPING = "mapping"
    LAYOUT = "layout"
    ROUTING = "routing"
    SYNTHESIS = "synthesis"
    TRANSPILATION = "transpilation"
    VERIFICATION = "verification"


@dataclass
class PassProfile:
    """Profile data for a single compilation pass."""
    pass_name: str
    pass_type: PassType
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Pass effectiveness metrics
    gates_before: int = 0
    gates_after: int = 0
    depth_before: int = 0
    depth_after: int = 0
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Pass-specific metrics
    pass_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def gate_reduction_ratio(self) -> float:
        """Calculate gate count reduction ratio."""
        if self.gates_before == 0:
            return 0.0
        return (self.gates_before - self.gates_after) / self.gates_before
    
    @property
    def depth_reduction_ratio(self) -> float:
        """Calculate circuit depth reduction ratio."""
        if self.depth_before == 0:
            return 0.0
        return (self.depth_before - self.depth_after) / self.depth_before
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'pass_name': self.pass_name,
            'pass_type': self.pass_type.value,
            'execution_time': self.execution_time,
            'gates_before': self.gates_before,
            'gates_after': self.gates_after,
            'depth_before': self.depth_before,
            'depth_after': self.depth_after,
            'gate_reduction_ratio': self.gate_reduction_ratio,
            'depth_reduction_ratio': self.depth_reduction_ratio,
            'memory_usage_mb': self.memory_usage_mb,
            'pass_metrics': self.pass_metrics
        }


@dataclass
class OptimizationTiming:
    """Timing breakdown for optimization phases."""
    
    # Overall timing
    total_compilation_time: float = 0.0
    
    # Phase breakdown
    parsing_time: float = 0.0
    optimization_time: float = 0.0
    mapping_time: float = 0.0
    synthesis_time: float = 0.0
    verification_time: float = 0.0
    
    # Pass details
    pass_profiles: List[PassProfile] = field(default_factory=list)
    
    # Efficiency metrics
    compilation_efficiency: float = 0.0  # gates_per_second
    optimization_effectiveness: float = 0.0  # total reduction achieved
    
    def calculate_efficiency_metrics(self, final_gates: int, initial_gates: int):
        """Calculate compilation efficiency metrics."""
        if self.total_compilation_time > 0:
            self.compilation_efficiency = final_gates / self.total_compilation_time
        
        if initial_gates > 0:
            self.optimization_effectiveness = (initial_gates - final_gates) / initial_gates
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'total_compilation_time': self.total_compilation_time,
            'parsing_time': self.parsing_time,
            'optimization_time': self.optimization_time,
            'mapping_time': self.mapping_time,
            'synthesis_time': self.synthesis_time,
            'verification_time': self.verification_time,
            'compilation_efficiency': self.compilation_efficiency,
            'optimization_effectiveness': self.optimization_effectiveness,
            'pass_count': len(self.pass_profiles)
        }


@dataclass
class CompilerProfile:
    """Complete compiler performance profile."""
    profile_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_time: float = 0.0
    
    # Circuit information
    circuit_name: str = ""
    initial_gates: int = 0
    final_gates: int = 0
    initial_depth: int = 0
    final_depth: int = 0
    target_backend: str = ""
    
    # Optimization details
    optimization_timing: OptimizationTiming = field(default_factory=OptimizationTiming)
    
    # Resource usage
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    
    # Compilation results
    optimization_level: int = 0
    compilation_successful: bool = True
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def calculate_summary_metrics(self):
        """Calculate summary compilation metrics."""
        self.optimization_timing.calculate_efficiency_metrics(self.final_gates, self.initial_gates)
        self.optimization_timing.total_compilation_time = self.total_time


class CompilerProfiler:
    """
    Profiler for quantum circuit compilation performance.
    
    Tracks timing and effectiveness of compilation passes,
    optimization levels, and resource usage during compilation.
    """
    
    def __init__(self, config):
        """Initialize the compiler profiler."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Active profiles
        self._active_profiles: Dict[str, CompilerProfile] = {}
        self._profile_history: List[CompilerProfile] = []
        self._lock = threading.RLock()
        
        # Current profiling context
        self._current_profile_id: Optional[str] = None
        self._phase_start_times: Dict[str, float] = {}
        self._pass_start_time: Optional[float] = None
        self._current_pass_info: Optional[Dict[str, Any]] = None
        
        self.logger.debug("CompilerProfiler initialized")
    
    def start_profile(self, profile_id: str) -> CompilerProfile:
        """Start profiling a compilation."""
        with self._lock:
            if profile_id in self._active_profiles:
                self.logger.warning(f"Compiler profile {profile_id} already active")
                return self._active_profiles[profile_id]
            
            profile = CompilerProfile(
                profile_id=profile_id,
                start_time=datetime.now()
            )
            
            self._active_profiles[profile_id] = profile
            self._current_profile_id = profile_id
            
            self.logger.debug(f"Started compiler profiling: {profile_id}")
            return profile
    
    def stop_profile(self, profile_id: str) -> Optional[CompilerProfile]:
        """Stop profiling and return results."""
        with self._lock:
            if profile_id not in self._active_profiles:
                self.logger.warning(f"Compiler profile {profile_id} not found")
                return None
            
            profile = self._active_profiles[profile_id]
            profile.end_time = datetime.now()
            profile.total_time = (profile.end_time - profile.start_time).total_seconds()
            
            # Calculate summary metrics
            profile.calculate_summary_metrics()
            
            # Move to history
            del self._active_profiles[profile_id]
            self._profile_history.append(profile)
            
            if self._current_profile_id == profile_id:
                self._current_profile_id = None
            
            self.logger.debug(f"Completed compiler profiling: {profile_id}")
            return profile
    
    def set_circuit_info(self, circuit_name: str, initial_gates: int, initial_depth: int, 
                        target_backend: str = "", optimization_level: int = 0):
        """Set circuit information for the current profile."""
        if not self._current_profile_id:
            return
        
        with self._lock:
            if self._current_profile_id in self._active_profiles:
                profile = self._active_profiles[self._current_profile_id]
                profile.circuit_name = circuit_name
                profile.initial_gates = initial_gates
                profile.initial_depth = initial_depth
                profile.target_backend = target_backend
                profile.optimization_level = optimization_level
    
    def start_pass_timing(self, pass_name: str, pass_type: PassType, 
                         gates_before: int = 0, depth_before: int = 0):
        """Start timing a compilation pass."""
        if not self._current_profile_id or not self.config.track_gate_timing:
            return
        
        self._pass_start_time = time.time()
        self._current_pass_info = {
            'pass_name': pass_name,
            'pass_type': pass_type,
            'gates_before': gates_before,
            'depth_before': depth_before,
            'start_time': datetime.now()
        }
    
    def end_pass_timing(self, gates_after: int = 0, depth_after: int = 0, **pass_metrics):
        """End timing a compilation pass."""
        if (not self._current_profile_id or not self.config.track_gate_timing or 
            not self._pass_start_time or not self._current_pass_info):
            return
        
        duration = time.time() - self._pass_start_time
        
        with self._lock:
            if self._current_profile_id in self._active_profiles:
                profile = self._active_profiles[self._current_profile_id]
                
                pass_profile = PassProfile(
                    pass_name=self._current_pass_info['pass_name'],
                    pass_type=self._current_pass_info['pass_type'],
                    execution_time=duration,
                    start_time=self._current_pass_info['start_time'],
                    end_time=datetime.now(),
                    gates_before=self._current_pass_info['gates_before'],
                    gates_after=gates_after,
                    depth_before=self._current_pass_info['depth_before'],
                    depth_after=depth_after,
                    pass_metrics=pass_metrics
                )
                
                profile.optimization_timing.pass_profiles.append(pass_profile)
                
                # Update phase timing based on pass type
                timing = profile.optimization_timing
                if pass_profile.pass_type == PassType.OPTIMIZATION:
                    timing.optimization_time += duration
                elif pass_profile.pass_type == PassType.MAPPING:
                    timing.mapping_time += duration
                elif pass_profile.pass_type == PassType.SYNTHESIS:
                    timing.synthesis_time += duration
                elif pass_profile.pass_type == PassType.VERIFICATION:
                    timing.verification_time += duration
        
        self._pass_start_time = None
        self._current_pass_info = None 