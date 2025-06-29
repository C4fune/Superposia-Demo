"""
Progress Tracking for Quantum Executions

This module provides comprehensive progress tracking capabilities for quantum
simulations and other long-running operations, with real-time updates and
estimated completion times.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any, List
from enum import Enum
from quantum_platform.observability.logging import get_logger

class ProgressType(Enum):
    """Types of progress tracking."""
    PERCENTAGE = "percentage"
    STEPS = "steps"
    SHOTS = "shots"
    TIME_BASED = "time_based"
    CUSTOM = "custom"

@dataclass
class SimulationProgress:
    """Progress information for quantum simulations."""
    current_step: int = 0
    total_steps: int = 0
    current_shot: int = 0
    total_shots: int = 0
    percentage: float = 0.0
    estimated_remaining: Optional[timedelta] = None
    message: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def elapsed_time(self) -> timedelta:
        """Get elapsed time since start."""
        return datetime.now() - self.started_at
    
    @property
    def estimated_total_time(self) -> Optional[timedelta]:
        """Get estimated total execution time."""
        if self.percentage > 0:
            return self.elapsed_time * (100.0 / self.percentage)
        return None
    
    @property
    def estimated_completion_time(self) -> Optional[datetime]:
        """Get estimated completion time."""
        if self.estimated_total_time:
            return self.started_at + self.estimated_total_time
        return None
    
    def update_percentage(self, percentage: float, message: str = ""):
        """Update progress by percentage."""
        self.percentage = max(0.0, min(100.0, percentage))
        self.message = message
        self.last_update = datetime.now()
        
        # Update estimated remaining time
        if self.percentage > 0 and self.percentage < 100:
            elapsed = self.elapsed_time
            total_estimated = elapsed * (100.0 / self.percentage)
            self.estimated_remaining = total_estimated - elapsed
    
    def update_steps(self, current_step: int, total_steps: int, message: str = ""):
        """Update progress by steps."""
        self.current_step = current_step
        self.total_steps = total_steps
        if total_steps > 0:
            self.percentage = (current_step / total_steps) * 100.0
        self.update_percentage(self.percentage, message)
    
    def update_shots(self, current_shot: int, total_shots: int, message: str = ""):
        """Update progress by shots."""
        self.current_shot = current_shot
        self.total_shots = total_shots
        if total_shots > 0:
            self.percentage = (current_shot / total_shots) * 100.0
        self.update_percentage(self.percentage, message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'current_shot': self.current_shot,
            'total_shots': self.total_shots,
            'percentage': self.percentage,
            'message': self.message,
            'elapsed_seconds': self.elapsed_time.total_seconds(),
            'estimated_remaining_seconds': (
                self.estimated_remaining.total_seconds() 
                if self.estimated_remaining else None
            ),
            'estimated_completion': (
                self.estimated_completion_time.isoformat()
                if self.estimated_completion_time else None
            ),
            'last_update': self.last_update.isoformat()
        }

# Type alias for progress callback functions
ProgressCallback = Callable[[SimulationProgress], None]

class ProgressTracker:
    """
    Advanced progress tracker for quantum operations.
    
    Provides real-time progress tracking with multiple update mechanisms,
    time estimation, and callback support for UI updates.
    """
    
    def __init__(self, 
                 operation_name: str,
                 progress_type: ProgressType = ProgressType.PERCENTAGE,
                 update_interval: float = 0.1):
        """
        Initialize progress tracker.
        
        Args:
            operation_name: Name of the operation being tracked
            progress_type: Type of progress tracking
            update_interval: Minimum interval between updates (seconds)
        """
        self.operation_name = operation_name
        self.progress_type = progress_type
        self.update_interval = update_interval
        
        # Progress state
        self.progress = SimulationProgress()
        self.is_active = False
        self.is_paused = False
        
        # Callbacks
        self.callbacks: List[ProgressCallback] = []
        
        # Threading
        self._lock = threading.RLock()
        self._last_callback_time = 0.0
        
        # Logging
        self.logger = get_logger("ProgressTracker")
    
    def start(self, total_steps: Optional[int] = None, total_shots: Optional[int] = None):
        """
        Start progress tracking.
        
        Args:
            total_steps: Total number of steps (for step-based tracking)
            total_shots: Total number of shots (for shot-based tracking)
        """
        with self._lock:
            self.progress = SimulationProgress()
            if total_steps:
                self.progress.total_steps = total_steps
            if total_shots:
                self.progress.total_shots = total_shots
            
            self.is_active = True
            self.is_paused = False
            
            self.logger.info(f"Started progress tracking for {self.operation_name}")
            self._notify_callbacks()
    
    def update(self, 
               current: Optional[int] = None,
               total: Optional[int] = None,
               percentage: Optional[float] = None,
               message: str = "",
               force_update: bool = False):
        """
        Update progress.
        
        Args:
            current: Current step/shot number
            total: Total steps/shots
            percentage: Direct percentage update
            message: Progress message
            force_update: Force callback notification even if within update interval
        """
        if not self.is_active or self.is_paused:
            return
        
        with self._lock:
            if percentage is not None:
                self.progress.update_percentage(percentage, message)
            elif self.progress_type == ProgressType.STEPS and current is not None:
                total = total or self.progress.total_steps
                self.progress.update_steps(current, total, message)
            elif self.progress_type == ProgressType.SHOTS and current is not None:
                total = total or self.progress.total_shots
                self.progress.update_shots(current, total, message)
            
            # Notify callbacks if enough time has passed or forced
            current_time = time.time()
            if (force_update or 
                current_time - self._last_callback_time >= self.update_interval):
                self._notify_callbacks()
                self._last_callback_time = current_time
    
    def update_percentage(self, percentage: float, message: str = ""):
        """Update progress by percentage."""
        self.update(percentage=percentage, message=message)
    
    def update_steps(self, current_step: int, total_steps: Optional[int] = None, message: str = ""):
        """Update progress by steps."""
        self.update(current=current_step, total=total_steps, message=message)
    
    def update_shots(self, current_shot: int, total_shots: Optional[int] = None, message: str = ""):
        """Update progress by shots."""
        self.update(current=current_shot, total=total_shots, message=message)
    
    def complete(self, message: str = "Completed"):
        """Mark operation as completed."""
        with self._lock:
            if self.is_active:
                self.progress.update_percentage(100.0, message)
                self.is_active = False
                self.logger.info(f"Completed progress tracking for {self.operation_name}")
                self._notify_callbacks(force=True)
    
    def pause(self):
        """Pause progress tracking."""
        with self._lock:
            self.is_paused = True
            self.logger.debug(f"Paused progress tracking for {self.operation_name}")
    
    def resume(self):
        """Resume progress tracking."""
        with self._lock:
            self.is_paused = False
            self.logger.debug(f"Resumed progress tracking for {self.operation_name}")
    
    def stop(self, message: str = "Stopped"):
        """Stop progress tracking."""
        with self._lock:
            self.is_active = False
            self.is_paused = False
            self.progress.message = message
            self.logger.info(f"Stopped progress tracking for {self.operation_name}")
            self._notify_callbacks(force=True)
    
    def add_callback(self, callback: ProgressCallback):
        """Add progress update callback."""
        with self._lock:
            if callback not in self.callbacks:
                self.callbacks.append(callback)
    
    def remove_callback(self, callback: ProgressCallback):
        """Remove progress update callback."""
        with self._lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)
    
    def _notify_callbacks(self, force: bool = False):
        """Notify all registered callbacks of progress update."""
        if not force and (not self.is_active or self.is_paused):
            return
        
        for callback in self.callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information."""
        with self._lock:
            return {
                'operation_name': self.operation_name,
                'progress_type': self.progress_type.value,
                'is_active': self.is_active,
                'is_paused': self.is_paused,
                **self.progress.to_dict()
            }

class SimulationProgressTracker(ProgressTracker):
    """Specialized progress tracker for quantum simulations."""
    
    def __init__(self, operation_name: str, total_shots: int):
        """
        Initialize simulation progress tracker.
        
        Args:
            operation_name: Name of the simulation
            total_shots: Total number of shots to execute
        """
        super().__init__(operation_name, ProgressType.SHOTS)
        self.total_shots = total_shots
    
    def start_simulation(self):
        """Start simulation progress tracking."""
        self.start(total_shots=self.total_shots)
    
    def update_shot_progress(self, completed_shots: int, message: str = ""):
        """Update simulation shot progress."""
        self.update_shots(completed_shots, message=message)
    
    def batch_update(self, batch_size: int, completed_batches: int, total_batches: int):
        """
        Update progress for batch-based simulation.
        
        Args:
            batch_size: Number of shots per batch
            completed_batches: Number of completed batches
            total_batches: Total number of batches
        """
        completed_shots = completed_batches * batch_size
        # Ensure we don't exceed total shots
        completed_shots = min(completed_shots, self.total_shots)
        
        percentage = (completed_batches / total_batches) * 100.0
        message = f"Batch {completed_batches}/{total_batches} (shots: {completed_shots}/{self.total_shots})"
        
        self.update_percentage(percentage, message)

class MultiStageProgressTracker(ProgressTracker):
    """Progress tracker for multi-stage operations."""
    
    def __init__(self, operation_name: str, stages: List[str]):
        """
        Initialize multi-stage progress tracker.
        
        Args:
            operation_name: Name of the operation
            stages: List of stage names
        """
        super().__init__(operation_name, ProgressType.STEPS)
        self.stages = stages
        self.current_stage_index = 0
        self.stage_progress = 0.0
    
    def start_multi_stage(self):
        """Start multi-stage progress tracking."""
        self.start(total_steps=len(self.stages))
        self.current_stage_index = 0
        self.stage_progress = 0.0
    
    def update_stage_progress(self, stage_percentage: float):
        """
        Update progress within current stage.
        
        Args:
            stage_percentage: Percentage complete of current stage (0-100)
        """
        if self.current_stage_index >= len(self.stages):
            return
        
        self.stage_progress = max(0.0, min(100.0, stage_percentage))
        
        # Calculate overall percentage
        stages_completed = self.current_stage_index
        current_stage_contribution = self.stage_progress / 100.0
        total_percentage = ((stages_completed + current_stage_contribution) / len(self.stages)) * 100.0
        
        stage_name = self.stages[self.current_stage_index]
        message = f"Stage {self.current_stage_index + 1}/{len(self.stages)}: {stage_name} ({self.stage_progress:.1f}%)"
        
        self.update_percentage(total_percentage, message)
    
    def complete_stage(self):
        """Mark current stage as completed and move to next."""
        if self.current_stage_index < len(self.stages):
            self.current_stage_index += 1
            self.stage_progress = 0.0
            
            if self.current_stage_index >= len(self.stages):
                self.complete("All stages completed")
            else:
                self.update_stage_progress(0.0)
    
    def get_current_stage(self) -> Optional[str]:
        """Get name of current stage."""
        if 0 <= self.current_stage_index < len(self.stages):
            return self.stages[self.current_stage_index]
        return None

# Utility functions for common progress tracking patterns

def create_simulation_tracker(operation_name: str, total_shots: int) -> SimulationProgressTracker:
    """Create a simulation progress tracker."""
    return SimulationProgressTracker(operation_name, total_shots)

def create_multi_stage_tracker(operation_name: str, stages: List[str]) -> MultiStageProgressTracker:
    """Create a multi-stage progress tracker."""
    return MultiStageProgressTracker(operation_name, stages)

def track_progress(tracker: ProgressTracker, 
                  iterable, 
                  total: Optional[int] = None,
                  message_template: str = "Processing item {current}/{total}"):
    """
    Generator that tracks progress over an iterable.
    
    Args:
        tracker: Progress tracker instance
        iterable: Iterable to track progress over
        total: Total number of items (if not available from iterable)
        message_template: Template for progress messages
        
    Yields:
        Items from the iterable
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            # Iterable doesn't support len()
            total = 0
    
    tracker.start(total_steps=total)
    
    try:
        for i, item in enumerate(iterable):
            yield item
            
            if total > 0:
                message = message_template.format(current=i+1, total=total)
                tracker.update_steps(i + 1, total, message)
            else:
                tracker.update_percentage((i + 1) * 10.0)  # Arbitrary progress
    
    finally:
        tracker.complete() 