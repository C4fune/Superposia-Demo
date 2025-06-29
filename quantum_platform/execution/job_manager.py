"""
Job Manager for Quantum Execution Monitoring

This module provides centralized management and tracking of quantum execution jobs,
including simulations and hardware executions, with real-time status updates
and progress monitoring capabilities.
"""

import threading
import uuid
import time
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from collections import defaultdict
from quantum_platform.observability.logging import get_logger
from quantum_platform.observability.monitor import get_monitor

class JobStatus(Enum):
    """Enumeration of possible job execution statuses."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    QUEUED = "queued"
    PAUSED = "paused"

class JobType(Enum):
    """Enumeration of job types."""
    SIMULATION = "simulation"
    HARDWARE = "hardware"
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"

@dataclass
class ExecutionJob:
    """
    Represents a quantum execution job with status tracking and metadata.
    """
    job_id: str
    job_type: JobType
    name: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0  # Progress as percentage (0.0 to 100.0)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    estimated_completion: Optional[datetime] = None
    
    # Job details
    circuit_name: Optional[str] = None
    backend_name: Optional[str] = None
    shots: Optional[int] = None
    queue_position: Optional[int] = None
    
    # Results and errors
    result: Optional[Any] = None
    error_message: Optional[str] = None
    
    # Control
    cancellation_token: Optional[threading.Event] = field(default_factory=threading.Event)
    progress_callback: Optional[Callable[[float, str], None]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize job after creation."""
        if not self.job_id:
            self.job_id = str(uuid.uuid4())
        if not self.cancellation_token:
            self.cancellation_token = threading.Event()
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get job duration if started."""
        if self.started_at:
            end_time = self.completed_at or datetime.now()
            return end_time - self.started_at
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if job is currently active (running, queued, or initializing)."""
        return self.status in [JobStatus.RUNNING, JobStatus.QUEUED, JobStatus.INITIALIZING]
    
    @property
    def is_finished(self) -> bool:
        """Check if job has finished (completed, failed, or cancelled)."""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    def update_progress(self, progress: float, message: str = ""):
        """
        Update job progress.
        
        Args:
            progress: Progress percentage (0.0 to 100.0)
            message: Optional progress message
        """
        self.progress = max(0.0, min(100.0, progress))
        
        if self.progress_callback:
            self.progress_callback(self.progress, message)
        
        # Update estimated completion
        if self.started_at and self.progress > 0:
            elapsed = datetime.now() - self.started_at
            if self.progress > 0:
                total_estimated = elapsed * (100.0 / self.progress)
                self.estimated_completion = self.started_at + total_estimated
    
    def start(self):
        """Mark job as started."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now()
    
    def complete(self, result: Any = None):
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now()
        self.progress = 100.0
        self.result = result
    
    def fail(self, error_message: str):
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
    
    def cancel(self):
        """Cancel the job."""
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.now()
        if self.cancellation_token:
            self.cancellation_token.set()
    
    def is_cancelled(self) -> bool:
        """Check if job has been cancelled."""
        return (self.cancellation_token and self.cancellation_token.is_set()) or \
               self.status == JobStatus.CANCELLED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary representation."""
        return {
            'job_id': self.job_id,
            'job_type': self.job_type.value,
            'name': self.name,
            'status': self.status.value,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration.total_seconds() if self.duration else None,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'circuit_name': self.circuit_name,
            'backend_name': self.backend_name,
            'shots': self.shots,
            'queue_position': self.queue_position,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'tags': self.tags
        }

class JobManager:
    """
    Central manager for quantum execution jobs.
    
    Provides job tracking, status management, and coordination between
    different execution backends (simulation, hardware, etc.).
    """
    
    def __init__(self, max_concurrent_jobs: int = 10, cleanup_interval: int = 3600):
        """
        Initialize job manager.
        
        Args:
            max_concurrent_jobs: Maximum number of concurrent jobs
            cleanup_interval: Interval in seconds to clean up old completed jobs
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.cleanup_interval = cleanup_interval
        
        # Job storage
        self.jobs: Dict[str, ExecutionJob] = {}
        self.job_queue: List[str] = []  # Queue of pending job IDs
        self.active_jobs: Dict[str, threading.Thread] = {}
        
        # Statistics
        self.job_stats = defaultdict(int)
        
        # Threading
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        # Logging and monitoring
        self.logger = get_logger("JobManager")
        self.monitor = get_monitor()
        
        # Event callbacks
        self.status_change_callbacks: List[Callable[[ExecutionJob], None]] = []
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        self.logger.info("Job manager initialized")
    
    def _start_cleanup_thread(self):
        """Start background thread for job cleanup."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="JobCleanup"
        )
        self._cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup of old completed jobs."""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            try:
                self._cleanup_old_jobs()
            except Exception as e:
                self.logger.error(f"Error in job cleanup: {e}")
    
    def _cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove old completed jobs."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._lock:
            jobs_to_remove = []
            for job_id, job in self.jobs.items():
                if (job.is_finished and 
                    job.completed_at and 
                    job.completed_at < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                self.logger.debug(f"Cleaned up old job: {job_id}")
    
    def create_job(self, 
                   job_type: JobType,
                   name: str,
                   circuit_name: Optional[str] = None,
                   backend_name: Optional[str] = None,
                   shots: Optional[int] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   tags: Optional[List[str]] = None) -> ExecutionJob:
        """
        Create a new execution job.
        
        Args:
            job_type: Type of job
            name: Human-readable job name
            circuit_name: Name of quantum circuit
            backend_name: Name of execution backend
            shots: Number of shots for simulation
            metadata: Additional metadata
            tags: Job tags for organization
            
        Returns:
            Created ExecutionJob
        """
        job = ExecutionJob(
            job_id=str(uuid.uuid4()),
            job_type=job_type,
            name=name,
            circuit_name=circuit_name,
            backend_name=backend_name,
            shots=shots,
            metadata=metadata or {},
            tags=tags or []
        )
        
        with self._lock:
            self.jobs[job.job_id] = job
            self.job_stats['total_created'] += 1
            self.job_stats[f'{job_type.value}_created'] += 1
        
        self.logger.info(f"Created {job_type.value} job: {name} ({job.job_id})")
        self._notify_status_change(job)
        
        return job
    
    def submit_job(self, job: ExecutionJob, 
                   executor_func: Callable[[ExecutionJob], Any],
                   auto_start: bool = True) -> bool:
        """
        Submit a job for execution.
        
        Args:
            job: Job to execute
            executor_func: Function to execute the job
            auto_start: Whether to start immediately if resources available
            
        Returns:
            True if job was submitted successfully
        """
        with self._lock:
            if job.job_id not in self.jobs:
                self.logger.error(f"Job {job.job_id} not found in manager")
                return False
            
            # Add to queue
            self.job_queue.append(job.job_id)
            job.status = JobStatus.QUEUED
            
            self.logger.info(f"Submitted job {job.name} to queue")
            self._notify_status_change(job)
            
            if auto_start:
                self._try_start_next_job()
            
            return True
    
    def _try_start_next_job(self):
        """Try to start the next queued job if resources are available."""
        with self._lock:
            if (len(self.active_jobs) >= self.max_concurrent_jobs or 
                not self.job_queue):
                return
            
            job_id = self.job_queue.pop(0)
            job = self.jobs.get(job_id)
            
            if not job or job.is_cancelled():
                # Job was cancelled while queued
                return
            
            # Start job in separate thread
            def job_wrapper():
                try:
                    job.start()
                    self._notify_status_change(job)
                    
                    # Execute job with monitoring
                    with self.monitor.measure_operation(job.name, "JobManager"):
                        result = self._execute_job(job)
                        job.complete(result)
                    
                    self.job_stats['total_completed'] += 1
                    self.job_stats[f'{job.job_type.value}_completed'] += 1
                    
                except Exception as e:
                    job.fail(str(e))
                    self.job_stats['total_failed'] += 1
                    self.job_stats[f'{job.job_type.value}_failed'] += 1
                    self.logger.error(f"Job {job.name} failed: {e}")
                
                finally:
                    # Remove from active jobs
                    with self._lock:
                        self.active_jobs.pop(job_id, None)
                    
                    self._notify_status_change(job)
                    
                    # Try to start next job
                    self._try_start_next_job()
            
            thread = threading.Thread(target=job_wrapper, name=f"Job-{job.name}")
            self.active_jobs[job_id] = thread
            thread.start()
            
            self.logger.info(f"Started job {job.name}")
    
    def _execute_job(self, job: ExecutionJob) -> Any:
        """
        Execute a job (placeholder - actual execution handled by specific executors).
        
        Args:
            job: Job to execute
            
        Returns:
            Job result
        """
        # This is a placeholder - actual job execution is handled by
        # specific executor functions passed to submit_job
        
        # Simulate progress for demonstration
        for i in range(101):
            if job.is_cancelled():
                raise RuntimeError("Job was cancelled")
            
            job.update_progress(float(i), f"Processing step {i}/100")
            time.sleep(0.01)  # Simulate work
        
        return {"status": "completed", "steps": 100}
    
    def get_job(self, job_id: str) -> Optional[ExecutionJob]:
        """Get job by ID."""
        with self._lock:
            return self.jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: ID of job to cancel
            
        Returns:
            True if job was cancelled successfully
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            
            if job.is_finished:
                self.logger.warning(f"Cannot cancel finished job {job_id}")
                return False
            
            job.cancel()
            
            # Remove from queue if queued
            if job_id in self.job_queue:
                self.job_queue.remove(job_id)
            
            # Cancel active thread if running
            if job_id in self.active_jobs:
                # Thread will check cancellation token and exit
                pass
            
            self.job_stats['total_cancelled'] += 1
            self.job_stats[f'{job.job_type.value}_cancelled'] += 1
            
            self.logger.info(f"Cancelled job {job.name}")
            self._notify_status_change(job)
            
            return True
    
    def get_active_jobs(self) -> List[ExecutionJob]:
        """Get list of active jobs."""
        with self._lock:
            return [job for job in self.jobs.values() if job.is_active]
    
    def get_jobs_by_status(self, status: JobStatus) -> List[ExecutionJob]:
        """Get jobs by status."""
        with self._lock:
            return [job for job in self.jobs.values() if job.status == status]
    
    def get_jobs_by_type(self, job_type: JobType) -> List[ExecutionJob]:
        """Get jobs by type."""
        with self._lock:
            return [job for job in self.jobs.values() if job.job_type == job_type]
    
    def get_recent_jobs(self, hours: int = 24) -> List[ExecutionJob]:
        """Get jobs created in the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        with self._lock:
            return [job for job in self.jobs.values() 
                   if job.created_at >= cutoff_time]
    
    def get_job_statistics(self) -> Dict[str, Any]:
        """Get job execution statistics."""
        with self._lock:
            stats = dict(self.job_stats)
            stats.update({
                'total_jobs': len(self.jobs),
                'active_jobs': len(self.active_jobs),
                'queued_jobs': len(self.job_queue),
                'max_concurrent': self.max_concurrent_jobs,
                'jobs_by_status': {
                    status.value: len(self.get_jobs_by_status(status))
                    for status in JobStatus
                },
                'jobs_by_type': {
                    job_type.value: len(self.get_jobs_by_type(job_type))
                    for job_type in JobType
                }
            })
            
            return stats
    
    def add_status_change_callback(self, callback: Callable[[ExecutionJob], None]):
        """Add callback for job status changes."""
        self.status_change_callbacks.append(callback)
    
    def remove_status_change_callback(self, callback: Callable[[ExecutionJob], None]):
        """Remove status change callback."""
        if callback in self.status_change_callbacks:
            self.status_change_callbacks.remove(callback)
    
    def _notify_status_change(self, job: ExecutionJob):
        """Notify all callbacks of job status change."""
        for callback in self.status_change_callbacks:
            try:
                callback(job)
            except Exception as e:
                self.logger.error(f"Error in status change callback: {e}")
    
    def shutdown(self):
        """Shutdown job manager and cleanup resources."""
        self.logger.info("Shutting down job manager")
        
        # Stop cleanup thread
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        
        # Cancel all active jobs
        with self._lock:
            for job_id in list(self.active_jobs.keys()):
                self.cancel_job(job_id)
        
        # Wait for active threads to finish
        for thread in self.active_jobs.values():
            thread.join(timeout=2.0)
        
        self.logger.info("Job manager shutdown complete")

# Global job manager instance
_global_job_manager: Optional[JobManager] = None
_job_manager_lock = threading.Lock()

def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _global_job_manager
    
    with _job_manager_lock:
        if _global_job_manager is None:
            _global_job_manager = JobManager()
    
    return _global_job_manager 