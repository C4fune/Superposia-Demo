"""
Quantum Hardware Job Manager

This module provides job management capabilities for tracking and monitoring
quantum hardware jobs across different providers.
"""

import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import queue
import uuid

from ..compiler.ir.circuit import QuantumCircuit
from ..errors import HardwareError, handle_errors
from .hal import (
    QuantumHardwareBackend, JobHandle, JobStatus, HardwareResult
)


class JobPriority(Enum):
    """Priority levels for quantum jobs."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class HardwareJob:
    """Represents a quantum hardware job."""
    job_id: str
    circuit: QuantumCircuit
    backend_name: str
    shots: int
    priority: JobPriority = JobPriority.NORMAL
    
    # Job metadata
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timing information
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Job state
    status: JobStatus = JobStatus.PENDING
    job_handle: Optional[JobHandle] = None
    result: Optional[HardwareResult] = None
    error_message: Optional[str] = None
    
    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    retry_delay: float = 60.0  # seconds
    
    def __post_init__(self):
        if not self.job_id:
            self.job_id = f"hw_job_{uuid.uuid4().hex[:8]}"
    
    def get_age(self) -> timedelta:
        """Get the age of the job."""
        return datetime.now() - self.created_at
    
    def get_execution_time(self) -> Optional[timedelta]:
        """Get the total execution time."""
        if self.submitted_at and self.completed_at:
            return self.completed_at - self.submitted_at
        return None
    
    def can_retry(self) -> bool:
        """Check if the job can be retried."""
        return (self.status == JobStatus.FAILED and 
                self.retry_count < self.max_retries)


class JobQueue:
    """Priority queue for hardware jobs."""
    
    def __init__(self):
        self._queue = queue.PriorityQueue()
        self._jobs: Dict[str, HardwareJob] = {}
        self._lock = threading.Lock()
    
    def enqueue(self, job: HardwareJob):
        """Add a job to the queue."""
        with self._lock:
            # Priority queue uses negative priority for max-heap behavior
            priority_value = -job.priority.value
            self._queue.put((priority_value, job.created_at, job))
            self._jobs[job.job_id] = job
    
    def dequeue(self, timeout: Optional[float] = None) -> Optional[HardwareJob]:
        """Get the next job from the queue."""
        try:
            _, _, job = self._queue.get(timeout=timeout)
            return job
        except queue.Empty:
            return None
    
    def get_job(self, job_id: str) -> Optional[HardwareJob]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a job from tracking."""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> List[HardwareJob]:
        """List jobs, optionally filtered by status."""
        with self._lock:
            jobs = list(self._jobs.values())
            if status:
                jobs = [job for job in jobs if job.status == status]
            return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    def size(self) -> int:
        """Get the number of jobs in queue."""
        return self._queue.qsize()
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()


class JobManager:
    """Manager for quantum hardware jobs."""
    
    def __init__(self, max_concurrent_jobs: int = 10):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue = JobQueue()
        self.backends: Dict[str, QuantumHardwareBackend] = {}
        
        # Monitoring
        self._active_jobs: Dict[str, HardwareJob] = {}
        self._completed_jobs: Dict[str, HardwareJob] = {}
        self._running = False
        self._worker_threads: List[threading.Thread] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Callbacks
        self._job_callbacks: Dict[str, List[Callable]] = {
            'on_submit': [],
            'on_start': [],
            'on_complete': [],
            'on_error': [],
            'on_cancel': []
        }
    
    def register_backend(self, name: str, backend: QuantumHardwareBackend):
        """Register a hardware backend."""
        self.backends[name] = backend
    
    def start(self):
        """Start the job manager."""
        if self._running:
            return
        
        self._running = True
        
        # Start worker threads
        for i in range(self.max_concurrent_jobs):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"JobWorker-{i}",
                daemon=True
            )
            worker.start()
            self._worker_threads.append(worker)
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="JobMonitor",
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop(self):
        """Stop the job manager."""
        self._running = False
        
        # Wait for threads to finish
        for worker in self._worker_threads:
            if worker.is_alive():
                worker.join(timeout=5.0)
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
    
    @handle_errors
    def submit_job(self, circuit: QuantumCircuit, backend_name: str, 
                   shots: int = 1000, priority: JobPriority = JobPriority.NORMAL,
                   **kwargs) -> str:
        """Submit a job for execution."""
        if backend_name not in self.backends:
            raise HardwareError(
                f"Backend '{backend_name}' not registered",
                user_message=f"Unknown hardware backend: {backend_name}"
            )
        
        # Create job
        job = HardwareJob(
            job_id=f"job_{int(time.time() * 1000)}_{id(circuit)}",
            circuit=circuit,
            backend_name=backend_name,
            shots=shots,
            priority=priority,
            **kwargs
        )
        
        # Add to queue
        self.job_queue.enqueue(job)
        
        # Trigger callbacks
        self._trigger_callbacks('on_submit', job)
        
        return job.job_id
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get the status of a job."""
        job = self.job_queue.get_job(job_id)
        if job:
            return job.status
        
        # Check completed jobs
        if job_id in self._completed_jobs:
            return self._completed_jobs[job_id].status
        
        return None
    
    def get_job_result(self, job_id: str) -> Optional[HardwareResult]:
        """Get the result of a completed job."""
        job = self.job_queue.get_job(job_id)
        if job and job.result:
            return job.result
        
        # Check completed jobs
        if job_id in self._completed_jobs:
            return self._completed_jobs[job_id].result
        
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.job_queue.get_job(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.PENDING, JobStatus.QUEUED]:
            # Job hasn't started yet
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            self._trigger_callbacks('on_cancel', job)
            return True
        
        elif job.status == JobStatus.RUNNING and job.job_handle:
            # Try to cancel running job
            backend = self.backends.get(job.backend_name)
            if backend and backend.cancel_job(job.job_handle):
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                self._trigger_callbacks('on_cancel', job)
                return True
        
        return False
    
    def list_jobs(self, status: Optional[JobStatus] = None, 
                  user_id: Optional[str] = None) -> List[HardwareJob]:
        """List jobs with optional filtering."""
        jobs = self.job_queue.list_jobs(status)
        
        # Add completed jobs
        completed = list(self._completed_jobs.values())
        if status:
            completed = [job for job in completed if job.status == status]
        jobs.extend(completed)
        
        # Filter by user
        if user_id:
            jobs = [job for job in jobs if job.user_id == user_id]
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            active_count = len(self._active_jobs)
            pending_count = self.job_queue.size()
            completed_count = len(self._completed_jobs)
            
            # Count by status
            status_counts = {}
            for job in self.job_queue.list_jobs():
                status = job.status
                status_counts[status.value] = status_counts.get(status.value, 0) + 1
        
        return {
            "active_jobs": active_count,
            "pending_jobs": pending_count,
            "completed_jobs": completed_count,
            "total_jobs": active_count + pending_count + completed_count,
            "status_breakdown": status_counts,
            "max_concurrent": self.max_concurrent_jobs,
            "available_backends": list(self.backends.keys())
        }
    
    def add_callback(self, event: str, callback: Callable):
        """Add a callback for job events."""
        if event in self._job_callbacks:
            self._job_callbacks[event].append(callback)
    
    def _worker_loop(self):
        """Main worker loop for processing jobs."""
        while self._running:
            try:
                # Get next job
                job = self.job_queue.dequeue(timeout=1.0)
                if not job:
                    continue
                
                # Skip cancelled jobs
                if job.status == JobStatus.CANCELLED:
                    continue
                
                # Execute job
                self._execute_job(job)
                
            except Exception as e:
                # Log error but keep worker running
                print(f"Worker error: {e}")
    
    def _execute_job(self, job: HardwareJob):
        """Execute a single job."""
        try:
            # Get backend
            backend = self.backends.get(job.backend_name)
            if not backend:
                raise HardwareError(f"Backend {job.backend_name} not available")
            
            # Track active job
            with self._lock:
                self._active_jobs[job.job_id] = job
            
            # Update job status
            job.status = JobStatus.RUNNING
            job.submitted_at = datetime.now()
            self._trigger_callbacks('on_start', job)
            
            # Submit to backend
            job.job_handle = backend.submit_circuit(job.circuit, job.shots)
            
            # Wait for completion (in monitoring thread)
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            with self._lock:
                if job.job_id in self._active_jobs:
                    del self._active_jobs[job.job_id]
                self._completed_jobs[job.job_id] = job
            
            self._trigger_callbacks('on_error', job)
    
    def _monitor_loop(self):
        """Monitor active jobs for completion."""
        while self._running:
            try:
                jobs_to_check = []
                with self._lock:
                    jobs_to_check = list(self._active_jobs.values())
                
                for job in jobs_to_check:
                    if job.job_handle and job.status == JobStatus.RUNNING:
                        backend = self.backends.get(job.backend_name)
                        if backend:
                            status = backend.get_job_status(job.job_handle)
                            
                            if status in [JobStatus.COMPLETED, JobStatus.FAILED, 
                                         JobStatus.CANCELLED]:
                                self._finalize_job(job, backend)
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(10.0)
    
    def _finalize_job(self, job: HardwareJob, backend: QuantumHardwareBackend):
        """Finalize a completed job."""
        try:
            # Get result
            result = backend.retrieve_results(job.job_handle)
            job.result = result
            job.status = result.status
            job.completed_at = datetime.now()
            
            # Move to completed jobs
            with self._lock:
                if job.job_id in self._active_jobs:
                    del self._active_jobs[job.job_id]
                self._completed_jobs[job.job_id] = job
            
            if job.status == JobStatus.COMPLETED:
                self._trigger_callbacks('on_complete', job)
            else:
                self._trigger_callbacks('on_error', job)
                
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            self._trigger_callbacks('on_error', job)
    
    def _trigger_callbacks(self, event: str, job: HardwareJob):
        """Trigger callbacks for job events."""
        for callback in self._job_callbacks.get(event, []):
            try:
                callback(job)
            except Exception as e:
                print(f"Callback error for {event}: {e}")


# Global job manager instance
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager 