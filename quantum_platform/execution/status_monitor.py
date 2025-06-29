"""
Status Monitoring for Hardware Jobs and External Systems

This module provides background monitoring of hardware job status, queue positions,
and external quantum system states with automatic polling and status updates.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from collections import defaultdict
from quantum_platform.observability.logging import get_logger
from quantum_platform.observability.monitor import get_monitor

class HardwareStatus(Enum):
    """Status of hardware systems."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    BUSY = "busy"
    UNKNOWN = "unknown"

@dataclass
class StatusUpdate:
    """Represents a status update for a monitored item."""
    item_id: str
    item_type: str  # "job", "hardware", "queue", etc.
    status: str
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'item_id': self.item_id,
            'item_type': self.item_type,
            'status': self.status,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class HardwareJobInfo:
    """Information about a hardware job being monitored."""
    job_id: str
    provider_job_id: str
    provider_name: str
    device_name: str
    status: str = "unknown"
    queue_position: Optional[int] = None
    estimated_start_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    result_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'job_id': self.job_id,
            'provider_job_id': self.provider_job_id,
            'provider_name': self.provider_name,
            'device_name': self.device_name,
            'status': self.status,
            'queue_position': self.queue_position,
            'estimated_start_time': self.estimated_start_time.isoformat() if self.estimated_start_time else None,
            'estimated_completion_time': self.estimated_completion_time.isoformat() if self.estimated_completion_time else None,
            'last_updated': self.last_updated.isoformat(),
            'error_message': self.error_message,
            'result_url': self.result_url,
            'metadata': self.metadata
        }

class HardwareJobMonitor:
    """
    Monitor for individual hardware jobs.
    
    Tracks the status of jobs submitted to quantum hardware providers
    and provides real-time updates on queue position, execution status, etc.
    """
    
    def __init__(self, job_info: HardwareJobInfo, polling_interval: float = 30.0):
        """
        Initialize hardware job monitor.
        
        Args:
            job_info: Information about the job to monitor
            polling_interval: How often to poll for status updates (seconds)
        """
        self.job_info = job_info
        self.polling_interval = polling_interval
        
        # Monitoring state
        self.is_monitoring = False
        self.last_status = job_info.status
        
        # Callbacks
        self.status_callbacks: List[Callable[[HardwareJobInfo], None]] = []
        
        # Threading
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Logging
        self.logger = get_logger("HardwareJobMonitor")
    
    def start_monitoring(self):
        """Start background monitoring of the hardware job."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._stop_monitoring.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name=f"HWJobMonitor-{self.job_info.job_id}"
        )
        self._monitor_thread.start()
        
        self.logger.info(f"Started monitoring hardware job {self.job_info.job_id}")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self._stop_monitoring.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info(f"Stopped monitoring hardware job {self.job_info.job_id}")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self._stop_monitoring.wait(self.polling_interval):
            try:
                self._poll_job_status()
            except Exception as e:
                self.logger.error(f"Error polling job status: {e}")
                
                # Update with error status
                self.job_info.status = "error"
                self.job_info.error_message = str(e)
                self.job_info.last_updated = datetime.now()
                self._notify_status_change()
    
    def _poll_job_status(self):
        """Poll the hardware provider for job status."""
        # This is a placeholder implementation
        # In a real implementation, this would make API calls to the provider
        
        old_status = self.job_info.status
        
        # Simulate status progression for demonstration
        if self.job_info.status == "queued":
            # Simulate queue movement
            if self.job_info.queue_position and self.job_info.queue_position > 1:
                self.job_info.queue_position -= 1
                self.job_info.message = f"Queue position: {self.job_info.queue_position}"
            else:
                self.job_info.status = "running"
                self.job_info.queue_position = None
                self.job_info.message = "Job is now running"
        
        elif self.job_info.status == "running":
            # Simulate job completion (for demo)
            import random
            if random.random() < 0.1:  # 10% chance per poll
                self.job_info.status = "completed"
                self.job_info.message = "Job completed successfully"
        
        self.job_info.last_updated = datetime.now()
        
        # Notify if status changed
        if self.job_info.status != old_status:
            self.logger.info(f"Job {self.job_info.job_id} status changed: {old_status} -> {self.job_info.status}")
            self._notify_status_change()
    
    def add_status_callback(self, callback: Callable[[HardwareJobInfo], None]):
        """Add callback for status updates."""
        self.status_callbacks.append(callback)
    
    def remove_status_callback(self, callback: Callable[[HardwareJobInfo], None]):
        """Remove status callback."""
        if callback in self.status_callbacks:
            self.status_callbacks.remove(callback)
    
    def _notify_status_change(self):
        """Notify all callbacks of status change."""
        for callback in self.status_callbacks:
            try:
                callback(self.job_info)
            except Exception as e:
                self.logger.error(f"Error in status callback: {e}")

class StatusMonitor:
    """
    Central status monitoring system for all external dependencies.
    
    Coordinates monitoring of hardware jobs, system status, and other
    external resources with centralized callback management.
    """
    
    def __init__(self, default_polling_interval: float = 30.0):
        """
        Initialize status monitor.
        
        Args:
            default_polling_interval: Default polling interval for new monitors
        """
        self.default_polling_interval = default_polling_interval
        
        # Monitored items
        self.hardware_monitors: Dict[str, HardwareJobMonitor] = {}
        self.system_status: Dict[str, HardwareStatus] = {}
        self.status_history: List[StatusUpdate] = []
        
        # Global callbacks
        self.global_callbacks: List[Callable[[StatusUpdate], None]] = []
        
        # Threading
        self._lock = threading.RLock()
        self._system_monitor_thread: Optional[threading.Thread] = None
        self._stop_system_monitoring = threading.Event()
        
        # Logging and monitoring
        self.logger = get_logger("StatusMonitor")
        self.monitor = get_monitor()
        
        # Statistics
        self.stats = defaultdict(int)
        
        # Start system monitoring
        self._start_system_monitoring()
    
    def _start_system_monitoring(self):
        """Start background system status monitoring."""
        self._system_monitor_thread = threading.Thread(
            target=self._system_monitoring_loop,
            daemon=True,
            name="SystemStatusMonitor"
        )
        self._system_monitor_thread.start()
    
    def _system_monitoring_loop(self):
        """Background loop for system status monitoring."""
        while not self._stop_system_monitoring.wait(60.0):  # Check every minute
            try:
                self._check_system_status()
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
    
    def _check_system_status(self):
        """Check overall system status."""
        # Placeholder implementation for system health checks
        # In a real implementation, this might check:
        # - Network connectivity to quantum providers
        # - API endpoint availability
        # - Authentication status
        # - Resource availability
        
        # For demonstration, we'll just update some basic status
        current_time = datetime.now()
        
        # Simulate periodic status updates
        systems = ["local_simulator", "network_connection", "auth_service"]
        for system in systems:
            if system not in self.system_status:
                self.system_status[system] = HardwareStatus.ONLINE
            
            # Occasionally simulate status changes
            import random
            if random.random() < 0.05:  # 5% chance
                old_status = self.system_status[system]
                new_status = random.choice(list(HardwareStatus))
                
                if old_status != new_status:
                    self.system_status[system] = new_status
                    
                    update = StatusUpdate(
                        item_id=system,
                        item_type="system",
                        status=new_status.value,
                        message=f"System {system} status changed"
                    )
                    
                    self._record_status_update(update)
    
    def add_hardware_job(self, job_info: HardwareJobInfo) -> str:
        """
        Add a hardware job for monitoring.
        
        Args:
            job_info: Information about the hardware job
            
        Returns:
            Monitor ID for this job
        """
        with self._lock:
            monitor = HardwareJobMonitor(job_info, self.default_polling_interval)
            
            # Add callback to receive status updates
            monitor.add_status_callback(self._on_hardware_job_update)
            
            self.hardware_monitors[job_info.job_id] = monitor
            monitor.start_monitoring()
            
            self.stats['hardware_jobs_added'] += 1
            
            self.logger.info(f"Added hardware job {job_info.job_id} for monitoring")
            
            return job_info.job_id
    
    def remove_hardware_job(self, job_id: str) -> bool:
        """
        Remove a hardware job from monitoring.
        
        Args:
            job_id: ID of the job to stop monitoring
            
        Returns:
            True if job was found and removed
        """
        with self._lock:
            monitor = self.hardware_monitors.get(job_id)
            if monitor:
                monitor.stop_monitoring()
                del self.hardware_monitors[job_id]
                
                self.stats['hardware_jobs_removed'] += 1
                self.logger.info(f"Removed hardware job {job_id} from monitoring")
                return True
            
            return False
    
    def _on_hardware_job_update(self, job_info: HardwareJobInfo):
        """Handle hardware job status updates."""
        update = StatusUpdate(
            item_id=job_info.job_id,
            item_type="hardware_job",
            status=job_info.status,
            message=f"Hardware job status: {job_info.status}",
            metadata={
                'provider': job_info.provider_name,
                'device': job_info.device_name,
                'queue_position': job_info.queue_position,
                'provider_job_id': job_info.provider_job_id
            }
        )
        
        self._record_status_update(update)
    
    def _record_status_update(self, update: StatusUpdate):
        """Record a status update and notify callbacks."""
        with self._lock:
            self.status_history.append(update)
            
            # Keep only recent history (last 1000 updates)
            if len(self.status_history) > 1000:
                self.status_history = self.status_history[-1000:]
            
            self.stats['total_status_updates'] += 1
            self.stats[f'{update.item_type}_updates'] += 1
        
        # Notify global callbacks
        for callback in self.global_callbacks:
            try:
                callback(update)
            except Exception as e:
                self.logger.error(f"Error in global status callback: {e}")
        
        self.logger.debug(f"Status update: {update.item_type} {update.item_id} -> {update.status}")
    
    def get_hardware_job_status(self, job_id: str) -> Optional[HardwareJobInfo]:
        """Get current status of a hardware job."""
        with self._lock:
            monitor = self.hardware_monitors.get(job_id)
            return monitor.job_info if monitor else None
    
    def get_all_hardware_jobs(self) -> List[HardwareJobInfo]:
        """Get status of all monitored hardware jobs."""
        with self._lock:
            return [monitor.job_info for monitor in self.hardware_monitors.values()]
    
    def get_system_status(self) -> Dict[str, HardwareStatus]:
        """Get current system status."""
        with self._lock:
            return dict(self.system_status)
    
    def get_recent_updates(self, hours: int = 24) -> List[StatusUpdate]:
        """Get recent status updates."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        with self._lock:
            return [update for update in self.status_history 
                   if update.timestamp >= cutoff_time]
    
    def get_updates_by_type(self, item_type: str) -> List[StatusUpdate]:
        """Get status updates for a specific item type."""
        with self._lock:
            return [update for update in self.status_history 
                   if update.item_type == item_type]
    
    def add_global_callback(self, callback: Callable[[StatusUpdate], None]):
        """Add global status update callback."""
        self.global_callbacks.append(callback)
    
    def remove_global_callback(self, callback: Callable[[StatusUpdate], None]):
        """Remove global status update callback."""
        if callback in self.global_callbacks:
            self.global_callbacks.remove(callback)
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        with self._lock:
            stats = dict(self.stats)
            stats.update({
                'active_hardware_monitors': len(self.hardware_monitors),
                'system_statuses': len(self.system_status),
                'total_status_history': len(self.status_history),
                'hardware_jobs_by_status': defaultdict(int),
                'system_statuses_by_type': defaultdict(int)
            })
            
            # Count hardware jobs by status
            for monitor in self.hardware_monitors.values():
                stats['hardware_jobs_by_status'][monitor.job_info.status] += 1
            
            # Count system statuses
            for status in self.system_status.values():
                stats['system_statuses_by_type'][status.value] += 1
            
            return stats
    
    def shutdown(self):
        """Shutdown status monitoring and cleanup resources."""
        self.logger.info("Shutting down status monitor")
        
        # Stop system monitoring
        self._stop_system_monitoring.set()
        if self._system_monitor_thread:
            self._system_monitor_thread.join(timeout=5.0)
        
        # Stop all hardware job monitors
        with self._lock:
            for monitor in self.hardware_monitors.values():
                monitor.stop_monitoring()
            self.hardware_monitors.clear()
        
        self.logger.info("Status monitor shutdown complete")

# Global status monitor instance
_global_status_monitor: Optional[StatusMonitor] = None
_status_monitor_lock = threading.Lock()

def get_status_monitor() -> StatusMonitor:
    """Get the global status monitor instance."""
    global _global_status_monitor
    
    with _status_monitor_lock:
        if _global_status_monitor is None:
            _global_status_monitor = StatusMonitor()
    
    return _global_status_monitor

# Convenience functions for hardware job monitoring

def monitor_hardware_job(provider_job_id: str,
                        provider_name: str,
                        device_name: str,
                        job_id: Optional[str] = None) -> str:
    """
    Convenience function to start monitoring a hardware job.
    
    Args:
        provider_job_id: Job ID from the hardware provider
        provider_name: Name of the quantum provider
        device_name: Name of the quantum device
        job_id: Optional local job ID
        
    Returns:
        Local job ID for the monitored job
    """
    import uuid
    
    if not job_id:
        job_id = str(uuid.uuid4())
    
    job_info = HardwareJobInfo(
        job_id=job_id,
        provider_job_id=provider_job_id,
        provider_name=provider_name,
        device_name=device_name,
        status="queued"
    )
    
    monitor = get_status_monitor()
    return monitor.add_hardware_job(job_info)

def stop_monitoring_hardware_job(job_id: str) -> bool:
    """
    Convenience function to stop monitoring a hardware job.
    
    Args:
        job_id: Local job ID to stop monitoring
        
    Returns:
        True if job was found and monitoring stopped
    """
    monitor = get_status_monitor()
    return monitor.remove_hardware_job(job_id) 