"""
System Monitoring and Performance Metrics

This module provides comprehensive monitoring of system performance, resource usage,
and operational metrics to help understand platform behavior and identify bottlenecks.
"""

import threading
import time
import psutil
import gc
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
from quantum_platform.observability.logging import get_logger

@dataclass
class PerformanceMetrics:
    """Container for performance metrics data."""
    operation_name: str
    duration: float
    start_time: datetime
    end_time: datetime
    component: str
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'operation_name': self.operation_name,
            'duration': self.duration,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'component': self.component,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata
        }

@dataclass
class ResourceUsage:
    """Container for system resource usage data."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    thread_count: int
    active_loggers: int
    gc_stats: Dict[str, Any] = field(default_factory=dict)
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert resource usage to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'thread_count': self.thread_count,
            'active_loggers': self.active_loggers,
            'gc_stats': self.gc_stats,
            'custom_metrics': self.custom_metrics
        }

class SystemMonitor:
    """
    Comprehensive system monitoring for the quantum computing platform.
    
    This class tracks performance metrics, resource usage, and provides
    insights into system behavior and potential bottlenecks.
    """
    
    def __init__(self, max_history: int = 1000, enable_continuous_monitoring: bool = True):
        """
        Initialize the system monitor.
        
        Args:
            max_history: Maximum number of metrics/resource entries to keep
            enable_continuous_monitoring: Whether to continuously collect resource metrics
        """
        self.max_history = max_history
        self.enable_continuous_monitoring = enable_continuous_monitoring
        
        # Data storage
        self.performance_metrics: deque = deque(maxlen=max_history)
        self.resource_history: deque = deque(maxlen=max_history)
        self.component_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_operations': 0,
            'total_duration': 0.0,
            'avg_duration': 0.0,
            'success_count': 0,
            'failure_count': 0,
            'last_operation': None
        })
        
        # Active monitoring
        self.active_operations: Dict[str, datetime] = {}
        self.custom_counters: Dict[str, int] = defaultdict(int)
        self.custom_gauges: Dict[str, float] = defaultdict(float)
        
        # Threading
        self._lock = threading.RLock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Logger
        self.logger = get_logger("Monitor")
        
        # Start continuous monitoring if enabled
        if self.enable_continuous_monitoring:
            self.start_continuous_monitoring()
    
    def start_continuous_monitoring(self, interval: float = 5.0):
        """
        Start continuous resource monitoring in background thread.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True,
            name="SystemMonitor"
        )
        self._monitoring_thread.start()
        self.logger.info(f"Started continuous monitoring with {interval}s interval")
    
    def stop_continuous_monitoring(self):
        """Stop continuous resource monitoring."""
        if self._monitoring_thread is not None:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=2.0)
            self.logger.info("Stopped continuous monitoring")
    
    def _monitoring_loop(self, interval: float):
        """Background monitoring loop."""
        while not self._stop_monitoring.wait(interval):
            try:
                self.collect_resource_metrics()
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def collect_resource_metrics(self) -> ResourceUsage:
        """
        Collect current system resource usage.
        
        Returns:
            ResourceUsage object with current metrics
        """
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            thread_count = threading.active_count()
            
            # Garbage collection stats
            gc_stats = {
                'collections': gc.get_stats(),
                'count': gc.get_count(),
                'threshold': gc.get_threshold()
            }
            
            # Platform-specific metrics
            with self._lock:
                active_loggers = len(self.active_operations)
                custom_metrics = dict(self.custom_gauges)
            
            resource_usage = ResourceUsage(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_info.percent,
                memory_used_mb=memory_info.used / (1024 * 1024),
                memory_available_mb=memory_info.available / (1024 * 1024),
                thread_count=thread_count,
                active_loggers=active_loggers,
                gc_stats=gc_stats,
                custom_metrics=custom_metrics
            )
            
            with self._lock:
                self.resource_history.append(resource_usage)
            
            return resource_usage
            
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}")
            return ResourceUsage(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                thread_count=0,
                active_loggers=0
            )
    
    @contextmanager
    def measure_operation(self, operation_name: str, component: str = "Platform", metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for measuring operation performance.
        
        Args:
            operation_name: Name of the operation being measured
            component: Component performing the operation
            metadata: Additional metadata to include
        """
        operation_id = f"{component}::{operation_name}::{id(threading.current_thread())}"
        start_time = datetime.now()
        
        with self._lock:
            self.active_operations[operation_id] = start_time
        
        self.logger.debug(f"Started measuring {operation_name} in {component}")
        
        try:
            yield
            
            # Success case
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            metric = PerformanceMetrics(
                operation_name=operation_name,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                component=component,
                success=True,
                metadata=metadata or {}
            )
            
            self._record_performance_metric(metric)
            self.logger.debug(f"Completed {operation_name} in {duration:.3f}s")
            
        except Exception as e:
            # Failure case
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            metric = PerformanceMetrics(
                operation_name=operation_name,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                component=component,
                success=False,
                error_message=str(e),
                metadata=metadata or {}
            )
            
            self._record_performance_metric(metric)
            self.logger.error(f"Failed {operation_name} after {duration:.3f}s: {e}")
            raise
            
        finally:
            with self._lock:
                self.active_operations.pop(operation_id, None)
    
    def _record_performance_metric(self, metric: PerformanceMetrics):
        """Record a performance metric and update component statistics."""
        with self._lock:
            # Add to history
            self.performance_metrics.append(metric)
            
            # Update component statistics
            stats = self.component_stats[metric.component]
            stats['total_operations'] += 1
            stats['total_duration'] += metric.duration
            stats['avg_duration'] = stats['total_duration'] / stats['total_operations']
            stats['last_operation'] = metric.operation_name
            
            if metric.success:
                stats['success_count'] += 1
            else:
                stats['failure_count'] += 1
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a custom counter."""
        with self._lock:
            self.custom_counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """Set a custom gauge value."""
        with self._lock:
            self.custom_gauges[name] = value
    
    def get_component_stats(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics for components.
        
        Args:
            component: Specific component to get stats for, or None for all
            
        Returns:
            Dictionary of component statistics
        """
        with self._lock:
            if component:
                return dict(self.component_stats.get(component, {}))
            else:
                return {comp: dict(stats) for comp, stats in self.component_stats.items()}
    
    def get_recent_metrics(self, component: Optional[str] = None, duration: Optional[timedelta] = None) -> List[PerformanceMetrics]:
        """
        Get recent performance metrics.
        
        Args:
            component: Filter by component name
            duration: Only include metrics from last duration period
            
        Returns:
            List of performance metrics
        """
        with self._lock:
            metrics = list(self.performance_metrics)
        
        # Filter by time if specified
        if duration:
            cutoff_time = datetime.now() - duration
            metrics = [m for m in metrics if m.start_time >= cutoff_time]
        
        # Filter by component if specified
        if component:
            metrics = [m for m in metrics if m.component == component]
        
        return metrics
    
    def get_recent_resource_usage(self, duration: Optional[timedelta] = None) -> List[ResourceUsage]:
        """
        Get recent resource usage data.
        
        Args:
            duration: Only include data from last duration period
            
        Returns:
            List of resource usage data
        """
        with self._lock:
            usage_data = list(self.resource_history)
        
        if duration:
            cutoff_time = datetime.now() - duration
            usage_data = [u for u in usage_data if u.timestamp >= cutoff_time]
        
        return usage_data
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive system summary.
        
        Returns:
            Dictionary with system performance and resource summary
        """
        recent_metrics = self.get_recent_metrics(duration=timedelta(minutes=5))
        recent_resources = self.get_recent_resource_usage(duration=timedelta(minutes=5))
        
        # Calculate summary statistics
        total_operations = len(recent_metrics)
        successful_operations = sum(1 for m in recent_metrics if m.success)
        avg_duration = sum(m.duration for m in recent_metrics) / max(total_operations, 1)
        
        # Resource averages
        avg_cpu = sum(r.cpu_percent for r in recent_resources) / max(len(recent_resources), 1)
        avg_memory = sum(r.memory_percent for r in recent_resources) / max(len(recent_resources), 1)
        
        with self._lock:
            active_ops = len(self.active_operations)
            component_count = len(self.component_stats)
            counter_summary = dict(self.custom_counters)
            gauge_summary = dict(self.custom_gauges)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'performance': {
                'total_operations_5min': total_operations,
                'successful_operations_5min': successful_operations,
                'success_rate_5min': successful_operations / max(total_operations, 1),
                'avg_duration_5min': avg_duration,
                'active_operations': active_ops
            },
            'resources': {
                'avg_cpu_5min': avg_cpu,
                'avg_memory_5min': avg_memory,
                'thread_count': threading.active_count(),
                'monitored_components': component_count
            },
            'custom_metrics': {
                'counters': counter_summary,
                'gauges': gauge_summary
            }
        }
    
    def export_metrics(self, format_type: str = "dict") -> Union[Dict[str, Any], str]:
        """
        Export all collected metrics.
        
        Args:
            format_type: Format for export ("dict" or "json")
            
        Returns:
            Exported metrics data
        """
        with self._lock:
            data = {
                'performance_metrics': [m.to_dict() for m in self.performance_metrics],
                'resource_history': [r.to_dict() for r in self.resource_history],
                'component_stats': dict(self.component_stats),
                'custom_counters': dict(self.custom_counters),
                'custom_gauges': dict(self.custom_gauges),
                'export_timestamp': datetime.now().isoformat()
            }
        
        if format_type == "json":
            import json
            return json.dumps(data, indent=2)
        else:
            return data
    
    def reset_metrics(self):
        """Reset all collected metrics and statistics."""
        with self._lock:
            self.performance_metrics.clear()
            self.resource_history.clear()
            self.component_stats.clear()
            self.custom_counters.clear()
            self.custom_gauges.clear()
            self.active_operations.clear()
        
        self.logger.info("Reset all monitoring metrics")

# Global monitor instance
_global_monitor: Optional[SystemMonitor] = None
_monitor_lock = threading.Lock()

def get_monitor() -> SystemMonitor:
    """Get the global system monitor instance."""
    global _global_monitor
    
    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = SystemMonitor()
    
    return _global_monitor

def measure_performance(operation_name: str, component: str = "Platform", metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator for measuring function performance.
    
    Args:
        operation_name: Name of the operation
        component: Component performing the operation
        metadata: Additional metadata
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            with monitor.measure_operation(operation_name, component, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator 