"""
Real-Time Execution Monitoring Dashboard

This module provides a comprehensive dashboard interface for monitoring quantum
executions in real-time, including simulation progress, hardware job status,
and system resource monitoring with web API and notification capabilities.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from collections import defaultdict
from quantum_platform.execution.job_manager import get_job_manager, ExecutionJob, JobStatus, JobType
from quantum_platform.execution.progress_tracker import ProgressTracker, SimulationProgress
from quantum_platform.execution.status_monitor import get_status_monitor, StatusUpdate
from quantum_platform.observability.logging import get_logger
from quantum_platform.observability.monitor import get_monitor

@dataclass
class DashboardNotification:
    """Notification for dashboard users."""
    notification_id: str
    title: str
    message: str
    notification_type: str  # "info", "warning", "error", "success"
    timestamp: datetime = field(default_factory=datetime.now)
    job_id: Optional[str] = None
    auto_dismiss: bool = True
    dismiss_after: int = 5000  # milliseconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'notification_id': self.notification_id,
            'title': self.title,
            'message': self.message,
            'type': self.notification_type,
            'timestamp': self.timestamp.isoformat(),
            'job_id': self.job_id,
            'auto_dismiss': self.auto_dismiss,
            'dismiss_after': self.dismiss_after
        }

@dataclass
class DashboardState:
    """Current state of the dashboard."""
    active_jobs: List[Dict[str, Any]] = field(default_factory=list)
    hardware_jobs: List[Dict[str, Any]] = field(default_factory=list)
    system_status: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    notifications: List[DashboardNotification] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'active_jobs': self.active_jobs,
            'hardware_jobs': self.hardware_jobs,
            'system_status': self.system_status,
            'resource_usage': self.resource_usage,
            'notifications': [n.to_dict() for n in self.notifications],
            'statistics': self.statistics,
            'last_updated': self.last_updated.isoformat()
        }

class DashboardAPI:
    """
    API interface for the execution monitoring dashboard.
    
    Provides REST-like methods for accessing dashboard data and
    controlling execution jobs.
    """
    
    def __init__(self, dashboard: 'ExecutionDashboard'):
        """Initialize dashboard API."""
        self.dashboard = dashboard
        self.logger = get_logger("DashboardAPI")
    
    def get_dashboard_state(self) -> Dict[str, Any]:
        """Get current dashboard state."""
        return self.dashboard.get_current_state().to_dict()
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get list of active jobs."""
        job_manager = get_job_manager()
        active_jobs = job_manager.get_active_jobs()
        return [job.to_dict() for job in active_jobs]
    
    def get_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific job."""
        job_manager = get_job_manager()
        job = job_manager.get_job(job_id)
        return job.to_dict() if job else None
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a job.
        
        Args:
            job_id: ID of job to cancel
            
        Returns:
            Result of cancellation attempt
        """
        job_manager = get_job_manager()
        success = job_manager.cancel_job(job_id)
        
        if success:
            self.dashboard._add_notification(
                "Job Cancelled",
                f"Job {job_id} has been cancelled",
                "info",
                job_id=job_id
            )
        
        return {
            'success': success,
            'message': f"Job {job_id} {'cancelled' if success else 'could not be cancelled'}"
        }
    
    def get_hardware_jobs(self) -> List[Dict[str, Any]]:
        """Get list of hardware jobs being monitored."""
        status_monitor = get_status_monitor()
        hardware_jobs = status_monitor.get_all_hardware_jobs()
        return [job.to_dict() for job in hardware_jobs]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status_monitor = get_status_monitor()
        system_status = status_monitor.get_system_status()
        return {name: status.value for name, status in system_status.items()}
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        monitor = get_monitor()
        usage_data = monitor.get_recent_resource_usage(duration=timedelta(minutes=1))
        
        if usage_data:
            latest = usage_data[-1]
            return latest.to_dict()
        else:
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        job_manager = get_job_manager()
        status_monitor = get_status_monitor()
        system_monitor = get_monitor()
        
        job_stats = job_manager.get_job_statistics()
        monitoring_stats = status_monitor.get_monitoring_statistics()
        system_summary = system_monitor.get_system_summary()
        
        return {
            'jobs': job_stats,
            'monitoring': monitoring_stats,
            'system': system_summary
        }
    
    def get_notifications(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent notifications."""
        return self.dashboard.get_recent_notifications(hours)
    
    def dismiss_notification(self, notification_id: str) -> Dict[str, Any]:
        """Dismiss a notification."""
        success = self.dashboard.dismiss_notification(notification_id)
        return {
            'success': success,
            'message': f"Notification {'dismissed' if success else 'not found'}"
        }

class ExecutionDashboard:
    """
    Main execution monitoring dashboard.
    
    Coordinates real-time monitoring of quantum executions, provides
    status updates, notifications, and resource monitoring capabilities.
    """
    
    def __init__(self, update_interval: float = 1.0, max_notifications: int = 100):
        """
        Initialize execution dashboard.
        
        Args:
            update_interval: How often to update dashboard state (seconds)
            max_notifications: Maximum number of notifications to keep
        """
        self.update_interval = update_interval
        self.max_notifications = max_notifications
        
        # Dashboard state
        self.current_state = DashboardState()
        self.notifications: List[DashboardNotification] = []
        
        # Subscribers and callbacks
        self.state_callbacks: List[Callable[[DashboardState], None]] = []
        self.notification_callbacks: List[Callable[[DashboardNotification], None]] = []
        
        # Threading
        self._lock = threading.RLock()
        self._update_thread: Optional[threading.Thread] = None
        self._stop_updates = threading.Event()
        
        # Components
        self.job_manager = get_job_manager()
        self.status_monitor = get_status_monitor()
        self.system_monitor = get_monitor()
        
        # API interface
        self.api = DashboardAPI(self)
        
        # Logging
        self.logger = get_logger("ExecutionDashboard")
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Start dashboard updates
        self.start_monitoring()
        
        self.logger.info("Execution dashboard initialized")
    
    def _setup_callbacks(self):
        """Setup callbacks from other components."""
        # Job status change callbacks
        self.job_manager.add_status_change_callback(self._on_job_status_change)
        
        # Hardware job status callbacks
        self.status_monitor.add_global_callback(self._on_status_update)
    
    def start_monitoring(self):
        """Start dashboard monitoring and updates."""
        if self._update_thread and self._update_thread.is_alive():
            return
        
        self._stop_updates.clear()
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="DashboardUpdater"
        )
        self._update_thread.start()
        
        self.logger.info("Started dashboard monitoring")
    
    def stop_monitoring(self):
        """Stop dashboard monitoring."""
        self._stop_updates.set()
        if self._update_thread:
            self._update_thread.join(timeout=5.0)
        
        self.logger.info("Stopped dashboard monitoring")
    
    def _update_loop(self):
        """Background update loop for dashboard state."""
        while not self._stop_updates.wait(self.update_interval):
            try:
                self._update_dashboard_state()
            except Exception as e:
                self.logger.error(f"Error updating dashboard state: {e}")
    
    def _update_dashboard_state(self):
        """Update current dashboard state."""
        with self._lock:
            # Update active jobs
            active_jobs = self.job_manager.get_active_jobs()
            self.current_state.active_jobs = [job.to_dict() for job in active_jobs]
            
            # Update hardware jobs
            hardware_jobs = self.status_monitor.get_all_hardware_jobs()
            self.current_state.hardware_jobs = [job.to_dict() for job in hardware_jobs]
            
            # Update system status
            system_status = self.status_monitor.get_system_status()
            self.current_state.system_status = {
                name: status.value for name, status in system_status.items()
            }
            
            # Update resource usage
            usage_data = self.system_monitor.get_recent_resource_usage(duration=timedelta(minutes=1))
            if usage_data:
                self.current_state.resource_usage = usage_data[-1].to_dict()
            
            # Update statistics
            self.current_state.statistics = {
                'jobs': self.job_manager.get_job_statistics(),
                'monitoring': self.status_monitor.get_monitoring_statistics(),
                'system': self.system_monitor.get_system_summary()
            }
            
            # Update timestamp
            self.current_state.last_updated = datetime.now()
            
            # Clean up old notifications
            self._cleanup_notifications()
        
        # Notify subscribers
        self._notify_state_callbacks()
    
    def _on_job_status_change(self, job: ExecutionJob):
        """Handle job status changes."""
        if job.status == JobStatus.COMPLETED:
            self._add_notification(
                "Job Completed",
                f"Job '{job.name}' completed successfully",
                "success",
                job_id=job.job_id
            )
        elif job.status == JobStatus.FAILED:
            self._add_notification(
                "Job Failed",
                f"Job '{job.name}' failed: {job.error_message or 'Unknown error'}",
                "error",
                job_id=job.job_id
            )
        elif job.status == JobStatus.CANCELLED:
            self._add_notification(
                "Job Cancelled",
                f"Job '{job.name}' was cancelled",
                "warning",
                job_id=job.job_id
            )
    
    def _on_status_update(self, update: StatusUpdate):
        """Handle status updates from monitoring systems."""
        if update.item_type == "hardware_job":
            status = update.status
            if status in ["completed", "failed", "error"]:
                notification_type = "success" if status == "completed" else "error"
                self._add_notification(
                    f"Hardware Job {status.title()}",
                    f"Hardware job {update.item_id} {status}",
                    notification_type,
                    job_id=update.item_id
                )
    
    def _add_notification(self, title: str, message: str, notification_type: str, 
                         job_id: Optional[str] = None, auto_dismiss: bool = True):
        """Add a new notification."""
        import uuid
        
        notification = DashboardNotification(
            notification_id=str(uuid.uuid4()),
            title=title,
            message=message,
            notification_type=notification_type,
            job_id=job_id,
            auto_dismiss=auto_dismiss
        )
        
        with self._lock:
            self.notifications.append(notification)
            
            # Keep only recent notifications
            if len(self.notifications) > self.max_notifications:
                self.notifications = self.notifications[-self.max_notifications:]
        
        # Notify subscribers
        self._notify_notification_callbacks(notification)
        
        self.logger.info(f"Added notification: {title} - {message}")
    
    def _cleanup_notifications(self):
        """Remove old auto-dismiss notifications."""
        cutoff_time = datetime.now() - timedelta(minutes=10)
        
        with self._lock:
            self.notifications = [
                n for n in self.notifications
                if not (n.auto_dismiss and n.timestamp < cutoff_time)
            ]
    
    def get_current_state(self) -> DashboardState:
        """Get current dashboard state."""
        with self._lock:
            return self.current_state
    
    def get_recent_notifications(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent notifications."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent = [n for n in self.notifications if n.timestamp >= cutoff_time]
            return [n.to_dict() for n in recent]
    
    def dismiss_notification(self, notification_id: str) -> bool:
        """Dismiss a notification by ID."""
        with self._lock:
            for i, notification in enumerate(self.notifications):
                if notification.notification_id == notification_id:
                    del self.notifications[i]
                    return True
            return False
    
    def add_state_callback(self, callback: Callable[[DashboardState], None]):
        """Add callback for dashboard state updates."""
        self.state_callbacks.append(callback)
    
    def remove_state_callback(self, callback: Callable[[DashboardState], None]):
        """Remove state update callback."""
        if callback in self.state_callbacks:
            self.state_callbacks.remove(callback)
    
    def add_notification_callback(self, callback: Callable[[DashboardNotification], None]):
        """Add callback for new notifications."""
        self.notification_callbacks.append(callback)
    
    def remove_notification_callback(self, callback: Callable[[DashboardNotification], None]):
        """Remove notification callback."""
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)
    
    def _notify_state_callbacks(self):
        """Notify all state update callbacks."""
        for callback in self.state_callbacks:
            try:
                callback(self.current_state)
            except Exception as e:
                self.logger.error(f"Error in state callback: {e}")
    
    def _notify_notification_callbacks(self, notification: DashboardNotification):
        """Notify all notification callbacks."""
        for callback in self.notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                self.logger.error(f"Error in notification callback: {e}")
    
    def export_dashboard_data(self, format_type: str = "json") -> Union[Dict[str, Any], str]:
        """
        Export dashboard data for analysis or backup.
        
        Args:
            format_type: Export format ("json" or "dict")
            
        Returns:
            Exported dashboard data
        """
        with self._lock:
            data = {
                'current_state': self.current_state.to_dict(),
                'notifications': [n.to_dict() for n in self.notifications],
                'export_timestamp': datetime.now().isoformat()
            }
        
        if format_type == "json":
            return json.dumps(data, indent=2)
        else:
            return data
    
    def shutdown(self):
        """Shutdown dashboard and cleanup resources."""
        self.logger.info("Shutting down execution dashboard")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Remove callbacks
        self.job_manager.remove_status_change_callback(self._on_job_status_change)
        # Note: Cannot easily remove global status callbacks without ID
        
        self.logger.info("Execution dashboard shutdown complete")

# Global dashboard instance
_global_dashboard: Optional[ExecutionDashboard] = None
_dashboard_lock = threading.Lock()

def get_dashboard() -> ExecutionDashboard:
    """Get the global execution dashboard instance."""
    global _global_dashboard
    
    with _dashboard_lock:
        if _global_dashboard is None:
            _global_dashboard = ExecutionDashboard()
    
    return _global_dashboard

# Simple web server for dashboard (optional - for demonstration)
class SimpleDashboardServer:
    """
    Simple HTTP server for dashboard access.
    
    This is a basic implementation for demonstration purposes.
    In production, you would use a proper web framework.
    """
    
    def __init__(self, dashboard: ExecutionDashboard, port: int = 8080):
        """Initialize dashboard server."""
        self.dashboard = dashboard
        self.port = port
        self.logger = get_logger("DashboardServer")
        self._server_thread: Optional[threading.Thread] = None
        self._stop_server = threading.Event()
    
    def start(self):
        """Start the dashboard server."""
        try:
            import http.server
            import socketserver
            from urllib.parse import urlparse, parse_qs
            
            class DashboardHandler(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    path = urlparse(self.path).path
                    query = parse_qs(urlparse(self.path).query)
                    
                    if path == "/api/dashboard":
                        self.send_json_response(self.server.dashboard.api.get_dashboard_state())
                    elif path == "/api/jobs":
                        self.send_json_response(self.server.dashboard.api.get_active_jobs())
                    elif path == "/api/hardware":
                        self.send_json_response(self.server.dashboard.api.get_hardware_jobs())
                    elif path == "/api/status":
                        self.send_json_response(self.server.dashboard.api.get_system_status())
                    elif path == "/api/resources":
                        self.send_json_response(self.server.dashboard.api.get_resource_usage())
                    elif path == "/api/stats":
                        self.send_json_response(self.server.dashboard.api.get_statistics())
                    elif path == "/api/notifications":
                        hours = int(query.get('hours', [24])[0])
                        self.send_json_response(self.server.dashboard.api.get_notifications(hours))
                    elif path == "/":
                        self.send_html_response(self.get_dashboard_html())
                    else:
                        self.send_error(404)
                
                def do_POST(self):
                    path = urlparse(self.path).path
                    
                    if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                        job_id = path.split("/")[-2]
                        result = self.server.dashboard.api.cancel_job(job_id)
                        self.send_json_response(result)
                    elif path.startswith("/api/notifications/") and path.endswith("/dismiss"):
                        notification_id = path.split("/")[-2]
                        result = self.server.dashboard.api.dismiss_notification(notification_id)
                        self.send_json_response(result)
                    else:
                        self.send_error(404)
                
                def send_json_response(self, data):
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode())
                
                def send_html_response(self, html):
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(html.encode())
                
                def get_dashboard_html(self):
                    return """
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Quantum Execution Dashboard</title>
                        <style>
                            body { font-family: Arial, sans-serif; margin: 20px; }
                            .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                            .job { margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 3px; }
                            .status { font-weight: bold; }
                            .running { color: #007bff; }
                            .completed { color: #28a745; }
                            .failed { color: #dc3545; }
                            .cancelled { color: #ffc107; }
                            button { padding: 5px 10px; margin: 5px; }
                        </style>
                    </head>
                    <body>
                        <h1>Quantum Execution Dashboard</h1>
                        <div id="dashboard">Loading...</div>
                        <script>
                            function updateDashboard() {
                                fetch('/api/dashboard')
                                    .then(response => response.json())
                                    .then(data => {
                                        document.getElementById('dashboard').innerHTML = renderDashboard(data);
                                    });
                            }
                            
                            function renderDashboard(data) {
                                let html = '<div class="section"><h2>Active Jobs</h2>';
                                data.active_jobs.forEach(job => {
                                    html += `<div class="job">
                                        <strong>${job.name}</strong> (${job.job_type}) 
                                        <span class="status ${job.status}">${job.status}</span><br>
                                        Progress: ${job.progress.toFixed(1)}%<br>
                                        <button onclick="cancelJob('${job.job_id}')">Cancel</button>
                                    </div>`;
                                });
                                html += '</div>';
                                
                                html += '<div class="section"><h2>System Status</h2>';
                                Object.entries(data.system_status).forEach(([system, status]) => {
                                    html += `<div>${system}: <span class="status">${status}</span></div>`;
                                });
                                html += '</div>';
                                
                                html += '<div class="section"><h2>Notifications</h2>';
                                data.notifications.forEach(notif => {
                                    html += `<div class="job">
                                        <strong>${notif.title}</strong>: ${notif.message}<br>
                                        <small>${notif.timestamp}</small>
                                        <button onclick="dismissNotification('${notif.notification_id}')">Dismiss</button>
                                    </div>`;
                                });
                                html += '</div>';
                                
                                return html;
                            }
                            
                            function cancelJob(jobId) {
                                fetch(`/api/jobs/${jobId}/cancel`, {method: 'POST'})
                                    .then(() => updateDashboard());
                            }
                            
                            function dismissNotification(notifId) {
                                fetch(`/api/notifications/${notifId}/dismiss`, {method: 'POST'})
                                    .then(() => updateDashboard());
                            }
                            
                            // Update every 2 seconds
                            setInterval(updateDashboard, 2000);
                            updateDashboard();
                        </script>
                    </body>
                    </html>
                    """
                
                def log_message(self, format, *args):
                    # Suppress default logging
                    pass
            
            class DashboardServer(socketserver.TCPServer):
                def __init__(self, server_address, handler_class, dashboard):
                    super().__init__(server_address, handler_class)
                    self.dashboard = dashboard
            
            def server_thread():
                with DashboardServer(("localhost", self.port), DashboardHandler, self.dashboard) as httpd:
                    self.logger.info(f"Dashboard server running on http://localhost:{self.port}")
                    while not self._stop_server.is_set():
                        httpd.handle_request()
            
            self._server_thread = threading.Thread(target=server_thread, daemon=True)
            self._server_thread.start()
            
        except ImportError:
            self.logger.warning("HTTP server not available - dashboard server not started")
        except Exception as e:
            self.logger.error(f"Failed to start dashboard server: {e}")
    
    def stop(self):
        """Stop the dashboard server."""
        self._stop_server.set()
        if self._server_thread:
            self._server_thread.join(timeout=5.0)
        self.logger.info("Dashboard server stopped")

def start_dashboard_server(port: int = 8080) -> SimpleDashboardServer:
    """Start a simple dashboard web server."""
    dashboard = get_dashboard()
    server = SimpleDashboardServer(dashboard, port)
    server.start()
    return server 