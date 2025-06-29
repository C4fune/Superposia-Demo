"""
Alert Management System

This module provides alerting and notification capabilities
for error reporting and user notifications.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4

from quantum_platform.observability.logging import get_logger


class AlertType(Enum):
    """Types of alerts."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"
    QUESTION = "question"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents an alert/notification."""
    id: str
    title: str
    message: str
    alert_type: AlertType
    severity: AlertSeverity
    timestamp: datetime
    component: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    auto_dismiss: bool = True
    dismiss_after: int = 5  # seconds
    persistent: bool = False
    acknowledged: bool = False
    dismissed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'type': self.alert_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'actions': self.actions,
            'metadata': self.metadata,
            'auto_dismiss': self.auto_dismiss,
            'dismiss_after': self.dismiss_after,
            'persistent': self.persistent,
            'acknowledged': self.acknowledged,
            'dismissed': self.dismissed
        }


class AlertManager:
    """
    Manages alerts and notifications throughout the platform.
    
    Provides a centralized system for displaying user notifications,
    error alerts, and interactive dialogs.
    """
    
    def __init__(self, max_alerts: int = 100):
        self.logger = get_logger(__name__)
        self.max_alerts = max_alerts
        
        # Alert storage
        self._alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._lock = threading.Lock()
        
        # Event handlers
        self._alert_handlers: List[Callable[[Alert], None]] = []
        self._dismissal_handlers: List[Callable[[Alert], None]] = []
        
        # Auto-dismissal thread
        self._auto_dismiss_thread: Optional[threading.Thread] = None
        self._running = True
        
        self.logger.info("Alert manager initialized")
        self._start_auto_dismiss_thread()
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add a handler for new alerts."""
        self._alert_handlers.append(handler)
    
    def add_dismissal_handler(self, handler: Callable[[Alert], None]):
        """Add a handler for alert dismissals."""
        self._dismissal_handlers.append(handler)
    
    def create_alert(
        self,
        title: str,
        message: str,
        alert_type: AlertType = AlertType.INFO,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        component: str = "",
        actions: Optional[List[Dict[str, Any]]] = None,
        auto_dismiss: bool = True,
        dismiss_after: int = 5,
        persistent: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Create a new alert.
        
        Args:
            title: Alert title
            message: Alert message
            alert_type: Type of alert
            severity: Severity level
            component: Component that generated the alert
            actions: List of action buttons/links
            auto_dismiss: Whether to auto-dismiss the alert
            dismiss_after: Seconds after which to auto-dismiss
            persistent: Whether to keep in history after dismissal
            metadata: Additional metadata
            
        Returns:
            Created alert
        """
        alert = Alert(
            id=str(uuid4()),
            title=title,
            message=message,
            alert_type=alert_type,
            severity=severity,
            timestamp=datetime.now(),
            component=component,
            actions=actions or [],
            auto_dismiss=auto_dismiss,
            dismiss_after=dismiss_after,
            persistent=persistent,
            metadata=metadata or {}
        )
        
        with self._lock:
            # Add to active alerts
            self._alerts[alert.id] = alert
            
            # Add to history
            self._alert_history.append(alert)
            
            # Limit history size
            if len(self._alert_history) > self.max_alerts:
                self._alert_history.pop(0)
        
        # Notify handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
        
        self.logger.info(f"Alert created: {alert.id} - {title}")
        return alert
    
    def error_alert(
        self,
        title: str,
        message: str,
        component: str = "",
        error_code: str = "",
        suggestions: Optional[List[str]] = None,
        show_details: bool = False,
        report_button: bool = True
    ) -> Alert:
        """Create an error alert with standard error handling features."""
        actions = []
        
        if show_details:
            actions.append({
                'label': 'Show Details',
                'action': 'show_details',
                'style': 'secondary'
            })
        
        if report_button:
            actions.append({
                'label': 'Report Error',
                'action': 'report_error',
                'style': 'primary'
            })
        
        actions.append({
            'label': 'Dismiss',
            'action': 'dismiss',
            'style': 'default'
        })
        
        metadata = {
            'error_code': error_code,
            'suggestions': suggestions or [],
            'show_details': show_details
        }
        
        return self.create_alert(
            title=title,
            message=message,
            alert_type=AlertType.ERROR,
            severity=AlertSeverity.HIGH,
            component=component,
            actions=actions,
            auto_dismiss=False,
            persistent=True,
            metadata=metadata
        )
    
    def warning_alert(
        self,
        title: str,
        message: str,
        component: str = "",
        auto_dismiss: bool = True
    ) -> Alert:
        """Create a warning alert."""
        return self.create_alert(
            title=title,
            message=message,
            alert_type=AlertType.WARNING,
            severity=AlertSeverity.MEDIUM,
            component=component,
            auto_dismiss=auto_dismiss,
            dismiss_after=8
        )
    
    def info_alert(
        self,
        title: str,
        message: str,
        component: str = "",
        auto_dismiss: bool = True
    ) -> Alert:
        """Create an info alert."""
        return self.create_alert(
            title=title,
            message=message,
            alert_type=AlertType.INFO,
            severity=AlertSeverity.LOW,
            component=component,
            auto_dismiss=auto_dismiss,
            dismiss_after=5
        )
    
    def success_alert(
        self,
        title: str,
        message: str,
        component: str = "",
        auto_dismiss: bool = True
    ) -> Alert:
        """Create a success alert."""
        return self.create_alert(
            title=title,
            message=message,
            alert_type=AlertType.SUCCESS,
            severity=AlertSeverity.LOW,
            component=component,
            auto_dismiss=auto_dismiss,
            dismiss_after=3
        )
    
    def question_alert(
        self,
        title: str,
        message: str,
        actions: List[Dict[str, Any]],
        component: str = ""
    ) -> Alert:
        """Create a question/confirmation alert."""
        return self.create_alert(
            title=title,
            message=message,
            alert_type=AlertType.QUESTION,
            severity=AlertSeverity.MEDIUM,
            component=component,
            actions=actions,
            auto_dismiss=False,
            persistent=True
        )
    
    def dismiss_alert(self, alert_id: str, acknowledged: bool = False) -> bool:
        """
        Dismiss an alert.
        
        Args:
            alert_id: ID of alert to dismiss
            acknowledged: Whether the user acknowledged the alert
            
        Returns:
            True if alert was dismissed
        """
        with self._lock:
            if alert_id not in self._alerts:
                return False
            
            alert = self._alerts[alert_id]
            alert.dismissed = True
            alert.acknowledged = acknowledged
            
            # Remove from active alerts
            del self._alerts[alert_id]
        
        # Notify dismissal handlers
        for handler in self._dismissal_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Dismissal handler failed: {e}")
        
        self.logger.debug(f"Alert dismissed: {alert_id}")
        return True
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return list(self._alerts.values())
    
    def get_alert_history(self) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            return self._alert_history.copy()
    
    def clear_all_alerts(self):
        """Clear all active alerts."""
        with self._lock:
            for alert in self._alerts.values():
                alert.dismissed = True
            self._alerts.clear()
        
        self.logger.info("All alerts cleared")
    
    def _start_auto_dismiss_thread(self):
        """Start the auto-dismissal background thread."""
        def auto_dismiss_worker():
            while self._running:
                try:
                    current_time = datetime.now()
                    to_dismiss = []
                    
                    with self._lock:
                        for alert in self._alerts.values():
                            if (alert.auto_dismiss and 
                                not alert.dismissed and
                                (current_time - alert.timestamp).total_seconds() > alert.dismiss_after):
                                to_dismiss.append(alert.id)
                    
                    # Dismiss alerts outside the lock
                    for alert_id in to_dismiss:
                        self.dismiss_alert(alert_id)
                    
                    time.sleep(1)  # Check every second
                    
                except Exception as e:
                    self.logger.error(f"Auto-dismiss thread error: {e}")
                    time.sleep(5)  # Wait longer on error
        
        self._auto_dismiss_thread = threading.Thread(
            target=auto_dismiss_worker,
            daemon=True,
            name="AlertAutoDissmiss"
        )
        self._auto_dismiss_thread.start()
    
    def shutdown(self):
        """Shutdown the alert manager."""
        self._running = False
        if self._auto_dismiss_thread:
            self._auto_dismiss_thread.join(timeout=1)
        
        self.logger.info("Alert manager shutdown")


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None
_manager_lock = threading.Lock()


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    
    if _alert_manager is None:
        with _manager_lock:
            if _alert_manager is None:
                _alert_manager = AlertManager()
    
    return _alert_manager


def create_error_alert(
    title: str,
    message: str,
    component: str = "",
    error_code: str = "",
    suggestions: Optional[List[str]] = None
) -> Alert:
    """Convenience function to create an error alert."""
    manager = get_alert_manager()
    return manager.error_alert(title, message, component, error_code, suggestions) 