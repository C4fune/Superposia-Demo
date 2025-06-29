"""
Security Audit Logging System

This module provides comprehensive audit logging for security events,
permission checks, and user actions for compliance and monitoring.
"""

import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from queue import Queue, Empty

from quantum_platform.security.user import UserContext


class AuditEventType(Enum):
    """Types of security events to audit."""
    
    # Authentication Events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    AUTHENTICATION_FAILED = "authentication_failed"
    
    # Authorization Events
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    ROLE_CHECK_PASSED = "role_check_passed"
    ROLE_CHECK_FAILED = "role_check_failed"
    
    # User Management Events
    USER_CREATED = "user_created"
    USER_DELETED = "user_deleted"
    USER_ROLE_CHANGED = "user_role_changed"
    USER_ACTIVATED = "user_activated"
    USER_DEACTIVATED = "user_deactivated"
    
    # System Operations
    PLUGIN_INSTALLED = "plugin_installed"
    PLUGIN_REMOVED = "plugin_removed"
    HARDWARE_ACCESS = "hardware_access"
    SIMULATION_RUN = "simulation_run"
    CIRCUIT_CREATED = "circuit_created"
    CIRCUIT_DELETED = "circuit_deleted"
    
    # Administrative Events
    ROLE_CREATED = "role_created"
    ROLE_DELETED = "role_deleted"
    SYSTEM_CONFIG_CHANGED = "system_config_changed"
    BACKUP_CREATED = "backup_created"
    DATA_EXPORTED = "data_exported"
    
    # Security Events
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ACCESS_DENIED = "access_denied"


@dataclass
class AuditEvent:
    """Represents a security audit event."""
    
    event_type: AuditEventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    username: Optional[str] = None
    user_role: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    outcome: str = "success"
    details: Dict[str, Any] = field(default_factory=dict)
    source_ip: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        """Auto-populate user information from context if available."""
        if not self.user_id:
            current_user = UserContext.get_current_user()
            if current_user:
                self.user_id = current_user.user_id
                self.username = current_user.username
                self.user_role = current_user.role
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "username": self.username,
            "user_role": self.user_role,
            "resource": self.resource,
            "action": self.action,
            "outcome": self.outcome,
            "details": self.details,
            "source_ip": self.source_ip,
            "session_id": self.session_id
        }
    
    def to_json(self) -> str:
        """Convert audit event to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create audit event from dictionary."""
        event = cls(
            event_type=AuditEventType(data["event_type"]),
            user_id=data.get("user_id"),
            username=data.get("username"),
            user_role=data.get("user_role"),
            resource=data.get("resource"),
            action=data.get("action"),
            outcome=data.get("outcome", "success"),
            details=data.get("details", {}),
            source_ip=data.get("source_ip"),
            session_id=data.get("session_id")
        )
        
        if data.get("timestamp"):
            event.timestamp = datetime.fromisoformat(data["timestamp"])
        
        return event
    
    def __str__(self) -> str:
        return f"AuditEvent({self.event_type.value}, {self.username}, {self.outcome})"


class AuditLogger:
    """Base audit logger interface."""
    
    def log_event(self, event: AuditEvent):
        """Log an audit event."""
        raise NotImplementedError
    
    def query_events(self, 
                    event_type: Optional[AuditEventType] = None,
                    user_id: Optional[str] = None,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None) -> List[AuditEvent]:
        """Query audit events with filters."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audit logging statistics."""
        raise NotImplementedError


class FileAuditLogger(AuditLogger):
    """File-based audit logger."""
    
    def __init__(self, log_file: str = "quantum_platform_audit.log"):
        """
        Initialize file audit logger.
        
        Args:
            log_file: Path to audit log file
        """
        self.log_file = Path(log_file)
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(self, event: AuditEvent):
        """Log event to file."""
        with self._lock:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(event.to_json() + '\n')
            except Exception as e:
                self._logger.error(f"Failed to write audit log: {e}")
    
    def query_events(self, 
                    event_type: Optional[AuditEventType] = None,
                    user_id: Optional[str] = None,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None) -> List[AuditEvent]:
        """Query events from log file."""
        events = []
        
        if not self.log_file.exists():
            return events
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        event = AuditEvent.from_dict(data)
                        
                        # Apply filters
                        if event_type and event.event_type != event_type:
                            continue
                        
                        if user_id and event.user_id != user_id:
                            continue
                        
                        if start_time and event.timestamp < start_time:
                            continue
                        
                        if end_time and event.timestamp > end_time:
                            continue
                        
                        events.append(event)
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        self._logger.warning(f"Invalid audit log line: {e}")
        
        except Exception as e:
            self._logger.error(f"Failed to read audit log: {e}")
        
        return events
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file audit logger statistics."""
        if not self.log_file.exists():
            return {"total_events": 0, "log_file_size": 0}
        
        try:
            events = self.query_events()
            file_size = self.log_file.stat().st_size
            
            event_types = {}
            for event in events:
                event_types[event.event_type.value] = event_types.get(event.event_type.value, 0) + 1
            
            return {
                "total_events": len(events),
                "log_file_size": file_size,
                "log_file_path": str(self.log_file),
                "events_by_type": event_types,
                "oldest_event": events[0].timestamp.isoformat() if events else None,
                "newest_event": events[-1].timestamp.isoformat() if events else None
            }
        
        except Exception as e:
            self._logger.error(f"Failed to get audit stats: {e}")
            return {"error": str(e)}


class InMemoryAuditLogger(AuditLogger):
    """In-memory audit logger for testing and development."""
    
    def __init__(self, max_events: int = 10000):
        """
        Initialize in-memory audit logger.
        
        Args:
            max_events: Maximum number of events to keep in memory
        """
        self.max_events = max_events
        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()
    
    def log_event(self, event: AuditEvent):
        """Log event to memory."""
        with self._lock:
            self._events.append(event)
            
            # Keep only recent events
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]
    
    def query_events(self, 
                    event_type: Optional[AuditEventType] = None,
                    user_id: Optional[str] = None,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None) -> List[AuditEvent]:
        """Query events from memory."""
        with self._lock:
            filtered_events = []
            
            for event in self._events:
                # Apply filters
                if event_type and event.event_type != event_type:
                    continue
                
                if user_id and event.user_id != user_id:
                    continue
                
                if start_time and event.timestamp < start_time:
                    continue
                
                if end_time and event.timestamp > end_time:
                    continue
                
                filtered_events.append(event)
            
            return filtered_events
    
    def get_stats(self) -> Dict[str, Any]:
        """Get in-memory audit logger statistics."""
        with self._lock:
            event_types = {}
            for event in self._events:
                event_types[event.event_type.value] = event_types.get(event.event_type.value, 0) + 1
            
            return {
                "total_events": len(self._events),
                "max_events": self.max_events,
                "events_by_type": event_types,
                "oldest_event": self._events[0].timestamp.isoformat() if self._events else None,
                "newest_event": self._events[-1].timestamp.isoformat() if self._events else None
            }
    
    def clear_events(self):
        """Clear all events from memory."""
        with self._lock:
            self._events.clear()


class SecurityAuditLogger:
    """
    High-level security audit logging system.
    
    This class provides a convenient interface for logging security events
    and integrates with the security enforcement system.
    """
    
    _instance: Optional['SecurityAuditLogger'] = None
    
    def __init__(self, logger: Optional[AuditLogger] = None):
        """
        Initialize security audit logger.
        
        Args:
            logger: Underlying audit logger implementation
        """
        self.logger = logger or FileAuditLogger()
        self._enabled = True
        self._system_logger = logging.getLogger(__name__)
        
        # Set as global instance if none exists
        if SecurityAuditLogger._instance is None:
            SecurityAuditLogger._instance = self
    
    @classmethod
    def get_instance(cls) -> 'SecurityAuditLogger':
        """Get the global security audit logger instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def log_authentication(self, username: str, success: bool, 
                         details: Optional[Dict[str, Any]] = None):
        """Log authentication event."""
        if not self._enabled:
            return
        
        event_type = AuditEventType.USER_LOGIN if success else AuditEventType.AUTHENTICATION_FAILED
        outcome = "success" if success else "failure"
        
        event = AuditEvent(
            event_type=event_type,
            username=username,
            action="authenticate",
            outcome=outcome,
            details=details or {}
        )
        
        self.logger.log_event(event)
    
    def log_permission_check(self, permission: str, granted: bool,
                           resource: Optional[str] = None,
                           details: Optional[Dict[str, Any]] = None):
        """Log permission check event."""
        if not self._enabled:
            return
        
        event_type = AuditEventType.PERMISSION_GRANTED if granted else AuditEventType.PERMISSION_DENIED
        outcome = "granted" if granted else "denied"
        
        event = AuditEvent(
            event_type=event_type,
            resource=resource,
            action=f"check_permission:{permission}",
            outcome=outcome,
            details=details or {}
        )
        
        self.logger.log_event(event)
    
    def log_user_management(self, action: str, target_username: str,
                          success: bool = True,
                          details: Optional[Dict[str, Any]] = None):
        """Log user management event."""
        if not self._enabled:
            return
        
        event_type_map = {
            "create": AuditEventType.USER_CREATED,
            "delete": AuditEventType.USER_DELETED,
            "activate": AuditEventType.USER_ACTIVATED,
            "deactivate": AuditEventType.USER_DEACTIVATED,
            "role_change": AuditEventType.USER_ROLE_CHANGED
        }
        
        event_type = event_type_map.get(action, AuditEventType.USER_CREATED)
        outcome = "success" if success else "failure"
        
        event = AuditEvent(
            event_type=event_type,
            resource=f"user:{target_username}",
            action=action,
            outcome=outcome,
            details=details or {}
        )
        
        self.logger.log_event(event)
    
    def log_system_operation(self, operation: str, resource: str,
                           success: bool = True,
                           details: Optional[Dict[str, Any]] = None):
        """Log system operation event."""
        if not self._enabled:
            return
        
        # Map operations to event types
        operation_map = {
            "plugin_install": AuditEventType.PLUGIN_INSTALLED,
            "plugin_remove": AuditEventType.PLUGIN_REMOVED,
            "hardware_access": AuditEventType.HARDWARE_ACCESS,
            "simulation_run": AuditEventType.SIMULATION_RUN,
            "circuit_create": AuditEventType.CIRCUIT_CREATED,
            "circuit_delete": AuditEventType.CIRCUIT_DELETED,
            "backup_create": AuditEventType.BACKUP_CREATED,
            "data_export": AuditEventType.DATA_EXPORTED
        }
        
        event_type = operation_map.get(operation, AuditEventType.SYSTEM_CONFIG_CHANGED)
        outcome = "success" if success else "failure"
        
        event = AuditEvent(
            event_type=event_type,
            resource=resource,
            action=operation,
            outcome=outcome,
            details=details or {}
        )
        
        self.logger.log_event(event)
    
    def log_security_violation(self, violation_type: str, 
                             details: Optional[Dict[str, Any]] = None):
        """Log security violation event."""
        if not self._enabled:
            return
        
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_VIOLATION,
            action=violation_type,
            outcome="violation",
            details=details or {}
        )
        
        self.logger.log_event(event)
        
        # Also log to system logger for immediate attention
        self._system_logger.warning(f"Security violation: {violation_type}")
    
    def query_events(self, **kwargs) -> List[AuditEvent]:
        """Query audit events."""
        return self.logger.query_events(**kwargs)
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for the last N hours."""
        from datetime import timedelta
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        events = self.query_events(start_time=start_time, end_time=end_time)
        
        summary = {
            "period_hours": hours,
            "total_events": len(events),
            "authentication_attempts": 0,
            "failed_authentications": 0,
            "permission_denials": 0,
            "security_violations": 0,
            "user_management_events": 0,
            "system_operations": 0,
            "unique_users": set(),
            "events_by_type": {}
        }
        
        for event in events:
            event_type = event.event_type.value
            summary["events_by_type"][event_type] = summary["events_by_type"].get(event_type, 0) + 1
            
            if event.username:
                summary["unique_users"].add(event.username)
            
            if event.event_type in [AuditEventType.USER_LOGIN, AuditEventType.AUTHENTICATION_FAILED]:
                summary["authentication_attempts"] += 1
                if event.event_type == AuditEventType.AUTHENTICATION_FAILED:
                    summary["failed_authentications"] += 1
            
            elif event.event_type == AuditEventType.PERMISSION_DENIED:
                summary["permission_denials"] += 1
            
            elif event.event_type == AuditEventType.SECURITY_VIOLATION:
                summary["security_violations"] += 1
            
            elif event.event_type in [AuditEventType.USER_CREATED, AuditEventType.USER_DELETED, 
                                    AuditEventType.USER_ROLE_CHANGED, AuditEventType.USER_ACTIVATED,
                                    AuditEventType.USER_DEACTIVATED]:
                summary["user_management_events"] += 1
            
            elif event.event_type in [AuditEventType.PLUGIN_INSTALLED, AuditEventType.PLUGIN_REMOVED,
                                    AuditEventType.HARDWARE_ACCESS, AuditEventType.SIMULATION_RUN]:
                summary["system_operations"] += 1
        
        summary["unique_users"] = len(summary["unique_users"])
        return summary
    
    def enable(self):
        """Enable audit logging."""
        self._enabled = True
        self._system_logger.info("Security audit logging enabled")
    
    def disable(self):
        """Disable audit logging."""
        self._enabled = False
        self._system_logger.info("Security audit logging disabled")
    
    def is_enabled(self) -> bool:
        """Check if audit logging is enabled."""
        return self._enabled 