"""
Error Reporting System

This module provides comprehensive error reporting capabilities,
including error collection, formatting, and submission mechanisms.
"""

import json
import platform
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading
import traceback

from quantum_platform.observability.logging import get_logger


@dataclass
class SystemInfo:
    """System information for error reports."""
    platform: str = ""
    python_version: str = ""
    platform_version: str = ""
    architecture: str = ""
    processor: str = ""
    memory_gb: float = 0.0
    disk_space_gb: float = 0.0
    
    @classmethod
    def collect(cls) -> 'SystemInfo':
        """Collect current system information."""
        import psutil
        
        return cls(
            platform=platform.system(),
            python_version=sys.version,
            platform_version=platform.release(),
            architecture=platform.machine(),
            processor=platform.processor(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            disk_space_gb=psutil.disk_usage('/').total / (1024**3)
        )


@dataclass
class ErrorContext:
    """Context information for error reports."""
    component: str = ""
    operation: str = ""
    user_action: str = ""
    circuit_info: Dict[str, Any] = field(default_factory=dict)
    session_info: Dict[str, Any] = field(default_factory=dict)
    recent_operations: List[str] = field(default_factory=list)


@dataclass
class ErrorReport:
    """Comprehensive error report structure."""
    report_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    user_message: str
    error_code: str
    severity: str
    
    # Context
    context: ErrorContext
    system_info: SystemInfo
    
    # Technical details
    traceback: str
    log_snippet: str
    
    # User information
    user_description: str = ""
    reproduction_steps: List[str] = field(default_factory=list)
    user_email: str = ""
    
    # Platform state
    platform_version: str = "1.0.0"
    active_plugins: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error report to dictionary."""
        return {
            'report_id': self.report_id,
            'timestamp': self.timestamp.isoformat(),
            'error_type': self.error_type,
            'error_message': self.error_message,
            'user_message': self.user_message,
            'error_code': self.error_code,
            'severity': self.severity,
            'context': {
                'component': self.context.component,
                'operation': self.context.operation,
                'user_action': self.context.user_action,
                'circuit_info': self.context.circuit_info,
                'session_info': self.context.session_info,
                'recent_operations': self.context.recent_operations
            },
            'system_info': {
                'platform': self.system_info.platform,
                'python_version': self.system_info.python_version,
                'platform_version': self.system_info.platform_version,
                'architecture': self.system_info.architecture,
                'processor': self.system_info.processor,
                'memory_gb': self.system_info.memory_gb,
                'disk_space_gb': self.system_info.disk_space_gb
            },
            'technical_details': {
                'traceback': self.traceback,
                'log_snippet': self.log_snippet,
                'platform_version': self.platform_version,
                'active_plugins': self.active_plugins,
                'configuration': self.configuration
            },
            'user_info': {
                'description': self.user_description,
                'reproduction_steps': self.reproduction_steps,
                'email': self.user_email
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert error report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_markdown(self) -> str:
        """Convert error report to Markdown format for GitHub issues."""
        md = f"""# Error Report: {self.error_code}

## Summary
**Error Type:** {self.error_type}
**Severity:** {self.severity}
**Timestamp:** {self.timestamp.isoformat()}

## User Description
{self.user_description or "No description provided"}

## Error Details
**User Message:** {self.user_message}
**Technical Message:** {self.error_message}

## Context
- **Component:** {self.context.component}
- **Operation:** {self.context.operation}
- **User Action:** {self.context.user_action}

## System Information
- **Platform:** {self.system_info.platform} {self.system_info.platform_version}
- **Python:** {self.system_info.python_version}
- **Architecture:** {self.system_info.architecture}
- **Memory:** {self.system_info.memory_gb:.1f} GB

## Circuit Information
```json
{json.dumps(self.context.circuit_info, indent=2)}
```

## Recent Operations
{chr(10).join(f"- {op}" for op in self.context.recent_operations)}

## Reproduction Steps
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(self.reproduction_steps))}

## Technical Details
<details>
<summary>Traceback</summary>

```
{self.traceback}
```
</details>

<details>
<summary>Log Snippet</summary>

```
{self.log_snippet}
```
</details>

## Active Plugins
{chr(10).join(f"- {plugin}" for plugin in self.active_plugins)}

---
*Report ID: {self.report_id}*
*Platform Version: {self.platform_version}*
"""
        return md


class ErrorReporter:
    """
    Comprehensive error reporting system.
    
    Collects, formats, and manages error reports with user consent.
    """
    
    def __init__(self, reports_dir: Optional[Path] = None):
        self.logger = get_logger(__name__)
        self.reports_dir = reports_dir or Path("error_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Error collection
        self._error_history: List[ErrorReport] = []
        self._max_history = 50
        self._lock = threading.Lock()
        
        # System info cache
        self._system_info: Optional[SystemInfo] = None
        self._system_info_cache_time = 0
        self._system_info_cache_duration = 300  # 5 minutes
        
        # User preferences
        self.auto_collect = True
        self.include_system_info = True
        self.include_logs = True
        self.max_log_lines = 100
        
        # Submission handlers
        self._submission_handlers: List[Callable[[ErrorReport], bool]] = []
        
        self.logger.info("Error reporter initialized")
    
    def get_system_info(self) -> SystemInfo:
        """Get cached system information."""
        current_time = time.time()
        
        if (self._system_info is None or 
            current_time - self._system_info_cache_time > self._system_info_cache_duration):
            try:
                self._system_info = SystemInfo.collect()
                self._system_info_cache_time = current_time
            except Exception as e:
                self.logger.warning(f"Failed to collect system info: {e}")
                self._system_info = SystemInfo()
        
        return self._system_info
    
    def add_submission_handler(self, handler: Callable[[ErrorReport], bool]):
        """Add a handler for error report submission."""
        self._submission_handlers.append(handler)
    
    def collect_error(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        user_action: str = "",
        include_logs: bool = True
    ) -> ErrorReport:
        """
        Collect comprehensive error information.
        
        Args:
            exception: The exception that occurred
            context: Additional context information
            user_action: Description of what the user was doing
            include_logs: Whether to include log snippet
            
        Returns:
            Complete error report
        """
        try:
            # Generate unique report ID
            report_id = f"QP-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
            
            # Get system information
            system_info = self.get_system_info() if self.include_system_info else SystemInfo()
            
            # Create context if not provided
            if context is None:
                context = ErrorContext()
            
            # Update context with user action
            if user_action:
                context.user_action = user_action
            
            # Get log snippet
            log_snippet = ""
            if include_logs and self.include_logs:
                log_snippet = self._get_recent_logs()
            
            # Extract exception details
            error_type = exception.__class__.__name__
            error_message = str(exception)
            traceback_str = traceback.format_exc()
            
            # Get user-friendly message
            user_message = getattr(exception, 'user_message', error_message)
            error_code = getattr(exception, 'error_code', f"QP{hash(error_message) % 10000:04d}")
            severity = getattr(exception, 'severity', 'medium')
            if hasattr(severity, 'value'):
                severity = severity.value
            
            # Create error report
            report = ErrorReport(
                report_id=report_id,
                timestamp=datetime.now(),
                error_type=error_type,
                error_message=error_message,
                user_message=user_message,
                error_code=error_code,
                severity=severity,
                context=context,
                system_info=system_info,
                traceback=traceback_str,
                log_snippet=log_snippet
            )
            
            # Add to history
            with self._lock:
                self._error_history.append(report)
                if len(self._error_history) > self._max_history:
                    self._error_history.pop(0)
            
            # Save to file
            self._save_report(report)
            
            self.logger.info(f"Error report collected: {report_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to collect error report: {e}")
            # Create minimal report
            return ErrorReport(
                report_id=f"QP-ERROR-{int(time.time())}",
                timestamp=datetime.now(),
                error_type="ReportingError",
                error_message=f"Failed to collect error report: {e}",
                user_message="An error occurred while reporting another error",
                error_code="QP9999",
                severity="high",
                context=ErrorContext(),
                system_info=SystemInfo(),
                traceback=traceback.format_exc(),
                log_snippet=""
            )
    
    def _get_recent_logs(self) -> str:
        """Get recent log entries."""
        try:
            # Try to read from log file
            log_file = Path("logs/quantum_platform.log")
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Get last N lines
                    recent_lines = lines[-self.max_log_lines:]
                    return ''.join(recent_lines)
        except Exception as e:
            self.logger.warning(f"Failed to read log file: {e}")
        
        return "Unable to retrieve log snippet"
    
    def _save_report(self, report: ErrorReport):
        """Save error report to file."""
        try:
            report_file = self.reports_dir / f"{report.report_id}.json"
            with open(report_file, 'w') as f:
                f.write(report.to_json())
            
            # Also save markdown version
            md_file = self.reports_dir / f"{report.report_id}.md"
            with open(md_file, 'w') as f:
                f.write(report.to_markdown())
                
        except Exception as e:
            self.logger.error(f"Failed to save error report: {e}")
    
    def get_error_history(self) -> List[ErrorReport]:
        """Get error history."""
        with self._lock:
            return self._error_history.copy()
    
    def submit_report(
        self,
        report: ErrorReport,
        user_description: str = "",
        reproduction_steps: List[str] = None,
        user_email: str = ""
    ) -> bool:
        """
        Submit error report with user information.
        
        Args:
            report: Error report to submit
            user_description: User's description of the issue
            reproduction_steps: Steps to reproduce the error
            user_email: User's email (optional)
            
        Returns:
            True if submission successful
        """
        try:
            # Update report with user information
            report.user_description = user_description
            report.reproduction_steps = reproduction_steps or []
            report.user_email = user_email
            
            # Try all submission handlers
            success = False
            for handler in self._submission_handlers:
                try:
                    if handler(report):
                        success = True
                        break
                except Exception as e:
                    self.logger.warning(f"Submission handler failed: {e}")
            
            if success:
                self.logger.info(f"Error report submitted: {report.report_id}")
            else:
                self.logger.warning(f"Failed to submit error report: {report.report_id}")
                # Save with submission attempt info
                self._save_failed_submission(report)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error during report submission: {e}")
            return False
    
    def _save_failed_submission(self, report: ErrorReport):
        """Save report that failed to submit."""
        try:
            failed_dir = self.reports_dir / "failed_submissions"
            failed_dir.mkdir(exist_ok=True)
            
            report_file = failed_dir / f"{report.report_id}.json"
            with open(report_file, 'w') as f:
                f.write(report.to_json())
                
        except Exception as e:
            self.logger.error(f"Failed to save failed submission: {e}")
    
    def create_clipboard_report(self, report: ErrorReport) -> str:
        """Create a clipboard-friendly version of the error report."""
        return f"""Quantum Platform Error Report
Report ID: {report.report_id}
Timestamp: {report.timestamp}

Error: {report.user_message}
Error Code: {report.error_code}

Component: {report.context.component}
Operation: {report.context.operation}

System: {report.system_info.platform} {report.system_info.platform_version}
Python: {report.system_info.python_version}

For full details, see: {report.report_id}.md
"""


# Global error reporter instance
_error_reporter: Optional[ErrorReporter] = None
_reporter_lock = threading.Lock()


def get_error_reporter() -> ErrorReporter:
    """Get the global error reporter instance."""
    global _error_reporter
    
    if _error_reporter is None:
        with _reporter_lock:
            if _error_reporter is None:
                _error_reporter = ErrorReporter()
    
    return _error_reporter


def report_error(
    exception: Exception,
    context: Optional[ErrorContext] = None,
    user_action: str = "",
    include_logs: bool = True
) -> ErrorReport:
    """Convenience function to report an error."""
    reporter = get_error_reporter()
    return reporter.collect_error(exception, context, user_action, include_logs)


# Default submission handlers
def github_issue_handler(report: ErrorReport) -> bool:
    """Create a GitHub issue template (requires manual submission)."""
    try:
        github_dir = Path("error_reports/github_issues")
        github_dir.mkdir(exist_ok=True)
        
        issue_file = github_dir / f"issue_{report.report_id}.md"
        with open(issue_file, 'w') as f:
            f.write(report.to_markdown())
        
        print(f"GitHub issue template created: {issue_file}")
        print("Please submit this as a GitHub issue manually.")
        return True
        
    except Exception:
        return False


def email_handler(report: ErrorReport) -> bool:
    """Create an email template (requires manual sending)."""
    try:
        email_dir = Path("error_reports/email_templates")
        email_dir.mkdir(exist_ok=True)
        
        email_file = email_dir / f"email_{report.report_id}.txt"
        with open(email_file, 'w') as f:
            f.write(f"Subject: Quantum Platform Error Report - {report.error_code}\n\n")
            f.write(report.to_markdown())
        
        print(f"Email template created: {email_file}")
        print("Please send this email to the development team.")
        return True
        
    except Exception:
        return False


# Register default handlers
def _register_default_handlers():
    """Register default submission handlers."""
    reporter = get_error_reporter()
    reporter.add_submission_handler(github_issue_handler)
    reporter.add_submission_handler(email_handler)


# Initialize on module import
_register_default_handlers() 