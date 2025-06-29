"""
Real-Time Execution Monitoring Module

This module provides comprehensive real-time monitoring of quantum program executions,
including simulation progress tracking, hardware job status monitoring, and a
dashboard interface for users to monitor and control their quantum computations.
"""

from quantum_platform.execution.job_manager import (
    JobManager, ExecutionJob, JobStatus, JobType
)
from quantum_platform.execution.progress_tracker import (
    ProgressTracker, SimulationProgress, ProgressCallback
)
from quantum_platform.execution.status_monitor import (
    StatusMonitor, HardwareJobMonitor, HardwareJobInfo, StatusUpdate
)
from quantum_platform.execution.dashboard import (
    ExecutionDashboard, DashboardAPI, get_dashboard
)

__all__ = [
    # Job management
    'JobManager', 'ExecutionJob', 'JobStatus', 'JobType',
    
    # Progress tracking
    'ProgressTracker', 'SimulationProgress', 'ProgressCallback',
    
    # Status monitoring
    'StatusMonitor', 'HardwareJobMonitor', 'HardwareJobInfo', 'StatusUpdate',
    
    # Dashboard interface
    'ExecutionDashboard', 'DashboardAPI', 'get_dashboard'
] 