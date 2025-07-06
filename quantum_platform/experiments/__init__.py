"""
Quantum Experiments and Results Database

This module provides comprehensive experiment tracking and results database
functionality for the quantum platform, including:

- Persistent experiment storage with metadata
- Circuit version tracking and management
- Result aggregation and analysis
- Performance metrics and comparisons
- Session continuity and experiment history

The system is designed to be commercial-grade with IBM-level quality standards.
"""

from .database import ExperimentDatabase
from .models import (
    Experiment, Circuit, ExperimentResult, 
    ParameterSet, ExecutionContext, ExperimentMetrics
)
from .manager import ExperimentManager
from .analyzer import ExperimentAnalyzer

__all__ = [
    'ExperimentDatabase',
    'Experiment',
    'Circuit', 
    'ExperimentResult',
    'ParameterSet',
    'ExecutionContext',
    'ExperimentMetrics',
    'ExperimentManager',
    'ExperimentAnalyzer'
] 