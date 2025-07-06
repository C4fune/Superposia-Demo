"""
Experiment Database Models

This module defines the data models for the quantum experiment tracking system,
including experiments, circuits, results, and associated metadata.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, 
    ForeignKey, JSON, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.types import TypeDecorator, VARCHAR


Base = declarative_base()


class JSONType(TypeDecorator):
    """Custom JSON type for SQLAlchemy that handles serialization."""
    
    impl = Text
    
    def process_bind_param(self, value, dialect):
        """Convert Python dict to JSON string."""
        if value is not None:
            return json.dumps(value, default=str)
        return value
    
    def process_result_value(self, value, dialect):
        """Convert JSON string to Python dict."""
        if value is not None:
            return json.loads(value)
        return value


class ExperimentStatus(Enum):
    """Status of an experiment."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    QUEUED = "queued"


class ExecutionBackend(Enum):
    """Types of execution backends."""
    LOCAL_SIMULATOR = "local_simulator"
    NOISY_SIMULATOR = "noisy_simulator"
    IBM_QUANTUM = "ibm_quantum"
    AWS_BRAKET = "aws_braket"
    GOOGLE_QUANTUM = "google_quantum"
    IONQ = "ionq"
    RIGETTI = "rigetti"
    CUSTOM = "custom"


class ExperimentType(Enum):
    """Types of experiments."""
    SINGLE_SHOT = "single_shot"
    PARAMETER_SWEEP = "parameter_sweep"
    OPTIMIZATION = "optimization"
    BENCHMARKING = "benchmarking"
    ALGORITHM_STUDY = "algorithm_study"
    NOISE_ANALYSIS = "noise_analysis"
    COMPARISON = "comparison"
    CUSTOM = "custom"


class Circuit(Base):
    """Circuit definitions table."""
    
    __tablename__ = 'circuits'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Circuit definition
    qasm_code = Column(Text)
    circuit_json = Column(JSONType)
    num_qubits = Column(Integer, nullable=False)
    num_classical_bits = Column(Integer, default=0)
    depth = Column(Integer, default=0)
    
    # Circuit metadata
    gate_count = Column(JSONType)  # {"H": 5, "CNOT": 3, ...}
    two_qubit_gate_count = Column(Integer, default=0)
    parameters = Column(JSONType)  # Parameter definitions
    
    # Versioning
    version = Column(String(50), default="1.0")
    parent_circuit_id = Column(String(36), ForeignKey('circuits.id'))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Hash for uniqueness checking
    content_hash = Column(String(64), unique=True)
    
    # Relationships
    parent_circuit = relationship("Circuit", remote_side=[id])
    child_circuits = relationship("Circuit", remote_side=[parent_circuit_id], overlaps="parent_circuit")
    experiments = relationship("Experiment", back_populates="circuit")
    
    # Indexes
    __table_args__ = (
        Index('idx_circuit_name', 'name'),
        Index('idx_circuit_hash', 'content_hash'),
        Index('idx_circuit_created', 'created_at'),
    )
    
    def __init__(self, **kwargs):
        """Initialize circuit with automatic hash generation."""
        super().__init__(**kwargs)
        if self.qasm_code and not self.content_hash:
            self.content_hash = self._generate_content_hash()
    
    def _generate_content_hash(self) -> str:
        """Generate hash of circuit content for uniqueness."""
        content = f"{self.qasm_code}{self.num_qubits}{self.parameters}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert circuit to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'qasm_code': self.qasm_code,
            'circuit_json': self.circuit_json,
            'num_qubits': self.num_qubits,
            'num_classical_bits': self.num_classical_bits,
            'depth': self.depth,
            'gate_count': self.gate_count,
            'two_qubit_gate_count': self.two_qubit_gate_count,
            'parameters': self.parameters,
            'version': self.version,
            'parent_circuit_id': self.parent_circuit_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'content_hash': self.content_hash
        }


class Experiment(Base):
    """Experiments table."""
    
    __tablename__ = 'experiments'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Experiment classification
    experiment_type = Column(String(50), nullable=False)
    tags = Column(JSONType)  # List of tags
    
    # Circuit reference
    circuit_id = Column(String(36), ForeignKey('circuits.id'), nullable=False)
    
    # Execution parameters
    backend = Column(String(100), nullable=False)
    provider = Column(String(100))
    device_name = Column(String(100))
    shots = Column(Integer, default=1000)
    
    # Parameter sweep information
    parameter_sweep = Column(JSONType)  # Parameter sweep configuration
    
    # Execution status
    status = Column(String(50), default=ExperimentStatus.CREATED.value)
    
    # Timing information
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # User information
    created_by = Column(String(100))
    session_id = Column(String(100))
    
    # Experiment metadata (renamed to avoid SQLAlchemy reserved word)
    experiment_metadata = Column(JSONType)
    notes = Column(Text)
    
    # Result summary
    total_runs = Column(Integer, default=0)
    successful_runs = Column(Integer, default=0)
    failed_runs = Column(Integer, default=0)
    
    # Performance summary
    avg_execution_time = Column(Float)
    total_execution_time = Column(Float)
    
    # Result metrics
    best_result_id = Column(String(36))
    metrics_summary = Column(JSONType)
    
    # Relationships
    circuit = relationship("Circuit", back_populates="experiments")
    results = relationship("ExperimentResult", back_populates="experiment", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_experiment_name', 'name'),
        Index('idx_experiment_status', 'status'),
        Index('idx_experiment_backend', 'backend'),
        Index('idx_experiment_type', 'experiment_type'),
        Index('idx_experiment_created', 'created_at'),
        Index('idx_experiment_circuit', 'circuit_id'),
        Index('idx_experiment_user', 'created_by'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'experiment_type': self.experiment_type,
            'tags': self.tags,
            'circuit_id': self.circuit_id,
            'backend': self.backend,
            'provider': self.provider,
            'device_name': self.device_name,
            'shots': self.shots,
            'parameter_sweep': self.parameter_sweep,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_by': self.created_by,
            'session_id': self.session_id,
            'metadata': self.experiment_metadata,  # Use original key name for external API
            'notes': self.notes,
            'total_runs': self.total_runs,
            'successful_runs': self.successful_runs,
            'failed_runs': self.failed_runs,
            'avg_execution_time': self.avg_execution_time,
            'total_execution_time': self.total_execution_time,
            'best_result_id': self.best_result_id,
            'metrics_summary': self.metrics_summary
        }


class ExperimentResult(Base):
    """Individual experiment results table."""
    
    __tablename__ = 'experiment_results'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Experiment reference
    experiment_id = Column(String(36), ForeignKey('experiments.id'), nullable=False)
    
    # Run information
    run_number = Column(Integer, nullable=False)
    run_id = Column(String(100))  # Provider-specific run ID
    
    # Execution parameters
    parameter_values = Column(JSONType)  # Actual parameter values used
    shots = Column(Integer, default=1000)
    
    # Results
    raw_counts = Column(JSONType)  # Raw measurement counts
    normalized_counts = Column(JSONType)  # Normalized probabilities
    
    # Metrics
    fidelity = Column(Float)
    success_probability = Column(Float)
    expectation_value = Column(Float)
    variance = Column(Float)
    custom_metrics = Column(JSONType)
    
    # Execution information
    execution_time = Column(Float)  # milliseconds
    queue_time = Column(Float)  # milliseconds
    
    # Status and error information
    status = Column(String(50), default="completed")
    error_message = Column(Text)
    error_code = Column(String(50))
    
    # Timing
    submitted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Provenance
    backend_version = Column(String(100))
    compiler_version = Column(String(100))
    
    # Calibration data
    calibration_data = Column(JSONType)
    
    # Raw result data (for debugging)
    raw_result_data = Column(JSONType)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="results")
    
    # Indexes
    __table_args__ = (
        Index('idx_result_experiment', 'experiment_id'),
        Index('idx_result_run', 'run_number'),
        Index('idx_result_status', 'status'),
        Index('idx_result_submitted', 'submitted_at'),
        UniqueConstraint('experiment_id', 'run_number', name='_experiment_run_uc'),
    )
    
    def __init__(self, **kwargs):
        """Initialize result with automatic calculations."""
        super().__init__(**kwargs)
        if self.raw_counts and not self.normalized_counts:
            self.normalized_counts = self._calculate_normalized_counts()
    
    def _calculate_normalized_counts(self) -> Dict[str, float]:
        """Calculate normalized counts from raw counts."""
        if not self.raw_counts:
            return {}
        
        total = sum(self.raw_counts.values())
        if total == 0:
            return {}
        
        return {state: count / total for state, count in self.raw_counts.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'run_number': self.run_number,
            'run_id': self.run_id,
            'parameter_values': self.parameter_values,
            'shots': self.shots,
            'raw_counts': self.raw_counts,
            'normalized_counts': self.normalized_counts,
            'fidelity': self.fidelity,
            'success_probability': self.success_probability,
            'expectation_value': self.expectation_value,
            'variance': self.variance,
            'custom_metrics': self.custom_metrics,
            'execution_time': self.execution_time,
            'queue_time': self.queue_time,
            'status': self.status,
            'error_message': self.error_message,
            'error_code': self.error_code,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'backend_version': self.backend_version,
            'compiler_version': self.compiler_version,
            'calibration_data': self.calibration_data,
            'raw_result_data': self.raw_result_data
        }


class ParameterSet(Base):
    """Parameter sets for experiments."""
    
    __tablename__ = 'parameter_sets'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Parameter definitions
    parameters = Column(JSONType, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100))
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    
    # Indexes
    __table_args__ = (
        Index('idx_parameterset_name', 'name'),
        Index('idx_parameterset_created', 'created_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter set to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'created_by': self.created_by,
            'usage_count': self.usage_count
        }


class ExecutionContext(Base):
    """Execution context configurations."""
    
    __tablename__ = 'execution_contexts'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Context configuration
    backend = Column(String(100), nullable=False)
    provider = Column(String(100))
    device_name = Column(String(100))
    shots = Column(Integer, default=1000)
    
    # Execution settings
    max_execution_time = Column(Integer)  # seconds
    retry_count = Column(Integer, default=0)
    
    # Optimization settings
    optimization_level = Column(Integer, default=1)
    
    # Noise settings
    noise_model = Column(JSONType)
    
    # Other settings
    settings = Column(JSONType)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100))
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    
    # Indexes
    __table_args__ = (
        Index('idx_execcontext_name', 'name'),
        Index('idx_execcontext_backend', 'backend'),
        Index('idx_execcontext_created', 'created_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution context to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'backend': self.backend,
            'provider': self.provider,
            'device_name': self.device_name,
            'shots': self.shots,
            'max_execution_time': self.max_execution_time,
            'retry_count': self.retry_count,
            'optimization_level': self.optimization_level,
            'noise_model': self.noise_model,
            'settings': self.settings,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'created_by': self.created_by,
            'usage_count': self.usage_count
        }


class ExperimentMetrics(Base):
    """Aggregated experiment metrics."""
    
    __tablename__ = 'experiment_metrics'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Experiment reference
    experiment_id = Column(String(36), ForeignKey('experiments.id'), nullable=False)
    
    # Time period
    calculated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Success metrics
    success_rate = Column(Float)
    avg_fidelity = Column(Float)
    std_fidelity = Column(Float)
    
    # Performance metrics
    avg_execution_time = Column(Float)
    std_execution_time = Column(Float)
    avg_queue_time = Column(Float)
    std_queue_time = Column(Float)
    
    # Result quality metrics
    avg_expectation_value = Column(Float)
    std_expectation_value = Column(Float)
    
    # Statistical metrics
    result_entropy = Column(Float)
    result_variance = Column(Float)
    
    # Custom metrics
    custom_metrics = Column(JSONType)
    
    # Indexes
    __table_args__ = (
        Index('idx_metrics_experiment', 'experiment_id'),
        Index('idx_metrics_calculated', 'calculated_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'calculated_at': self.calculated_at.isoformat() if self.calculated_at else None,
            'success_rate': self.success_rate,
            'avg_fidelity': self.avg_fidelity,
            'std_fidelity': self.std_fidelity,
            'avg_execution_time': self.avg_execution_time,
            'std_execution_time': self.std_execution_time,
            'avg_queue_time': self.avg_queue_time,
            'std_queue_time': self.std_queue_time,
            'avg_expectation_value': self.avg_expectation_value,
            'std_expectation_value': self.std_expectation_value,
            'result_entropy': self.result_entropy,
            'result_variance': self.result_variance,
            'custom_metrics': self.custom_metrics
        }


@dataclass
class ExperimentSummary:
    """Summary of an experiment for quick display."""
    
    experiment_id: str
    name: str
    experiment_type: str
    status: str
    backend: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    avg_execution_time: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]
    success_rate: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.total_runs > 0:
            self.success_rate = self.successful_runs / self.total_runs
        else:
            self.success_rate = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Result of comparing two experiments."""
    
    experiment1_id: str
    experiment2_id: str
    comparison_type: str
    
    # Statistical comparison
    fidelity_difference: Optional[float] = None
    execution_time_difference: Optional[float] = None
    success_rate_difference: Optional[float] = None
    
    # Significance testing
    p_value: Optional[float] = None
    is_significant: bool = False
    
    # Detailed analysis
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self) 