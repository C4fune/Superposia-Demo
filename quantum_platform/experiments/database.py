"""
Experiment Database Layer

This module provides the core database functionality for the quantum experiment
tracking system, including database initialization, connection management,
and data access operations.
"""

import os
import sqlite3
import threading
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, func, and_, or_, desc, asc, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

from quantum_platform.observability.logging import get_logger
from .models import (
    Base, Circuit, Experiment, ExperimentResult, ParameterSet, 
    ExecutionContext, ExperimentMetrics, ExperimentSummary, ComparisonResult,
    ExperimentStatus, ExecutionBackend, ExperimentType
)


class ExperimentDatabase:
    """
    Core database manager for quantum experiment tracking.
    
    This class provides a comprehensive interface for managing experiment data,
    including database initialization, connection pooling, and data operations.
    """
    
    def __init__(self, database_path: str = "quantum_experiments.db", 
                 echo: bool = False, pool_size: int = 20):
        """
        Initialize the experiment database.
        
        Args:
            database_path: Path to the SQLite database file
            echo: Whether to echo SQL statements (for debugging)
            pool_size: Connection pool size (ignored for SQLite)
        """
        self.database_path = database_path
        self.echo = echo
        self.pool_size = pool_size
        self.logger = get_logger("ExperimentDatabase")
        
        # Create database directory if it doesn't exist
        db_dir = Path(database_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Create database engine (SQLite-specific configuration)
        self.engine = create_engine(
            f"sqlite:///{database_path}",
            echo=echo,
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 30
            }
        )
        
        # Configure SQLite for better performance
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=1000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
            cursor.close()
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Thread-local storage for sessions
        self._thread_local = threading.local()
        
        # Initialize database
        self._initialize_database()
        
        self.logger.info(f"Experiment database initialized at {database_path}")
    
    def _initialize_database(self):
        """Initialize database tables and indexes."""
        try:
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            
            # Create additional indexes for performance
            with self.engine.connect() as conn:
                # Add custom indexes that might not be in the model
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_results_timing ON experiment_results(submitted_at, completed_at)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_experiments_duration ON experiments(created_at, completed_at)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_circuits_usage ON circuits(created_at, name)"))
                conn.commit()
            
            self.logger.info("Database tables and indexes created successfully")
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_thread_session(self) -> Session:
        """Get a thread-local database session."""
        if not hasattr(self._thread_local, 'session'):
            self._thread_local.session = self.SessionLocal()
        return self._thread_local.session
    
    def close_thread_session(self):
        """Close the thread-local session."""
        if hasattr(self._thread_local, 'session'):
            self._thread_local.session.close()
            delattr(self._thread_local, 'session')
    
    # Circuit Operations
    def create_circuit(self, name: str, qasm_code: str, num_qubits: int, 
                      description: str = None, circuit_json: Dict = None,
                      parameters: Dict = None, version: str = "1.0",
                      parent_circuit_id: str = None) -> Circuit:
        """Create a new circuit in the database."""
        with self.get_session() as session:
            circuit = Circuit(
                name=name,
                description=description,
                qasm_code=qasm_code,
                circuit_json=circuit_json,
                num_qubits=num_qubits,
                parameters=parameters,
                version=version,
                parent_circuit_id=parent_circuit_id
            )
            
            # Check if circuit already exists (by content hash)
            existing = session.query(Circuit).filter_by(
                content_hash=circuit.content_hash
            ).first()
            
            if existing:
                self.logger.info(f"Circuit with same content already exists: {existing.id}")
                # Detach from session to prevent issues when session closes
                session.expunge(existing)
                return existing
            
            session.add(circuit)
            session.flush()
            
            self.logger.info(f"Created circuit: {circuit.id} ({circuit.name})")
            # Detach from session to prevent issues when session closes
            session.expunge(circuit)
            return circuit
    
    def get_circuit(self, circuit_id: str) -> Optional[Circuit]:
        """Get a circuit by ID."""
        with self.get_session() as session:
            circuit = session.query(Circuit).filter_by(id=circuit_id).first()
            if circuit:
                session.expunge(circuit)
            return circuit
    
    def get_circuits(self, limit: int = 100, offset: int = 0,
                    name_filter: str = None, created_after: datetime = None) -> List[Circuit]:
        """Get circuits with optional filtering."""
        with self.get_session() as session:
            query = session.query(Circuit)
            
            if name_filter:
                query = query.filter(Circuit.name.contains(name_filter))
            
            if created_after:
                query = query.filter(Circuit.created_at >= created_after)
            
            circuits = query.order_by(desc(Circuit.created_at)).offset(offset).limit(limit).all()
            # Detach all circuits from session
            for circuit in circuits:
                session.expunge(circuit)
            return circuits
    
    def update_circuit(self, circuit_id: str, updates: Dict[str, Any]) -> bool:
        """Update circuit properties."""
        with self.get_session() as session:
            circuit = session.query(Circuit).filter_by(id=circuit_id).first()
            if not circuit:
                return False
            
            for key, value in updates.items():
                if hasattr(circuit, key):
                    setattr(circuit, key, value)
            
            circuit.updated_at = datetime.now(timezone.utc)
            return True
    
    def delete_circuit(self, circuit_id: str) -> bool:
        """Delete a circuit (only if no experiments reference it)."""
        with self.get_session() as session:
            circuit = session.query(Circuit).filter_by(id=circuit_id).first()
            if not circuit:
                return False
            
            # Check if any experiments reference this circuit
            experiment_count = session.query(Experiment).filter_by(circuit_id=circuit_id).count()
            if experiment_count > 0:
                raise ValueError(f"Cannot delete circuit {circuit_id}: {experiment_count} experiments reference it")
            
            session.delete(circuit)
            return True
    
    # Experiment Operations
    def create_experiment(self, name: str, circuit_id: str, backend: str,
                         experiment_type: str = ExperimentType.SINGLE_SHOT.value,
                         description: str = None, shots: int = 1000,
                         provider: str = None, device_name: str = None,
                         parameter_sweep: Dict = None, tags: List[str] = None,
                         created_by: str = None, session_id: str = None,
                         metadata: Dict = None) -> Experiment:
        """Create a new experiment."""
        with self.get_session() as session:
            # Verify circuit exists
            circuit = session.query(Circuit).filter_by(id=circuit_id).first()
            if not circuit:
                raise ValueError(f"Circuit {circuit_id} not found")
            
            experiment = Experiment(
                name=name,
                description=description,
                experiment_type=experiment_type,
                circuit_id=circuit_id,
                backend=backend,
                provider=provider,
                device_name=device_name,
                shots=shots,
                parameter_sweep=parameter_sweep,
                tags=tags,
                created_by=created_by,
                session_id=session_id,
                experiment_metadata=metadata or {}
            )
            
            session.add(experiment)
            session.flush()
            
            self.logger.info(f"Created experiment: {experiment.id} ({experiment.name})")
            # Detach from session to prevent issues when session closes
            session.expunge(experiment)
            return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        with self.get_session() as session:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if experiment:
                session.expunge(experiment)
            return experiment
    
    def get_experiments(self, limit: int = 100, offset: int = 0,
                       status_filter: str = None, backend_filter: str = None,
                       created_by_filter: str = None, experiment_type_filter: str = None,
                       created_after: datetime = None, created_before: datetime = None,
                       tags_filter: List[str] = None) -> List[Experiment]:
        """Get experiments with optional filtering."""
        with self.get_session() as session:
            query = session.query(Experiment)
            
            if status_filter:
                query = query.filter(Experiment.status == status_filter)
            
            if backend_filter:
                query = query.filter(Experiment.backend == backend_filter)
            
            if created_by_filter:
                query = query.filter(Experiment.created_by == created_by_filter)
            
            if experiment_type_filter:
                query = query.filter(Experiment.experiment_type == experiment_type_filter)
            
            if created_after:
                query = query.filter(Experiment.created_at >= created_after)
            
            if created_before:
                query = query.filter(Experiment.created_at <= created_before)
            
            if tags_filter:
                for tag in tags_filter:
                    query = query.filter(Experiment.tags.contains([tag]))
            
            experiments = query.order_by(desc(Experiment.created_at)).offset(offset).limit(limit).all()
            # Detach all experiments from session
            for experiment in experiments:
                session.expunge(experiment)
            return experiments
    
    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> bool:
        """Update experiment properties."""
        with self.get_session() as session:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if not experiment:
                return False
            
            for key, value in updates.items():
                if hasattr(experiment, key):
                    setattr(experiment, key, value)
            
            return True
    
    def update_experiment_status(self, experiment_id: str, status: str,
                                started_at: datetime = None, completed_at: datetime = None) -> bool:
        """Update experiment status with timing information."""
        with self.get_session() as session:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if not experiment:
                return False
            
            experiment.status = status
            
            if started_at:
                experiment.started_at = started_at
            elif status == ExperimentStatus.RUNNING.value and not experiment.started_at:
                experiment.started_at = datetime.now(timezone.utc)
            
            if completed_at:
                experiment.completed_at = completed_at
            elif status in [ExperimentStatus.COMPLETED.value, ExperimentStatus.FAILED.value, 
                           ExperimentStatus.CANCELLED.value]:
                experiment.completed_at = datetime.now(timezone.utc)
            
            return True
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all its results."""
        with self.get_session() as session:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if not experiment:
                return False
            
            # Delete all results first (cascade should handle this, but being explicit)
            session.query(ExperimentResult).filter_by(experiment_id=experiment_id).delete()
            session.query(ExperimentMetrics).filter_by(experiment_id=experiment_id).delete()
            
            session.delete(experiment)
            
            self.logger.info(f"Deleted experiment: {experiment_id}")
            return True
    
    # Result Operations
    def create_result(self, experiment_id: str, run_number: int,
                     raw_counts: Dict[str, int], shots: int,
                     parameter_values: Dict = None, run_id: str = None,
                     execution_time: float = None, queue_time: float = None,
                     fidelity: float = None, success_probability: float = None,
                     expectation_value: float = None, variance: float = None,
                     custom_metrics: Dict = None, error_message: str = None,
                     error_code: str = None, calibration_data: Dict = None,
                     raw_result_data: Dict = None) -> ExperimentResult:
        """Create a new experiment result."""
        with self.get_session() as session:
            # Verify experiment exists
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            result = ExperimentResult(
                experiment_id=experiment_id,
                run_number=run_number,
                run_id=run_id or str(uuid.uuid4()),
                raw_counts=raw_counts,
                shots=shots,
                parameter_values=parameter_values or {},
                execution_time=execution_time,
                queue_time=queue_time,
                fidelity=fidelity,
                success_probability=success_probability,
                expectation_value=expectation_value,
                variance=variance,
                custom_metrics=custom_metrics or {},
                error_message=error_message,
                error_code=error_code,
                calibration_data=calibration_data or {},
                raw_result_data=raw_result_data or {}
            )
            
            session.add(result)
            session.flush()
            
            # Update experiment statistics
            self._update_experiment_stats(session, experiment_id)
            
            self.logger.info(f"Created result: {result.id} for experiment {experiment_id}")
            # Detach from session to prevent issues when session closes
            session.expunge(result)
            return result
    
    def get_results(self, experiment_id: str, limit: int = 100, offset: int = 0) -> List[ExperimentResult]:
        """Get results for an experiment."""
        with self.get_session() as session:
            results = session.query(ExperimentResult).filter_by(
                experiment_id=experiment_id
            ).order_by(ExperimentResult.run_number).offset(offset).limit(limit).all()
            # Detach all results from session
            for result in results:
                session.expunge(result)
            return results
    
    def get_result(self, result_id: str) -> Optional[ExperimentResult]:
        """Get a specific result by ID."""
        with self.get_session() as session:
            result = session.query(ExperimentResult).filter_by(id=result_id).first()
            if result:
                session.expunge(result)
            return result
    
    def _update_experiment_stats(self, session: Session, experiment_id: str):
        """Update experiment statistics based on results."""
        experiment = session.query(Experiment).filter_by(id=experiment_id).first()
        if not experiment:
            return
        
        # Get result statistics
        results = session.query(ExperimentResult).filter_by(experiment_id=experiment_id).all()
        
        experiment.total_runs = len(results)
        experiment.successful_runs = sum(1 for r in results if r.status == "completed")
        experiment.failed_runs = sum(1 for r in results if r.status == "failed")
        
        # Calculate execution time statistics
        execution_times = [r.execution_time for r in results if r.execution_time is not None]
        if execution_times:
            experiment.avg_execution_time = sum(execution_times) / len(execution_times)
            experiment.total_execution_time = sum(execution_times)
        
        session.flush()
    
    # Analytics and Reporting
    def get_experiment_summary(self, experiment_id: str) -> Optional[ExperimentSummary]:
        """Get a summary of an experiment."""
        with self.get_session() as session:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if not experiment:
                return None
            
            return ExperimentSummary(
                experiment_id=experiment.id,
                name=experiment.name,
                experiment_type=experiment.experiment_type,
                status=experiment.status,
                backend=experiment.backend,
                total_runs=experiment.total_runs,
                successful_runs=experiment.successful_runs,
                failed_runs=experiment.failed_runs,
                avg_execution_time=experiment.avg_execution_time,
                created_at=experiment.created_at,
                completed_at=experiment.completed_at
            )
    
    def get_experiment_summaries(self, limit: int = 50, offset: int = 0,
                                **filters) -> List[ExperimentSummary]:
        """Get summaries of multiple experiments."""
        experiments = self.get_experiments(limit=limit, offset=offset, **filters)
        return [
            ExperimentSummary(
                experiment_id=exp.id,
                name=exp.name,
                experiment_type=exp.experiment_type,
                status=exp.status,
                backend=exp.backend,
                total_runs=exp.total_runs,
                successful_runs=exp.successful_runs,
                failed_runs=exp.failed_runs,
                avg_execution_time=exp.avg_execution_time,
                created_at=exp.created_at,
                completed_at=exp.completed_at
            )
            for exp in experiments
        ]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_session() as session:
            stats = {
                'total_circuits': session.query(Circuit).count(),
                'total_experiments': session.query(Experiment).count(),
                'total_results': session.query(ExperimentResult).count(),
                'experiments_by_status': {},
                'experiments_by_backend': {},
                'experiments_by_type': {},
                'recent_activity': {
                    'experiments_last_24h': 0,
                    'results_last_24h': 0
                }
            }
            
            # Get experiments by status
            status_counts = session.query(
                Experiment.status, func.count(Experiment.id)
            ).group_by(Experiment.status).all()
            stats['experiments_by_status'] = {status: count for status, count in status_counts}
            
            # Get experiments by backend
            backend_counts = session.query(
                Experiment.backend, func.count(Experiment.id)
            ).group_by(Experiment.backend).all()
            stats['experiments_by_backend'] = {backend: count for backend, count in backend_counts}
            
            # Get experiments by type
            type_counts = session.query(
                Experiment.experiment_type, func.count(Experiment.id)
            ).group_by(Experiment.experiment_type).all()
            stats['experiments_by_type'] = {exp_type: count for exp_type, count in type_counts}
            
            # Get recent activity
            last_24h = datetime.now(timezone.utc) - timedelta(hours=24)
            stats['recent_activity']['experiments_last_24h'] = session.query(Experiment).filter(
                Experiment.created_at >= last_24h
            ).count()
            stats['recent_activity']['results_last_24h'] = session.query(ExperimentResult).filter(
                ExperimentResult.submitted_at >= last_24h
            ).count()
            
            return stats
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old experiment data."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        with self.get_session() as session:
            # Find old experiments
            old_experiments = session.query(Experiment).filter(
                Experiment.created_at < cutoff_date,
                Experiment.status.in_([ExperimentStatus.COMPLETED.value, ExperimentStatus.FAILED.value])
            ).all()
            
            cleanup_stats = {
                'experiments_deleted': 0,
                'results_deleted': 0,
                'circuits_deleted': 0
            }
            
            for experiment in old_experiments:
                # Delete results
                result_count = session.query(ExperimentResult).filter_by(
                    experiment_id=experiment.id
                ).count()
                session.query(ExperimentResult).filter_by(experiment_id=experiment.id).delete()
                cleanup_stats['results_deleted'] += result_count
                
                # Delete metrics
                session.query(ExperimentMetrics).filter_by(experiment_id=experiment.id).delete()
                
                # Delete experiment
                session.delete(experiment)
                cleanup_stats['experiments_deleted'] += 1
            
            # Clean up orphaned circuits (circuits with no experiments)
            orphaned_circuits = session.query(Circuit).filter(
                ~Circuit.experiments.any()
            ).all()
            
            for circuit in orphaned_circuits:
                session.delete(circuit)
                cleanup_stats['circuits_deleted'] += 1
            
            self.logger.info(f"Cleaned up old data: {cleanup_stats}")
            return cleanup_stats
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            import shutil
            shutil.copy2(self.database_path, backup_path)
            self.logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            return False
    
    def close(self):
        """Close all database connections."""
        self.close_thread_session()
        self.engine.dispose()
        self.logger.info("Database connections closed") 