"""
Experiment Manager

This module provides the high-level interface for managing quantum experiments,
integrating with the existing quantum platform components and providing
a unified API for experiment creation, execution, and analysis.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from quantum_platform.hardware.hal import QuantumHardwareBackend, HardwareResult
from quantum_platform.hardware.results import AggregatedResult, MultiShotExecutor
from quantum_platform.simulation.base import SimulationResult
from quantum_platform.security.audit import SecurityAuditLogger, AuditEventType
from quantum_platform.compiler.ir.circuit import QuantumCircuit
from quantum_platform.orchestration.hybrid_executor import HybridExecutor, HybridResult

from .database import ExperimentDatabase
from .models import (
    Experiment, ExperimentResult, Circuit, ExperimentStatus,
    ExperimentType, ExperimentSummary
)
from .analyzer import ExperimentAnalyzer


class ExperimentManager:
    """
    High-level experiment management interface.
    
    This class provides the primary API for creating, running, and managing
    quantum experiments, integrating with all quantum platform components.
    """
    
    def __init__(self, database: Optional[ExperimentDatabase] = None,
                 enable_audit: bool = True, max_workers: int = 4):
        """
        Initialize the experiment manager.
        
        Args:
            database: Experiment database instance
            enable_audit: Whether to enable audit logging
            max_workers: Maximum number of worker threads for parallel execution
        """
        self.database = database or ExperimentDatabase()
        self.enable_audit = enable_audit
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.analyzer = ExperimentAnalyzer(self.database)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Multi-shot executor for hardware runs
        self.multi_shot_executor = MultiShotExecutor()
        
        # Audit logger
        if enable_audit:
            self.audit_logger = SecurityAuditLogger.get_instance()
        else:
            self.audit_logger = None
        
        # Active experiments tracking
        self._active_experiments: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Experiment manager initialized")
    
    def create_circuit(self, name: str, circuit: QuantumCircuit,
                      description: str = None, version: str = "1.0",
                      parameters: Dict[str, Any] = None) -> Circuit:
        """
        Create and store a circuit definition.
        
        Args:
            name: Circuit name
            circuit: QuantumCircuit object
            description: Circuit description
            version: Circuit version
            parameters: Circuit parameters
            
        Returns:
            Circuit database record
        """
        try:
            # Convert circuit to QASM
            qasm_code = self._circuit_to_qasm(circuit)
            
            # Create circuit record
            circuit_record = self.database.create_circuit(
                name=name,
                qasm_code=qasm_code,
                num_qubits=circuit.num_qubits,
                description=description,
                circuit_json=self._circuit_to_json(circuit),
                parameters=parameters,
                version=version
            )
            
            # Audit log
            if self.audit_logger:
                self.audit_logger.log_system_operation(
                    operation="circuit_create",
                    resource=f"circuit:{circuit_record.id}",
                    details={"name": name, "qubits": circuit.num_qubits}
                )
            
            self.logger.info(f"Created circuit: {circuit_record.id} ({name})")
            return circuit_record
            
        except Exception as e:
            self.logger.error(f"Failed to create circuit: {e}")
            raise
    
    def create_experiment(self, name: str, circuit_id: str, backend: QuantumHardwareBackend,
                         experiment_type: str = ExperimentType.SINGLE_SHOT.value,
                         description: str = None, shots: int = 1000,
                         parameter_sweep: Dict[str, Any] = None,
                         tags: List[str] = None, metadata: Dict[str, Any] = None,
                         execution_config: Dict[str, Any] = None) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            circuit_id: ID of the circuit to execute
            backend: Backend to use for execution
            experiment_type: Type of experiment
            description: Experiment description
            shots: Number of shots per execution
            parameter_sweep: Parameter sweep configuration
            tags: Experiment tags
            metadata: Additional metadata
            execution_config: Execution configuration
            
        Returns:
            Experiment database record
        """
        try:
            # Get current user context
            current_user = self._get_current_user()
            session_id = self._get_session_id()
            
            # Create experiment record
            experiment = self.database.create_experiment(
                name=name,
                circuit_id=circuit_id,
                backend=backend.name,
                experiment_type=experiment_type,
                description=description,
                shots=shots,
                provider=getattr(backend, 'provider', None),
                device_name=getattr(backend, 'device_name', None),
                parameter_sweep=parameter_sweep,
                tags=tags,
                created_by=current_user,
                session_id=session_id,
                metadata=metadata or {}
            )
            
            # Audit log
            if self.audit_logger:
                self.audit_logger.log_system_operation(
                    operation="experiment_create",
                    resource=f"experiment:{experiment.id}",
                    details={
                        "name": name,
                        "circuit_id": circuit_id,
                        "backend": backend.name,
                        "experiment_type": experiment_type
                    }
                )
            
            self.logger.info(f"Created experiment: {experiment.id} ({name})")
            return experiment
            
        except Exception as e:
            self.logger.error(f"Failed to create experiment: {e}")
            raise
    
    def run_experiment(self, experiment_id: str, callback: Callable[[str, Dict], None] = None,
                      async_execution: bool = False) -> Union[Experiment, Dict[str, Any]]:
        """
        Execute an experiment.
        
        Args:
            experiment_id: ID of the experiment to run
            callback: Optional callback function for progress updates
            async_execution: Whether to run asynchronously
            
        Returns:
            Experiment record or future result if async
        """
        experiment = self.database.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if async_execution:
            future = self.executor.submit(self._execute_experiment, experiment, callback)
            return {"future": future, "experiment_id": experiment_id}
        else:
            return self._execute_experiment(experiment, callback)
    
    def _execute_experiment(self, experiment: Experiment, 
                          callback: Callable[[str, Dict], None] = None) -> Experiment:
        """
        Internal method to execute an experiment.
        
        Args:
            experiment: Experiment to execute
            callback: Optional progress callback
            
        Returns:
            Updated experiment record
        """
        experiment_id = experiment.id
        
        try:
            # Update status to running
            self.database.update_experiment_status(
                experiment_id, ExperimentStatus.RUNNING.value
            )
            
            # Track active experiment
            self._active_experiments[experiment_id] = {
                "start_time": time.time(),
                "status": "running",
                "progress": 0.0
            }
            
            # Get circuit
            circuit = self.database.get_circuit(experiment.circuit_id)
            if not circuit:
                raise ValueError(f"Circuit {experiment.circuit_id} not found")
            
            # Convert circuit back to QuantumCircuit object
            quantum_circuit = self._circuit_from_json(circuit.circuit_json)
            
            # Get backend
            backend = self._get_backend(experiment.backend, experiment.provider)
            
            # Execute based on experiment type
            if experiment.experiment_type == ExperimentType.SINGLE_SHOT.value:
                self._execute_single_shot(experiment, quantum_circuit, backend, callback)
            elif experiment.experiment_type == ExperimentType.PARAMETER_SWEEP.value:
                self._execute_parameter_sweep(experiment, quantum_circuit, backend, callback)
            elif experiment.experiment_type == ExperimentType.OPTIMIZATION.value:
                self._execute_optimization(experiment, quantum_circuit, backend, callback)
            else:
                raise ValueError(f"Unsupported experiment type: {experiment.experiment_type}")
            
            # Update status to completed
            self.database.update_experiment_status(
                experiment_id, ExperimentStatus.COMPLETED.value
            )
            
            # Clean up active experiment tracking
            if experiment_id in self._active_experiments:
                del self._active_experiments[experiment_id]
            
            # Audit log
            if self.audit_logger:
                self.audit_logger.log_system_operation(
                    operation="experiment_execute",
                    resource=f"experiment:{experiment_id}",
                    details={"status": "completed"}
                )
            
            self.logger.info(f"Experiment {experiment_id} completed successfully")
            return self.database.get_experiment(experiment_id)
            
        except Exception as e:
            # Update status to failed
            self.database.update_experiment_status(
                experiment_id, ExperimentStatus.FAILED.value
            )
            
            # Clean up active experiment tracking
            if experiment_id in self._active_experiments:
                del self._active_experiments[experiment_id]
            
            # Audit log
            if self.audit_logger:
                self.audit_logger.log_system_operation(
                    operation="experiment_execute",
                    resource=f"experiment:{experiment_id}",
                    success=False,
                    details={"error": str(e)}
                )
            
            self.logger.error(f"Experiment {experiment_id} failed: {e}")
            raise
    
    def _execute_single_shot(self, experiment: Experiment, circuit: QuantumCircuit,
                           backend: QuantumHardwareBackend, callback: Callable = None):
        """Execute a single-shot experiment."""
        # Execute circuit
        result = self.multi_shot_executor.execute_with_aggregation(
            backend, circuit, experiment.shots
        )
        
        # Store result
        self.database.create_result(
            experiment_id=experiment.id,
            run_number=1,
            raw_counts=result.measurement_counts,
            shots=experiment.shots,
            execution_time=result.execution_time,
            fidelity=result.fidelity,
            success_probability=result.success_probability,
            custom_metrics=result.custom_metrics
        )
        
        if callback:
            callback(experiment.id, {"progress": 1.0, "status": "completed"})
    
    def _execute_parameter_sweep(self, experiment: Experiment, circuit: QuantumCircuit,
                               backend: QuantumHardwareBackend, callback: Callable = None):
        """Execute a parameter sweep experiment."""
        if not experiment.parameter_sweep:
            raise ValueError("Parameter sweep configuration not found")
        
        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_combinations(
            experiment.parameter_sweep
        )
        
        total_runs = len(parameter_combinations)
        
        for i, params in enumerate(parameter_combinations):
            # Update progress
            progress = (i + 1) / total_runs
            if experiment.id in self._active_experiments:
                self._active_experiments[experiment.id]["progress"] = progress
            
            # Bind parameters to circuit
            parameterized_circuit = self._bind_parameters(circuit, params)
            
            # Execute
            result = self.multi_shot_executor.execute_with_aggregation(
                backend, parameterized_circuit, experiment.shots
            )
            
            # Store result
            self.database.create_result(
                experiment_id=experiment.id,
                run_number=i + 1,
                raw_counts=result.measurement_counts,
                shots=experiment.shots,
                parameter_values=params,
                execution_time=result.execution_time,
                fidelity=result.fidelity,
                success_probability=result.success_probability,
                custom_metrics=result.custom_metrics
            )
            
            if callback:
                callback(experiment.id, {
                    "progress": progress,
                    "run_number": i + 1,
                    "total_runs": total_runs,
                    "parameters": params
                })
    
    def _execute_optimization(self, experiment: Experiment, circuit: QuantumCircuit,
                            backend: QuantumHardwareBackend, callback: Callable = None):
        """Execute an optimization experiment."""
        # Use hybrid executor for optimization
        hybrid_executor = HybridExecutor(backend)
        
        # Get optimization parameters
        opt_config = experiment.parameter_sweep or {}
        initial_params = opt_config.get("initial_parameters", {})
        
        # Execute optimization
        result = hybrid_executor.execute(initial_params)
        
        # Store result
        self.database.create_result(
            experiment_id=experiment.id,
            run_number=1,
            raw_counts=result.raw_result.counts,
            shots=experiment.shots,
            parameter_values=result.parameters,
            execution_time=result.execution_time,
            expectation_value=result.expectation_value,
            custom_metrics={"optimization_result": result.metadata}
        )
        
        if callback:
            callback(experiment.id, {"progress": 1.0, "status": "completed"})
    
    def get_experiment_results(self, experiment_id: str) -> List[ExperimentResult]:
        """Get all results for an experiment."""
        return self.database.get_results(experiment_id)
    
    def get_experiment_summary(self, experiment_id: str) -> Optional[ExperimentSummary]:
        """Get a summary of an experiment."""
        return self.database.get_experiment_summary(experiment_id)
    
    def list_experiments(self, limit: int = 50, offset: int = 0, **filters) -> List[ExperimentSummary]:
        """List experiments with optional filtering."""
        return self.database.get_experiment_summaries(limit=limit, offset=offset, **filters)
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all its results."""
        try:
            success = self.database.delete_experiment(experiment_id)
            
            if success and self.audit_logger:
                self.audit_logger.log_system_operation(
                    operation="experiment_delete",
                    resource=f"experiment:{experiment_id}"
                )
            
            return success
        except Exception as e:
            self.logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            return False
    
    def get_active_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active experiments."""
        return self._active_experiments.copy()
    
    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running experiment."""
        if experiment_id in self._active_experiments:
            self._active_experiments[experiment_id]["status"] = "cancelled"
            self.database.update_experiment_status(
                experiment_id, ExperimentStatus.CANCELLED.value
            )
            return True
        return False
    
    def compare_experiments(self, experiment_id1: str, experiment_id2: str) -> Dict[str, Any]:
        """Compare two experiments."""
        return self.analyzer.compare_experiments(experiment_id1, experiment_id2)
    
    def get_experiment_trends(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Analyze trends across multiple experiments."""
        return self.analyzer.analyze_trends(experiment_ids)
    
    def export_experiment_data(self, experiment_id: str, format: str = "json") -> str:
        """Export experiment data in various formats."""
        experiment = self.database.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        results = self.database.get_results(experiment_id)
        circuit = self.database.get_circuit(experiment.circuit_id)
        
        if format == "json":
            import json
            export_data = {
                "experiment": experiment.to_dict(),
                "circuit": circuit.to_dict() if circuit else None,
                "results": [result.to_dict() for result in results]
            }
            return json.dumps(export_data, indent=2)
        elif format == "csv":
            return self._export_to_csv(experiment, results)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.database.get_database_stats()
    
    def cleanup_old_experiments(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old experiment data."""
        return self.database.cleanup_old_data(days_to_keep)
    
    def backup_experiments(self, backup_path: str) -> bool:
        """Create a backup of the experiment database."""
        return self.database.backup_database(backup_path)
    
    # Helper methods
    def _circuit_to_qasm(self, circuit: QuantumCircuit) -> str:
        """Convert QuantumCircuit to QASM string."""
        # Implementation depends on QuantumCircuit structure
        # This is a placeholder - actual implementation would depend on the circuit format
        return f"// QASM for circuit with {circuit.num_qubits} qubits"
    
    def _circuit_to_json(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Convert QuantumCircuit to JSON representation."""
        # Implementation depends on QuantumCircuit structure
        return {
            "num_qubits": circuit.num_qubits,
            "operations": [],  # Would include actual operations
            "measurements": []
        }
    
    def _circuit_from_json(self, circuit_json: Dict[str, Any]) -> QuantumCircuit:
        """Convert JSON representation back to QuantumCircuit."""
        # Implementation depends on QuantumCircuit structure
        return QuantumCircuit(num_qubits=circuit_json["num_qubits"])
    
    def _get_backend(self, backend_name: str, provider: str = None) -> QuantumHardwareBackend:
        """Get backend instance by name."""
        # This would integrate with the provider system
        # For now, return a placeholder
        from quantum_platform.hardware.backends.local_simulator import LocalSimulatorBackend
        return LocalSimulatorBackend()
    
    def _get_current_user(self) -> str:
        """Get current user ID."""
        try:
            from quantum_platform.security.user import UserContext
            current_user = UserContext.get_current_user()
            return current_user.user_id if current_user else "anonymous"
        except ImportError:
            return "anonymous"
    
    def _get_session_id(self) -> str:
        """Get current session ID."""
        return str(uuid.uuid4())
    
    def _generate_parameter_combinations(self, sweep_config: Dict[str, Any]) -> List[Dict[str, float]]:
        """Generate parameter combinations for parameter sweep."""
        # Implementation would generate all combinations based on sweep configuration
        return [{"theta": 0.0, "phi": 0.0}]  # Placeholder
    
    def _bind_parameters(self, circuit: QuantumCircuit, params: Dict[str, float]) -> QuantumCircuit:
        """Bind parameters to a circuit."""
        # Implementation depends on QuantumCircuit structure
        return circuit  # Placeholder
    
    def _export_to_csv(self, experiment: Experiment, results: List[ExperimentResult]) -> str:
        """Export experiment results to CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "run_number", "execution_time", "fidelity", "success_probability",
            "expectation_value", "shots", "status"
        ])
        
        # Write results
        for result in results:
            writer.writerow([
                result.run_number,
                result.execution_time,
                result.fidelity,
                result.success_probability,
                result.expectation_value,
                result.shots,
                result.status
            ])
        
        return output.getvalue()
    
    def close(self):
        """Close the experiment manager and clean up resources."""
        self.executor.shutdown(wait=True)
        self.database.close()
        self.logger.info("Experiment manager closed") 