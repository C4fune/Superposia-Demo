"""
Quantum Simulation Executor with Real-Time Monitoring

This module provides high-level interfaces for executing quantum circuits
on various simulation backends with comprehensive progress tracking,
job management, and dashboard integration.
"""

import threading
import time
from typing import Dict, Any, Optional, List, Union, Callable
import logging
import numpy as np

from quantum_platform.simulation.base import QuantumSimulator, SimulationResult
from quantum_platform.simulation.statevector import StateVectorSimulator
from quantum_platform.compiler.ir.circuit import QuantumCircuit
from quantum_platform.observability.logging import get_logger
from quantum_platform.observability.monitor import get_monitor

# Import execution monitoring components
from quantum_platform.execution.job_manager import (
    get_job_manager, ExecutionJob, JobType, JobStatus
)
from quantum_platform.execution.progress_tracker import (
    create_simulation_tracker, ProgressTracker, ProgressType
)
from quantum_platform.execution.dashboard import get_dashboard
from quantum_platform.observability.debug import get_debugger

class MonitoredSimulationExecutor:
    """
    Enhanced simulation executor with real-time monitoring capabilities.
    
    Integrates with the job manager and dashboard to provide comprehensive
    tracking of quantum simulation executions with progress updates.
    """
    
    def __init__(self, 
                 progress_update_interval: float = 0.1,
                 enable_monitoring: bool = True):
        """
        Initialize monitored simulation executor.
        
        Args:
            progress_update_interval: How often to update progress (seconds)
            enable_monitoring: Whether to enable execution monitoring
        """
        self.progress_update_interval = progress_update_interval
        self.enable_monitoring = enable_monitoring
        
        # Backends
        self.backends: Dict[str, QuantumSimulator] = {}
        self.default_backend = "statevector"
        
        # Logging and monitoring (initialize first)
        self.logger = get_logger("MonitoredSimulationExecutor")
        self.monitor = get_monitor()
        self.debugger = get_debugger()
        
        # Monitoring components
        self.job_manager = get_job_manager() if enable_monitoring else None
        self.dashboard = get_dashboard() if enable_monitoring else None
        
        # Register default backends
        self._register_default_backends()
        
        self.logger.info("Monitored simulation executor initialized")
    
    def _register_default_backends(self):
        """Register default simulation backends."""
        self.register_backend("statevector", StateVectorSimulator())
        
        # Future backends can be added here
        # self.register_backend("density_matrix", DensityMatrixSimulator())
        # self.register_backend("stabilizer", StabilizerSimulator())
    
    def register_backend(self, name: str, backend: QuantumSimulator):
        """
        Register a simulation backend.
        
        Args:
            name: Name of the backend
            backend: Simulator instance
        """
        self.backends[name] = backend
        self.logger.info(f"Registered simulation backend: {name}")
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backend names."""
        return list(self.backends.keys())
    
    def execute_circuit(self,
                       circuit: QuantumCircuit,
                       shots: int = 1000,
                       backend: Optional[str] = None,
                       job_name: Optional[str] = None,
                       initial_state: Optional[Union[str, List[complex]]] = None,
                       enable_progress_tracking: bool = True,
                       **kwargs) -> Union[SimulationResult, ExecutionJob]:
        """
        Execute a quantum circuit with optional monitoring.
        
        Args:
            circuit: Quantum circuit to execute
            shots: Number of measurement shots
            backend: Backend name (default: statevector)
            job_name: Optional name for the job
            initial_state: Optional initial state
            enable_progress_tracking: Whether to track progress
            **kwargs: Additional simulator parameters
            
        Returns:
            SimulationResult if monitoring disabled, ExecutionJob if enabled
        """
        # Select backend
        backend_name = backend or self.default_backend
        if backend_name not in self.backends:
            raise ValueError(f"Backend '{backend_name}' not available. "
                           f"Available: {list(self.backends.keys())}")
        
        simulator = self.backends[backend_name]
        
        # Generate job name if not provided
        if not job_name:
            job_name = f"Simulation: {circuit.name or 'Unnamed'} ({shots} shots)"
        
        # If monitoring is disabled, execute directly
        if not self.enable_monitoring:
            return self._execute_direct(simulator, circuit, shots, initial_state, **kwargs)
        
        # Create monitored job
        job = self.job_manager.create_job(
            job_type=JobType.SIMULATION,
            name=job_name,
            circuit_name=circuit.name,
            backend_name=backend_name,
            shots=shots,
            metadata={
                'circuit_depth': circuit.depth,
                'num_qubits': len(circuit.qubits),
                'num_operations': len(circuit.operations),
                'initial_state': str(initial_state) if initial_state else None,
                'parameters': kwargs
            },
            tags=['simulation', backend_name]
        )
        
        # Create progress tracker
        progress_tracker = None
        if enable_progress_tracking and shots > 100:  # Only for longer simulations
            progress_tracker = create_simulation_tracker(job_name, shots)
            
            # Set up progress callback
            def progress_callback(progress):
                job.update_progress(progress.percentage, progress.message)
            
            progress_tracker.add_callback(progress_callback)
        
        # Define execution function
        def execute_with_monitoring():
            try:
                # Start progress tracking
                if progress_tracker:
                    progress_tracker.start_simulation()
                
                # Execute simulation with progress updates
                result = self._execute_with_progress(
                    simulator, circuit, shots, initial_state, 
                    progress_tracker, **kwargs
                )
                
                # Complete progress tracking
                if progress_tracker:
                    progress_tracker.complete("Simulation completed successfully")
                
                return result
                
            except Exception as e:
                if progress_tracker:
                    progress_tracker.stop(f"Simulation failed: {str(e)}")
                raise
        
        # Submit job for execution
        self.job_manager.submit_job(job, lambda j: execute_with_monitoring())
        
        self.logger.info(f"Submitted simulation job: {job_name} ({job.job_id})")
        
        return job
    
    def _execute_direct(self,
                       simulator: QuantumSimulator,
                       circuit: QuantumCircuit,
                       shots: int,
                       initial_state: Optional[Union[str, List[complex]]] = None,
                       **kwargs) -> SimulationResult:
        """Execute simulation directly without monitoring."""
        with self.monitor.measure_operation("direct_simulation", "MonitoredExecutor"):
            return simulator.run(circuit, shots, initial_state=initial_state, **kwargs)
    
    def _execute_with_progress(self,
                              simulator: QuantumSimulator,
                              circuit: QuantumCircuit,
                              shots: int,
                              initial_state: Optional[Union[str, List[complex]]] = None,
                              progress_tracker: Optional[ProgressTracker] = None,
                              **kwargs) -> SimulationResult:
        """Execute simulation with progress tracking."""
        
        # For large simulations, break into batches for progress updates
        batch_size = max(1, min(100, shots // 10))  # Update every 10% or every 100 shots
        
        if shots <= batch_size or not progress_tracker:
            # Small simulation - execute all at once
            with self.monitor.measure_operation("simulation", "MonitoredExecutor"):
                return simulator.run(circuit, shots, initial_state=initial_state, **kwargs)
        
        # Large simulation - execute in batches
        self.logger.info(f"Executing {shots} shots in batches of {batch_size}")
        
        total_counts = {}
        total_shots_executed = 0
        batch_results = []
        
        with self.monitor.measure_operation("batched_simulation", "MonitoredExecutor"):
            # Calculate number of complete batches
            num_complete_batches = shots // batch_size
            remaining_shots = shots % batch_size
            
            # Execute complete batches
            for batch_idx in range(num_complete_batches):
                # Check for cancellation
                if (hasattr(simulator, 'is_cancelled') and 
                    simulator.is_cancelled()):
                    break
                
                # Execute batch
                batch_result = simulator.run(
                    circuit, batch_size, 
                    initial_state=initial_state, **kwargs
                )
                batch_results.append(batch_result)
                
                # Accumulate counts
                for outcome, count in batch_result.counts.items():
                    total_counts[outcome] = total_counts.get(outcome, 0) + count
                
                total_shots_executed += batch_size
                
                # Update progress
                progress_tracker.update_shot_progress(
                    total_shots_executed,
                    f"Completed batch {batch_idx + 1}/{num_complete_batches + (1 if remaining_shots > 0 else 0)}"
                )
                
                # Brief pause to allow other operations
                time.sleep(0.001)
            
            # Execute remaining shots if any
            if remaining_shots > 0:
                batch_result = simulator.run(
                    circuit, remaining_shots,
                    initial_state=initial_state, **kwargs
                )
                batch_results.append(batch_result)
                
                for outcome, count in batch_result.counts.items():
                    total_counts[outcome] = total_counts.get(outcome, 0) + count
                
                total_shots_executed += remaining_shots
                
                progress_tracker.update_shot_progress(
                    total_shots_executed,
                    "Completed all shots"
                )
        
        # Combine results from all batches
        # Use the last batch's state vector and metadata
        final_result = batch_results[-1] if batch_results else SimulationResult(
            counts={}, shots=0, execution_time=0.0
        )
        
        # Update with combined counts and total shots
        final_result.counts = total_counts
        final_result.shots = total_shots_executed
        
        return final_result
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a simulation job."""
        if not self.job_manager:
            return None
        
        job = self.job_manager.get_job(job_id)
        return job.to_dict() if job else None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a simulation job."""
        if not self.job_manager:
            return False
        
        return self.job_manager.cancel_job(job_id)
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get list of active simulation jobs."""
        if not self.job_manager:
            return []
        
        jobs = self.job_manager.get_jobs_by_type(JobType.SIMULATION)
        active_jobs = [job for job in jobs if job.is_active]
        return [job.to_dict() for job in active_jobs]
    
    def wait_for_job(self, job: ExecutionJob, timeout: Optional[float] = None) -> SimulationResult:
        """
        Wait for a job to complete and return the result.
        
        Args:
            job: Job to wait for
            timeout: Maximum time to wait (seconds)
            
        Returns:
            Simulation result
            
        Raises:
            TimeoutError: If job doesn't complete within timeout
            RuntimeError: If job fails
        """
        start_time = time.time()
        
        while not job.is_finished:
            time.sleep(0.1)
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job.job_id} did not complete within {timeout} seconds")
        
        if job.status == JobStatus.COMPLETED:
            return job.result
        elif job.status == JobStatus.FAILED:
            raise RuntimeError(f"Job {job.job_id} failed: {job.error_message}")
        elif job.status == JobStatus.CANCELLED:
            raise RuntimeError(f"Job {job.job_id} was cancelled")
        else:
            raise RuntimeError(f"Job {job.job_id} in unexpected state: {job.status}")

# Legacy compatibility - updated original executor
class SimulationExecutor(MonitoredSimulationExecutor):
    """
    Legacy SimulationExecutor class that now uses monitoring by default.
    
    This maintains backwards compatibility while adding monitoring capabilities.
    """
    
    def __init__(self):
        """Initialize with default monitoring enabled."""
        super().__init__(enable_monitoring=True)
        self.logger.info("Legacy SimulationExecutor initialized with monitoring")
    
    def run_circuit(self, 
                   circuit: QuantumCircuit,
                   backend_name: str = "statevector",
                   shots: int = 1000,
                   initial_state: Optional[Union[str, List[complex]]] = None,
                   **kwargs) -> SimulationResult:
        """
        Legacy method for running circuits - now returns results directly.
        
        Args:
            circuit: Quantum circuit to execute
            backend_name: Name of simulation backend
            shots: Number of measurement shots
            initial_state: Optional initial state
            **kwargs: Additional parameters
            
        Returns:
            Simulation result (waits for completion)
        """
        # For backwards compatibility, execute and wait for result
        job = self.execute_circuit(
            circuit=circuit,
            shots=shots,
            backend=backend_name,
            initial_state=initial_state,
            enable_progress_tracking=False,  # Disable for legacy compatibility
            **kwargs
        )
        
        if isinstance(job, SimulationResult):
            # Monitoring was disabled
            return job
        else:
            # Wait for job completion and return result
            return self.wait_for_job(job)

# Convenience functions

def execute_quantum_circuit(circuit: QuantumCircuit,
                           shots: int = 1000,
                           backend: str = "statevector",
                           job_name: Optional[str] = None,
                           monitor_progress: bool = True,
                           **kwargs) -> ExecutionJob:
    """
    Convenience function to execute a quantum circuit with monitoring.
    
    Args:
        circuit: Quantum circuit to execute
        shots: Number of measurement shots
        backend: Simulation backend name
        job_name: Optional job name
        monitor_progress: Whether to track progress
        **kwargs: Additional simulation parameters
        
    Returns:
        ExecutionJob for monitoring
    """
    executor = MonitoredSimulationExecutor()
    return executor.execute_circuit(
        circuit=circuit,
        shots=shots,
        backend=backend,
        job_name=job_name,
        enable_progress_tracking=monitor_progress,
        **kwargs
    )

def execute_and_wait(circuit: QuantumCircuit,
                    shots: int = 1000,
                    backend: str = "statevector",
                    timeout: Optional[float] = None,
                    **kwargs) -> SimulationResult:
    """
    Convenience function to execute and wait for completion.
    
    Args:
        circuit: Quantum circuit to execute
        shots: Number of measurement shots
        backend: Simulation backend name
        timeout: Maximum wait time
        **kwargs: Additional simulation parameters
        
    Returns:
        Simulation result
    """
    executor = MonitoredSimulationExecutor()
    job = executor.execute_circuit(
        circuit=circuit,
        shots=shots,
        backend=backend,
        enable_progress_tracking=True,
        **kwargs
    )
    
    if isinstance(job, SimulationResult):
        return job
    else:
        return executor.wait_for_job(job, timeout)

# Global executor instance
_global_executor: Optional[MonitoredSimulationExecutor] = None
_executor_lock = threading.Lock()

def get_simulation_executor() -> MonitoredSimulationExecutor:
    """Get the global simulation executor instance."""
    global _global_executor
    
    with _executor_lock:
        if _global_executor is None:
            _global_executor = MonitoredSimulationExecutor()
    
    return _global_executor 