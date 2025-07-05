"""
Local Simulator Backend

This backend provides a local quantum simulator that implements the HAL interface.
It's useful for testing, development, and offline usage.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...compiler.ir.circuit import QuantumCircuit
from ...simulation import StateVectorSimulator
from ...errors import SimulationError, handle_errors
from ..hal import (
    QuantumHardwareBackend, JobHandle, JobStatus, DeviceInfo, 
    HardwareResult, DeviceType
)


class LocalSimulatorBackend(QuantumHardwareBackend):
    """Local quantum simulator backend."""
    
    def __init__(self, name: str = "local_simulator", max_qubits: int = 30):
        super().__init__(name, "local")
        self.max_qubits = max_qubits
        self.simulator = StateVectorSimulator()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._job_lock = threading.Lock()
        
    def initialize(self) -> bool:
        """Initialize the simulator backend."""
        try:
            # Test the simulator
            from ...compiler.language.dsl import QuantumProgram
            with QuantumProgram() as qp:
                qubit = qp.allocate(1)
                from ...compiler.language.operations import H
                H(qubit)
            
            test_result = self.simulator.run(qp.circuit, shots=10)
            self._initialized = True
            return True
            
        except Exception as e:
            raise SimulationError(
                f"Failed to initialize local simulator: {e}",
                user_message="Local simulator initialization failed"
            )
    
    def get_device_info(self) -> DeviceInfo:
        """Get information about the local simulator."""
        if self._device_info is None:
            self._device_info = DeviceInfo(
                name=self.name,
                provider=self.provider,
                device_type=DeviceType.SIMULATOR,
                num_qubits=self.max_qubits,
                coupling_map=self._generate_all_to_all_coupling(),
                basis_gates=[
                    "h", "x", "y", "z", "rx", "ry", "rz", "cx", "cy", "cz",
                    "swap", "ccx", "u1", "u2", "u3", "measure"
                ],
                max_shots=1000000,
                max_experiments=1000,
                simulator=True,
                operational=True,
                queue_length=0,
                avg_wait_time=0.0,
                metadata={
                    "simulator_type": "statevector",
                    "noise_model": False,
                    "supports_transpilation": False  # All gates supported
                }
            )
        return self._device_info
    
    def _generate_all_to_all_coupling(self) -> List[List[int]]:
        """Generate all-to-all coupling map."""
        coupling = []
        for i in range(self.max_qubits):
            for j in range(self.max_qubits):
                if i != j:
                    coupling.append([i, j])
        return coupling
    
    def submit_circuit(self, circuit: QuantumCircuit, shots: int = 1000, 
                      **kwargs) -> JobHandle:
        """Submit a circuit for simulation."""
        # Validate circuit
        self.validate_circuit(circuit)
        
        # Create job handle
        job_handle = JobHandle(
            job_id=f"sim_{int(time.time() * 1000)}_{id(circuit)}",
            backend_name=self.name,
            circuit_id=getattr(circuit, 'name', None),
            shots=shots,
            metadata=kwargs
        )
        
        # Store job information
        with self._job_lock:
            self._jobs[job_handle.job_id] = {
                "handle": job_handle,
                "circuit": circuit,
                "status": JobStatus.PENDING,
                "submitted_at": datetime.now(),
                "shots": shots,
                "kwargs": kwargs
            }
        
        # Start simulation in background thread
        thread = threading.Thread(
            target=self._run_simulation,
            args=(job_handle.job_id,),
            daemon=True
        )
        thread.start()
        
        return job_handle
    
    def _run_simulation(self, job_id: str):
        """Run simulation in background thread."""
        try:
            with self._job_lock:
                if job_id not in self._jobs:
                    return
                job_data = self._jobs[job_id]
                job_data["status"] = JobStatus.RUNNING
                job_data["start_time"] = datetime.now()
            
            # Add realistic delay for simulation
            time.sleep(0.1 + job_data["shots"] / 100000.0)  # Scale with shots
            
            # Run simulation
            result = self.simulator.run(job_data["circuit"], shots=job_data["shots"])
            
            # Store results
            with self._job_lock:
                job_data["status"] = JobStatus.COMPLETED
                job_data["result"] = result
                job_data["end_time"] = datetime.now()
                
        except Exception as e:
            with self._job_lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["status"] = JobStatus.FAILED
                    self._jobs[job_id]["error"] = str(e)
                    self._jobs[job_id]["end_time"] = datetime.now()
    
    def get_job_status(self, job_handle: JobHandle) -> JobStatus:
        """Get the status of a simulation job."""
        with self._job_lock:
            if job_handle.job_id not in self._jobs:
                return JobStatus.FAILED
            return self._jobs[job_handle.job_id]["status"]
    
    def retrieve_results(self, job_handle: JobHandle) -> HardwareResult:
        """Retrieve simulation results."""
        with self._job_lock:
            if job_handle.job_id not in self._jobs:
                return HardwareResult(
                    job_handle=job_handle,
                    status=JobStatus.FAILED,
                    error_message="Job not found",
                    error_code="JOB_NOT_FOUND"
                )
            
            job_data = self._jobs[job_handle.job_id]
            status = job_data["status"]
            
            if status == JobStatus.FAILED:
                return HardwareResult(
                    job_handle=job_handle,
                    status=status,
                    error_message=job_data.get("error", "Simulation failed"),
                    error_code="SIMULATION_ERROR"
                )
            
            if status != JobStatus.COMPLETED:
                return HardwareResult(
                    job_handle=job_handle,
                    status=status
                )
            
            # Extract results
            sim_result = job_data["result"]
            execution_time = None
            
            if "start_time" in job_data and "end_time" in job_data:
                execution_time = (
                    job_data["end_time"] - job_data["start_time"]
                ).total_seconds() * 1000  # Convert to milliseconds
            
            return HardwareResult(
                job_handle=job_handle,
                status=status,
                counts=getattr(sim_result, 'counts', {}),
                execution_time=execution_time,
                queue_time=0.0,  # No queue time for local simulator
                shots=job_data["shots"],
                metadata={
                    "simulator": True,
                    "backend": self.name,
                    "submitted_at": job_data["submitted_at"].isoformat()
                },
                raw_result=sim_result
            )
    
    def cancel_job(self, job_handle: JobHandle) -> bool:
        """Cancel a simulation job."""
        with self._job_lock:
            if job_handle.job_id not in self._jobs:
                return False
            
            job_data = self._jobs[job_handle.job_id]
            if job_data["status"] in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING]:
                job_data["status"] = JobStatus.CANCELLED
                job_data["end_time"] = datetime.now()
                return True
            
            return False
    
    def validate_circuit(self, circuit: QuantumCircuit) -> bool:
        """Validate circuit for local simulator."""
        if circuit.num_qubits > self.max_qubits:
            raise SimulationError(
                f"Circuit requires {circuit.num_qubits} qubits, "
                f"but simulator only supports {self.max_qubits}",
                user_message=f"Circuit too large for local simulator"
            )
        return True
    
    def cleanup_jobs(self, max_age_hours: int = 24):
        """Clean up old job data."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._job_lock:
            jobs_to_remove = []
            for job_id, job_data in self._jobs.items():
                if job_data["submitted_at"] < cutoff_time:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self._jobs[job_id]
        
        return len(jobs_to_remove) 