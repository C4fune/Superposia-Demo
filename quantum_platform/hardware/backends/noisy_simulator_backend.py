"""
Noisy Simulator Backend

This backend provides quantum simulation with realistic noise models,
allowing users to test circuits with device-specific noise characteristics.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ...compiler.ir.circuit import QuantumCircuit
from ...simulation.noisy_simulator import NoisyQuantumSimulator, create_device_simulator
from ...simulation.noise_models import NoiseModel, get_noise_library
from ...errors import SimulationError
from ..hal import (
    QuantumHardwareBackend, JobHandle, JobStatus, DeviceInfo, 
    HardwareResult, DeviceType
)


class NoisySimulatorBackend(QuantumHardwareBackend):
    """Quantum simulator backend with realistic noise models."""
    
    def __init__(self, name: str = "noisy_simulator", 
                 device_type: str = "ibm_like", max_qubits: int = 30):
        super().__init__(name, "local")
        self.max_qubits = max_qubits
        self.device_type = device_type
        self.simulator = None
        self.noise_model = None
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._job_lock = threading.Lock()
        
        # Performance settings
        self.parallel_threshold = 200  # Use parallel simulation for shots > this
        self.max_circuit_size = 20     # Maximum qubits for efficient simulation
        
    def initialize(self) -> bool:
        """Initialize the noisy simulator backend."""
        try:
            # Load noise model from library
            library = get_noise_library()
            self.noise_model = library.get_model(self.device_type)
            
            if self.noise_model is None:
                raise SimulationError(f"Unknown device type: {self.device_type}")
            
            # Create noisy simulator
            self.simulator = NoisyQuantumSimulator(self.noise_model)
            
            # Test the simulator with a simple circuit
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
                f"Failed to initialize noisy simulator: {e}",
                user_message="Noisy simulator initialization failed"
            )
    
    def get_device_info(self) -> DeviceInfo:
        """Get information about the simulated device."""
        if self._device_info is None:
            # Base device info on the noise model
            device_type_map = {
                "ibm_like": DeviceType.SUPERCONDUCTING,
                "ionq_like": DeviceType.TRAPPED_ION,
                "google_like": DeviceType.SUPERCONDUCTING,
                "ideal": DeviceType.SIMULATOR
            }
            
            device_type = device_type_map.get(self.device_type, DeviceType.SIMULATOR)
            
            # Get noise characteristics for metadata
            noise_info = {}
            if self.noise_model:
                if self.noise_model.coherence_params:
                    # Average T1 and T2 times
                    t1_values = [params.T1.value for params in self.noise_model.coherence_params.values()]
                    t2_values = [params.T2.value for params in self.noise_model.coherence_params.values()]
                    
                    noise_info.update({
                        "avg_T1": sum(t1_values) / len(t1_values) if t1_values else 0,
                        "avg_T2": sum(t2_values) / len(t2_values) if t2_values else 0,
                        "min_T1": min(t1_values) if t1_values else 0,
                        "max_T1": max(t1_values) if t1_values else 0,
                        "min_T2": min(t2_values) if t2_values else 0,
                        "max_T2": max(t2_values) if t2_values else 0,
                    })
                
                noise_info.update({
                    "single_qubit_error": self.noise_model.gate_errors.single_qubit_error,
                    "two_qubit_error": self.noise_model.gate_errors.two_qubit_error,
                    "measurement_error": self.noise_model.gate_errors.measurement_error,
                    "noise_model_name": self.noise_model.name,
                    "thermal_population": self.noise_model.thermal_population
                })
            
            self._device_info = DeviceInfo(
                name=self.name,
                provider=self.provider,
                device_type=device_type,
                num_qubits=self.max_qubits,
                coupling_map=self._generate_coupling_map(),
                basis_gates=self._get_basis_gates(),
                max_shots=1000000,
                max_experiments=1000,
                simulator=True,
                operational=True,
                queue_length=0,
                avg_wait_time=0.0,
                metadata={
                    "simulator_type": "noisy_monte_carlo",
                    "device_type": self.device_type,
                    "noise_enabled": self.noise_model.enabled if self.noise_model else False,
                    "supports_noise_toggle": True,
                    "parallel_simulation": True,
                    **noise_info
                }
            )
        return self._device_info
    
    def _generate_coupling_map(self) -> List[List[int]]:
        """Generate coupling map based on device type."""
        if self.device_type == "ionq_like":
            # All-to-all connectivity for trapped ions
            coupling = []
            for i in range(min(self.max_qubits, 11)):  # IonQ-like has 11 qubits
                for j in range(min(self.max_qubits, 11)):
                    if i != j:
                        coupling.append([i, j])
            return coupling
        
        elif self.device_type in ["ibm_like", "google_like"]:
            # Limited connectivity for superconducting devices
            coupling = []
            # Create a simple linear chain with some additional connections
            for i in range(min(self.max_qubits - 1, 19)):
                coupling.append([i, i + 1])
                coupling.append([i + 1, i])
            
            # Add some additional connections for more realistic topology
            if self.max_qubits >= 5:
                additional_connections = [
                    [0, 2], [2, 0], [1, 3], [3, 1], [2, 4], [4, 2]
                ]
                for conn in additional_connections:
                    if conn[0] < self.max_qubits and conn[1] < self.max_qubits:
                        coupling.append(conn)
            
            return coupling
        
        else:
            # Default all-to-all for other types
            coupling = []
            for i in range(self.max_qubits):
                for j in range(self.max_qubits):
                    if i != j:
                        coupling.append([i, j])
            return coupling
    
    def _get_basis_gates(self) -> List[str]:
        """Get basis gates for the simulated device."""
        if self.device_type == "ionq_like":
            return ["gpi", "gpi2", "ms", "measure", "reset"]
        elif self.device_type in ["ibm_like", "google_like"]:
            return ["id", "rz", "sx", "x", "cx", "measure", "reset"]
        else:
            return [
                "h", "x", "y", "z", "rx", "ry", "rz", "cx", "cy", "cz",
                "swap", "ccx", "u1", "u2", "u3", "measure", "reset"
            ]
    
    def submit_circuit(self, circuit: QuantumCircuit, shots: int = 1000, 
                      **kwargs) -> JobHandle:
        """Submit a circuit for noisy simulation."""
        # Validate circuit
        self.validate_circuit(circuit)
        
        # Check for noise-related options
        compare_ideal = kwargs.get('compare_ideal', False)
        noise_enabled = kwargs.get('noise_enabled', True)
        
        # Create job handle
        job_handle = JobHandle(
            job_id=f"noisy_sim_{int(time.time() * 1000)}_{id(circuit)}",
            backend_name=self.name,
            circuit_id=getattr(circuit, 'name', None),
            shots=shots,
            metadata={
                **kwargs,
                'noise_enabled': noise_enabled,
                'compare_ideal': compare_ideal,
                'device_type': self.device_type
            }
        )
        
        # Store job information
        with self._job_lock:
            self._jobs[job_handle.job_id] = {
                "handle": job_handle,
                "circuit": circuit,
                "status": JobStatus.PENDING,
                "submitted_at": datetime.now(),
                "shots": shots,
                "kwargs": kwargs,
                "noise_enabled": noise_enabled,
                "compare_ideal": compare_ideal
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
        """Run noisy simulation in background thread."""
        try:
            with self._job_lock:
                if job_id not in self._jobs:
                    return
                job_data = self._jobs[job_id]
                job_data["status"] = JobStatus.RUNNING
                job_data["start_time"] = datetime.now()
            
            # Get job parameters
            circuit = job_data["circuit"]
            shots = job_data["shots"]
            noise_enabled = job_data["noise_enabled"]
            compare_ideal = job_data["compare_ideal"]
            
            # Temporarily disable noise if requested
            original_noise_state = None
            if not noise_enabled and self.noise_model:
                original_noise_state = self.noise_model.enabled
                self.noise_model.enabled = False
            
            # Add realistic delay based on circuit complexity and shots
            complexity_factor = circuit.num_qubits * len(circuit.operations)
            base_delay = 0.1  # Base simulation time
            complexity_delay = complexity_factor * 0.001
            shots_delay = shots * 0.00001  # Delay scales with shots
            
            simulation_delay = min(base_delay + complexity_delay + shots_delay, 5.0)
            time.sleep(simulation_delay)
            
            # Run simulation
            if compare_ideal:
                result = self.simulator.run(circuit, shots=shots, compare_ideal=True)
            else:
                result = self.simulator.run(circuit, shots=shots)
            
            # Restore noise state
            if original_noise_state is not None:
                self.noise_model.enabled = original_noise_state
            
            # Store results
            with self._job_lock:
                job_data["status"] = JobStatus.COMPLETED
                job_data["result"] = result
                job_data["end_time"] = datetime.now()
                
        except Exception as e:
            # Restore noise state in case of error
            if 'original_noise_state' in locals() and original_noise_state is not None:
                self.noise_model.enabled = original_noise_state
            
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
            
            # Create metadata with noise information
            metadata = {
                "simulator": True,
                "noisy_simulation": True,
                "backend": self.name,
                "device_type": self.device_type,
                "noise_model": sim_result.noise_model_name,
                "submitted_at": job_data["submitted_at"].isoformat(),
                "noise_enabled": job_data["noise_enabled"]
            }
            
            # Add noise-specific metrics if available
            if hasattr(sim_result, 'noise_overhead'):
                metadata["noise_overhead"] = sim_result.noise_overhead
            
            if hasattr(sim_result, 'ideal_counts') and sim_result.ideal_counts:
                metadata["ideal_counts"] = sim_result.ideal_counts
            
            if hasattr(sim_result, 'error_events'):
                metadata["error_events"] = len(sim_result.error_events)
            
            return HardwareResult(
                job_handle=job_handle,
                status=status,
                counts=sim_result.counts,
                execution_time=execution_time,
                queue_time=0.0,  # No queue time for local simulator
                shots=job_data["shots"],
                metadata=metadata,
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
        """Validate circuit for noisy simulation."""
        if circuit.num_qubits > self.max_qubits:
            raise SimulationError(
                f"Circuit requires {circuit.num_qubits} qubits, "
                f"but simulator only supports {self.max_qubits}",
                user_message=f"Circuit too large for {self.device_type} simulator"
            )
        
        # Check if circuit is too complex for efficient simulation
        if circuit.num_qubits > self.max_circuit_size:
            raise SimulationError(
                f"Circuit with {circuit.num_qubits} qubits may be too large "
                f"for efficient noisy simulation (max recommended: {self.max_circuit_size})",
                user_message="Consider using a smaller circuit or ideal simulation"
            )
        
        return True
    
    def set_noise_model(self, noise_model: NoiseModel):
        """Change the noise model for this backend."""
        self.noise_model = noise_model
        if self.simulator:
            self.simulator.set_noise_model(noise_model)
        
        # Update device info to reflect new noise model
        self._device_info = None
    
    def toggle_noise(self, enabled: bool):
        """Enable or disable noise simulation."""
        if self.noise_model:
            self.noise_model.enabled = enabled
    
    def get_noise_characteristics(self) -> Dict[str, Any]:
        """Get detailed noise characteristics of the current model."""
        if not self.noise_model:
            return {"noise_enabled": False}
        
        return {
            "noise_enabled": self.noise_model.enabled,
            "noise_model_name": self.noise_model.name,
            "device_name": self.noise_model.device_name,
            "calibration_date": self.noise_model.calibration_date,
            "coherence_params": {
                str(qid): {
                    "T1_us": params.T1.value,
                    "T2_us": params.T2.value,
                    "T2_star_us": params.T2_star
                }
                for qid, params in self.noise_model.coherence_params.items()
            },
            "gate_errors": {
                "single_qubit": self.noise_model.gate_errors.single_qubit_error,
                "two_qubit": self.noise_model.gate_errors.two_qubit_error,
                "measurement": self.noise_model.gate_errors.measurement_error,
                "reset": self.noise_model.gate_errors.reset_error
            },
            "readout_errors": {
                str(qid): {
                    "prob_0_given_1": error.prob_0_given_1,
                    "prob_1_given_0": error.prob_1_given_0
                }
                for qid, error in self.noise_model.readout_errors.items()
            }
        }
    
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