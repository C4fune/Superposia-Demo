"""
IBM Quantum Backend

This backend provides integration with IBM Quantum services through their API.
Supports both simulators and real hardware devices.
"""

import os
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...compiler.ir.circuit import QuantumCircuit
from ...errors import HardwareError, NetworkError, handle_errors, ComplianceError
from ..hal import (
    QuantumHardwareBackend, JobHandle, JobStatus, DeviceInfo, 
    HardwareResult, DeviceType
)


class IBMQBackend(QuantumHardwareBackend):
    """IBM Quantum backend implementation."""
    
    def __init__(self, name: str, device_name: str = "ibm_qasm_simulator", 
                 token: Optional[str] = None, hub: str = "ibm-q", 
                 group: str = "open", project: str = "main"):
        super().__init__(name, "IBM")
        self.device_name = device_name
        self.token = token or os.getenv("IBM_QUANTUM_TOKEN")
        self.hub = hub
        self.group = group
        self.project = project
        
        # IBM Quantum service components (imported lazily)
        self._service = None
        self._backend = None
        self._device_info_cache = None
        
    def initialize(self) -> bool:
        """Initialize IBM Quantum backend."""
        if not self.token:
            raise HardwareError(
                "IBM Quantum token not provided",
                user_message="Please set IBM_QUANTUM_TOKEN environment variable"
            )
        
        try:
            # Try to import qiskit-ibm-runtime (preferred) or qiskit-ibm-provider
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                self._service = QiskitRuntimeService(token=self.token)
                self._backend = self._service.backend(self.device_name)
                self._api_type = "runtime"
            except ImportError:
                try:
                    from qiskit_ibm_provider import IBMProvider
                    provider = IBMProvider(token=self.token)
                    self._backend = provider.get_backend(self.device_name)
                    self._api_type = "provider"
                except ImportError:
                    raise HardwareError(
                        "IBM Quantum libraries not installed",
                        user_message="Please install qiskit-ibm-runtime or qiskit-ibm-provider"
                    )
            
            # Test connection
            _ = self._backend.configuration()
            self._initialized = True
            return True
            
        except Exception as e:
            raise HardwareError(
                f"Failed to initialize IBM backend: {e}",
                user_message=f"Could not connect to IBM Quantum device {self.device_name}"
            )
    
    def get_device_info(self) -> DeviceInfo:
        """Get IBM device information."""
        if not self._initialized:
            self.initialize()
        
        if self._device_info_cache is None:
            try:
                config = self._backend.configuration()
                
                # Determine device type based on backend name
                device_type = DeviceType.SIMULATOR
                if not config.simulator:
                    if "superconducting" in config.backend_name.lower():
                        device_type = DeviceType.SUPERCONDUCTING
                    elif "ion" in config.backend_name.lower():
                        device_type = DeviceType.TRAPPED_ION
                
                # Get coupling map
                coupling_map = []
                if hasattr(config, 'coupling_map') and config.coupling_map:
                    coupling_map = config.coupling_map
                
                # Get basis gates
                basis_gates = getattr(config, 'basis_gates', [])
                
                # Get queue information if available
                queue_length = None
                avg_wait_time = None
                operational = True
                
                try:
                    if hasattr(self._backend, 'status'):
                        status = self._backend.status()
                        queue_length = getattr(status, 'pending_jobs', None)
                        operational = getattr(status, 'operational', True)
                except:
                    pass  # Status might not be available
                
                # Get error rates if available
                gate_error_rates = None
                readout_error_rates = None
                
                try:
                    if hasattr(self._backend, 'properties'):
                        props = self._backend.properties()
                        if props:
                            # Extract gate error rates
                            gate_error_rates = {}
                            for gate in props.gates:
                                if gate.gate and gate.parameters:
                                    for param in gate.parameters:
                                        if param.name == 'gate_error':
                                            gate_error_rates[gate.gate] = param.value
                            
                            # Extract readout error rates
                            readout_error_rates = []
                            for qubit in props.qubits:
                                for param in qubit:
                                    if param.name == 'readout_error':
                                        readout_error_rates.append(param.value)
                except:
                    pass  # Properties might not be available
                
                self._device_info_cache = DeviceInfo(
                    name=self.device_name,
                    provider="IBM",
                    device_type=device_type,
                    num_qubits=config.n_qubits,
                    coupling_map=coupling_map,
                    basis_gates=basis_gates,
                    max_shots=getattr(config, 'max_shots', 100000),
                    max_experiments=getattr(config, 'max_experiments', 1000),
                    simulator=config.simulator,
                    operational=operational,
                    queue_length=queue_length,
                    avg_wait_time=avg_wait_time,
                    gate_error_rates=gate_error_rates,
                    readout_error_rates=readout_error_rates,
                    metadata={
                        "backend_name": config.backend_name,
                        "backend_version": getattr(config, 'backend_version', ''),
                        "api_type": self._api_type
                    }
                )
                
            except Exception as e:
                raise HardwareError(
                    f"Failed to get device info: {e}",
                    user_message=f"Could not retrieve information for {self.device_name}"
                )
        
        return self._device_info_cache
    
    @handle_errors
    def submit_circuit(self, circuit: QuantumCircuit, shots: int = 1000, 
                      **kwargs) -> JobHandle:
        """Submit circuit to IBM Quantum."""
        if not self._initialized:
            self.initialize()
        
        # Validate circuit
        self.validate_circuit(circuit)
        
        try:
            # Convert our circuit to Qiskit format
            qiskit_circuit = self._convert_to_qiskit(circuit)
            
            # Submit job
            if self._api_type == "runtime":
                # Use Qiskit Runtime
                from qiskit_ibm_runtime import Sampler
                sampler = Sampler(backend=self._backend)
                job = sampler.run([qiskit_circuit], shots=shots)
            else:
                # Use traditional provider
                job = self._backend.run(qiskit_circuit, shots=shots, **kwargs)
            
            # Create job handle
            job_handle = JobHandle(
                job_id=f"ibm_{int(time.time() * 1000)}_{id(circuit)}",
                backend_name=self.name,
                provider_job_id=job.job_id(),
                circuit_id=getattr(circuit, 'name', None),
                shots=shots,
                metadata={
                    "backend": self.device_name,
                    "api_type": self._api_type,
                    **kwargs
                }
            )
            
            return job_handle
            
        except Exception as e:
            raise HardwareError(
                f"Failed to submit circuit to IBM: {e}",
                user_message=f"Could not submit job to {self.device_name}"
            )
    
    def _convert_to_qiskit(self, circuit: QuantumCircuit):
        """Convert our circuit to Qiskit format."""
        try:
            from qiskit import QuantumCircuit as QiskitCircuit, ClassicalRegister
            
            # Create Qiskit circuit
            qc = QiskitCircuit(circuit.num_qubits, circuit.num_qubits)
            
            # Add operations
            for operation in circuit.operations:
                gate_name = operation.__class__.__name__.lower()
                
                if gate_name == 'h':
                    qc.h(operation.targets[0].id)
                elif gate_name == 'x':
                    qc.x(operation.targets[0].id)
                elif gate_name == 'y':
                    qc.y(operation.targets[0].id)
                elif gate_name == 'z':
                    qc.z(operation.targets[0].id)
                elif gate_name == 'cnot':
                    qc.cx(operation.targets[0].id, operation.targets[1].id)
                elif gate_name == 'rx':
                    angle = list(operation.parameters.values())[0]
                    qc.rx(angle, operation.targets[0].id)
                elif gate_name == 'ry':
                    angle = list(operation.parameters.values())[0]
                    qc.ry(angle, operation.targets[0].id)
                elif gate_name == 'rz':
                    angle = list(operation.parameters.values())[0]
                    qc.rz(angle, operation.targets[0].id)
                elif gate_name == 'measure':
                    for i, target in enumerate(operation.targets):
                        qc.measure(target.id, target.id)
                else:
                    # Try to handle other gates generically
                    pass
            
            # Add measurements if not present
            if not any(op.__class__.__name__.lower() == 'measure' 
                      for op in circuit.operations):
                qc.measure_all()
            
            return qc
            
        except Exception as e:
            raise HardwareError(
                f"Failed to convert circuit to Qiskit format: {e}",
                user_message="Circuit conversion failed"
            )
    
    def get_job_status(self, job_handle: JobHandle) -> JobStatus:
        """Get IBM job status."""
        if not job_handle.provider_job_id:
            return JobStatus.FAILED
        
        try:
            if self._api_type == "runtime":
                job = self._service.job(job_handle.provider_job_id)
            else:
                job = self._backend.job(job_handle.provider_job_id)
            
            status = job.status()
            
            # Map IBM status to our JobStatus
            status_map = {
                'INITIALIZING': JobStatus.PENDING,
                'QUEUED': JobStatus.QUEUED,
                'VALIDATING': JobStatus.PENDING,
                'RUNNING': JobStatus.RUNNING,
                'DONE': JobStatus.COMPLETED,
                'ERROR': JobStatus.FAILED,
                'CANCELLED': JobStatus.CANCELLED
            }
            
            return status_map.get(status.name, JobStatus.FAILED)
            
        except Exception as e:
            self._error_reporter.collect_error(
                NetworkError(f"Failed to get job status: {e}")
            )
            return JobStatus.FAILED
    
    def retrieve_results(self, job_handle: JobHandle) -> HardwareResult:
        """Retrieve results from IBM Quantum."""
        if not job_handle.provider_job_id:
            return HardwareResult(
                job_handle=job_handle,
                status=JobStatus.FAILED,
                error_message="No provider job ID",
                error_code="NO_JOB_ID"
            )
        
        try:
            if self._api_type == "runtime":
                job = self._service.job(job_handle.provider_job_id)
            else:
                job = self._backend.job(job_handle.provider_job_id)
            
            result = job.result()
            
            # Extract counts
            counts = {}
            if hasattr(result, 'get_counts'):
                counts = result.get_counts()
            elif hasattr(result, 'quasi_dists'):
                # Runtime results
                quasi_dist = result.quasi_dists[0]
                counts = {format(k, f'0{job_handle.shots}b'): v 
                         for k, v in quasi_dist.items()}
            
            # Get execution time
            execution_time = None
            if hasattr(result, 'time_taken'):
                execution_time = result.time_taken * 1000  # Convert to ms
            
            return HardwareResult(
                job_handle=job_handle,
                status=JobStatus.COMPLETED,
                counts=counts,
                execution_time=execution_time,
                shots=job_handle.shots,
                metadata={
                    "backend": self.device_name,
                    "job_id": job_handle.provider_job_id,
                    "api_type": self._api_type
                },
                raw_result=result
            )
            
        except Exception as e:
            return HardwareResult(
                job_handle=job_handle,
                status=JobStatus.FAILED,
                error_message=str(e),
                error_code="RESULT_RETRIEVAL_ERROR"
            )
    
    def cancel_job(self, job_handle: JobHandle) -> bool:
        """Cancel IBM Quantum job."""
        if not job_handle.provider_job_id:
            return False
        
        try:
            if self._api_type == "runtime":
                job = self._service.job(job_handle.provider_job_id)
            else:
                job = self._backend.job(job_handle.provider_job_id)
            
            job.cancel()
            return True
            
        except Exception as e:
            self._error_reporter.collect_error(
                HardwareError(f"Failed to cancel job: {e}")
            )
            return False 