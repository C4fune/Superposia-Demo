"""
Hardware Abstraction Layer (HAL) for Quantum Devices

This module provides the core abstraction layer for interacting with
different quantum hardware providers through a common interface.
"""

import abc
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid

from ..compiler.ir.circuit import QuantumCircuit
from ..errors import (
    HardwareError, NetworkError, handle_errors,
    get_error_reporter, ComplianceError
)


class JobStatus(Enum):
    """Status of a quantum job."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class DeviceType(Enum):
    """Type of quantum device."""
    SUPERCONDUCTING = "superconducting"
    TRAPPED_ION = "trapped_ion"
    PHOTONIC = "photonic"
    NEUTRAL_ATOM = "neutral_atom"
    SIMULATOR = "simulator"
    HYBRID = "hybrid"


@dataclass
class JobHandle:
    """Handle for tracking quantum hardware jobs."""
    job_id: str
    backend_name: str
    provider_job_id: Optional[str] = None
    submitted_at: datetime = field(default_factory=datetime.now)
    circuit_id: Optional[str] = None
    shots: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.job_id:
            self.job_id = f"job_{uuid.uuid4().hex[:8]}"


@dataclass
class DeviceInfo:
    """Information about a quantum device."""
    name: str
    provider: str
    device_type: DeviceType
    num_qubits: int
    coupling_map: List[List[int]]
    basis_gates: List[str]
    max_shots: int = 100000
    max_experiments: int = 1000
    simulator: bool = False
    
    # Calibration and status
    operational: bool = True
    queue_length: Optional[int] = None
    avg_wait_time: Optional[float] = None  # in seconds
    
    # Error rates (if available)
    gate_error_rates: Optional[Dict[str, float]] = None
    readout_error_rates: Optional[List[float]] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareResult:
    """Result from quantum hardware execution."""
    job_handle: JobHandle
    status: JobStatus
    counts: Dict[str, int] = field(default_factory=dict)
    execution_time: Optional[float] = None
    queue_time: Optional[float] = None
    shots: int = 0
    
    # Error information
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # Metadata from provider
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Raw provider result (for debugging)
    raw_result: Optional[Any] = None


class QuantumHardwareBackend(abc.ABC):
    """Abstract base class for quantum hardware backends."""
    
    def __init__(self, name: str, provider: str):
        self.name = name
        self.provider = provider
        self._error_reporter = get_error_reporter()
        self._initialized = False
        self._device_info = None
        
    @abc.abstractmethod
    def initialize(self) -> bool:
        """Initialize the backend connection."""
        pass
    
    @abc.abstractmethod
    def get_device_info(self) -> DeviceInfo:
        """Get information about the device."""
        pass
    
    @abc.abstractmethod
    def submit_circuit(self, circuit: QuantumCircuit, shots: int = 1000, 
                      **kwargs) -> JobHandle:
        """Submit a quantum circuit for execution."""
        pass
    
    @abc.abstractmethod
    def get_job_status(self, job_handle: JobHandle) -> JobStatus:
        """Get the status of a submitted job."""
        pass
    
    @abc.abstractmethod
    def retrieve_results(self, job_handle: JobHandle) -> HardwareResult:
        """Retrieve results from a completed job."""
        pass
    
    @abc.abstractmethod
    def cancel_job(self, job_handle: JobHandle) -> bool:
        """Cancel a submitted job (if supported)."""
        pass
    
    def validate_circuit(self, circuit: QuantumCircuit) -> bool:
        """Validate that a circuit is compatible with this backend."""
        device_info = self.get_device_info()
        
        # Check qubit count
        if circuit.num_qubits > device_info.num_qubits:
            raise ComplianceError(
                f"Circuit requires {circuit.num_qubits} qubits, "
                f"but device only has {device_info.num_qubits}",
                user_message=f"Circuit too large for {device_info.name}"
            )
        
        # Check gate compatibility (basic check)
        unsupported_gates = []
        for operation in circuit.operations:
            gate_name = operation.__class__.__name__.lower()
            if gate_name not in [g.lower() for g in device_info.basis_gates]:
                unsupported_gates.append(gate_name)
        
        if unsupported_gates:
            raise ComplianceError(
                f"Circuit contains unsupported gates: {unsupported_gates}",
                user_message="Circuit needs transpilation for this device"
            )
        
        return True
    
    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transpile circuit for this backend (default implementation)."""
        # Import here to avoid circular imports
        from .transpilation import transpile_for_device
        return transpile_for_device(circuit, self.get_device_info())
    
    def submit_and_wait(self, circuit: QuantumCircuit, shots: int = 1000,
                       timeout: Optional[float] = None) -> HardwareResult:
        """Submit circuit and wait for completion."""
        job_handle = self.submit_circuit(circuit, shots)
        
        start_time = time.time()
        poll_interval = 5.0  # seconds
        
        while True:
            status = self.get_job_status(job_handle)
            
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, 
                         JobStatus.CANCELLED, JobStatus.TIMEOUT]:
                break
            
            if timeout and (time.time() - start_time) > timeout:
                try:
                    self.cancel_job(job_handle)
                except:
                    pass  # Cancellation might not be supported
                raise HardwareError(
                    f"Job {job_handle.job_id} timed out after {timeout} seconds",
                    user_message="Hardware job took too long to complete"
                )
            
            time.sleep(poll_interval)
        
        return self.retrieve_results(job_handle)
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get current queue information."""
        device_info = self.get_device_info()
        return {
            "queue_length": device_info.queue_length,
            "avg_wait_time": device_info.avg_wait_time,
            "operational": device_info.operational
        }
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', provider='{self.provider}')"


class BackendRegistry:
    """Registry for quantum hardware backends."""
    
    def __init__(self):
        self._backends: Dict[str, QuantumHardwareBackend] = {}
        self._backend_classes: Dict[str, type] = {}
        self._lock = threading.Lock()
    
    def register_backend(self, backend_class: type, name: Optional[str] = None):
        """Register a backend class."""
        with self._lock:
            backend_name = name or backend_class.__name__
            self._backend_classes[backend_name] = backend_class
    
    def create_backend(self, backend_type: str, name: str, **kwargs) -> QuantumHardwareBackend:
        """Create a backend instance."""
        with self._lock:
            if backend_type not in self._backend_classes:
                raise ValueError(f"Unknown backend type: {backend_type}")
            
            backend_class = self._backend_classes[backend_type]
            backend = backend_class(name=name, **kwargs)
            
            # Initialize if not already done
            if not backend._initialized:
                try:
                    backend.initialize()
                    backend._initialized = True
                except Exception as e:
                    raise HardwareError(
                        f"Failed to initialize backend {name}: {e}",
                        user_message=f"Could not connect to {name}"
                    )
            
            self._backends[name] = backend
            return backend
    
    def get_backend(self, name: str) -> Optional[QuantumHardwareBackend]:
        """Get a backend by name."""
        return self._backends.get(name)
    
    def list_backends(self) -> List[str]:
        """List all registered backend names."""
        return list(self._backends.keys())
    
    def list_backend_types(self) -> List[str]:
        """List all available backend types."""
        return list(self._backend_classes.keys())
    
    def remove_backend(self, name: str):
        """Remove a backend from registry."""
        with self._lock:
            if name in self._backends:
                del self._backends[name]


# Global registry instance
_backend_registry = BackendRegistry()


def get_backend_registry() -> BackendRegistry:
    """Get the global backend registry."""
    return _backend_registry


def register_backend(backend_class: type, name: Optional[str] = None):
    """Register a backend class with the global registry."""
    _backend_registry.register_backend(backend_class, name)


def get_backend(name: str) -> Optional[QuantumHardwareBackend]:
    """Get a backend by name from the global registry."""
    return _backend_registry.get_backend(name)


def list_available_backends() -> List[str]:
    """List all available backends."""
    return _backend_registry.list_backends()


def create_backend(backend_type: str, name: str, **kwargs) -> QuantumHardwareBackend:
    """Create a backend instance."""
    return _backend_registry.create_backend(backend_type, name, **kwargs) 