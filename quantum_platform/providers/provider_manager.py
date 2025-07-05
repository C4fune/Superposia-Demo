"""
Provider Manager

This module provides centralized management of multiple quantum hardware providers,
including device discovery, credential management, and seamless provider switching.
"""

import threading
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import os
from pathlib import Path

from ..hardware.hal import (
    QuantumHardwareBackend, DeviceInfo, DeviceType, JobStatus, 
    get_backend_registry, create_backend
)
from ..hardware.backends import LocalSimulatorBackend
from ..errors import HardwareError, ConfigurationError, NetworkError
from ..observability import get_logger

logger = get_logger(__name__)


class ProviderStatus(Enum):
    """Status of a quantum provider."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class ProviderInfo:
    """Information about a quantum provider."""
    name: str
    provider_type: str
    description: str
    website: str
    
    # Status and capabilities
    status: ProviderStatus = ProviderStatus.UNKNOWN
    supports_hardware: bool = True
    supports_simulation: bool = True
    
    # Device information
    devices: List[DeviceInfo] = field(default_factory=list)
    total_devices: int = 0
    available_devices: int = 0
    
    # Credential requirements
    requires_credentials: bool = True
    credential_types: List[str] = field(default_factory=list)
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """Configuration for a quantum provider."""
    name: str
    provider_type: str
    backend_class: str
    
    # Connection settings
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    timeout: float = 30.0
    
    # Credential settings
    credential_required: bool = True
    credential_env_var: Optional[str] = None
    
    # Device settings
    auto_discover_devices: bool = True
    device_refresh_interval: int = 300  # seconds
    
    # Feature flags
    supports_batching: bool = False
    supports_async: bool = True
    max_concurrent_jobs: int = 10
    
    # Configuration overrides
    config_overrides: Dict[str, Any] = field(default_factory=dict)


class ProviderManager:
    """Manager for multiple quantum hardware providers."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize provider manager.
        
        Args:
            config_path: Path to provider configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.providers: Dict[str, ProviderInfo] = {}
        self.backends: Dict[str, QuantumHardwareBackend] = {}
        self.active_provider: Optional[str] = None
        self.active_device: Optional[str] = None
        
        # Threading for device discovery
        self._discovery_thread: Optional[threading.Thread] = None
        self._discovery_running = False
        self._discovery_lock = threading.Lock()
        
        # Callbacks
        self._status_callbacks: List[Callable] = []
        self._device_callbacks: List[Callable] = []
        
        # Initialize built-in providers
        self._initialize_builtin_providers()
        
        # Load configuration
        self._load_configuration()
        
        logger.info("Provider manager initialized")
    
    def _get_default_config_path(self) -> str:
        """Get default configuration path."""
        home_dir = Path.home()
        config_dir = home_dir / ".quantum_platform"
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / "providers.json")
    
    def _initialize_builtin_providers(self):
        """Initialize built-in providers."""
        # Local simulator provider
        local_provider = ProviderInfo(
            name="local",
            provider_type="simulator",
            description="Local quantum simulator",
            website="https://quantum-platform.dev",
            status=ProviderStatus.AVAILABLE,
            supports_hardware=False,
            supports_simulation=True,
            requires_credentials=False,
            credential_types=[]
        )
        
        # Create local simulator device info
        local_device = DeviceInfo(
            name="local_simulator",
            provider="local",
            device_type=DeviceType.SIMULATOR,
            num_qubits=32,  # Configurable
            coupling_map=[],
            basis_gates=["x", "y", "z", "h", "s", "t", "cx", "cy", "cz", "rx", "ry", "rz"],
            max_shots=1000000,
            simulator=True,
            operational=True
        )
        
        local_provider.devices = [local_device]
        local_provider.total_devices = 1
        local_provider.available_devices = 1
        
        self.providers["local"] = local_provider
        
        # IBM provider configuration
        ibm_provider = ProviderInfo(
            name="ibm",
            provider_type="hardware",
            description="IBM Quantum Network",
            website="https://quantum-computing.ibm.com",
            status=ProviderStatus.UNKNOWN,
            supports_hardware=True,
            supports_simulation=True,
            requires_credentials=True,
            credential_types=["token"]
        )
        
        self.providers["ibm"] = ibm_provider
        
        # AWS provider configuration
        aws_provider = ProviderInfo(
            name="aws",
            provider_type="hardware",
            description="AWS Braket",
            website="https://aws.amazon.com/braket",
            status=ProviderStatus.UNKNOWN,
            supports_hardware=True,
            supports_simulation=True,
            requires_credentials=True,
            credential_types=["access_key", "secret_key", "region"]
        )
        
        self.providers["aws"] = aws_provider
        
        # Set local as default active provider
        self.active_provider = "local"
        self.active_device = "local_simulator"
    
    def _load_configuration(self):
        """Load provider configuration from file."""
        if not os.path.exists(self.config_path):
            self._save_configuration()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Load active provider settings
            self.active_provider = config.get("active_provider", "local")
            self.active_device = config.get("active_device", "local_simulator")
            
            # Load provider-specific configurations
            provider_configs = config.get("providers", {})
            for provider_name, provider_config in provider_configs.items():
                if provider_name in self.providers:
                    self.providers[provider_name].metadata.update(provider_config)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def _save_configuration(self):
        """Save current configuration to file."""
        try:
            config = {
                "active_provider": self.active_provider,
                "active_device": self.active_device,
                "providers": {
                    name: {
                        "last_updated": provider.last_updated.isoformat(),
                        "metadata": provider.metadata
                    }
                    for name, provider in self.providers.items()
                }
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.debug(f"Saved configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def list_providers(self) -> Dict[str, ProviderInfo]:
        """List all available providers."""
        return self.providers.copy()
    
    def get_provider_info(self, provider_name: str) -> Optional[ProviderInfo]:
        """Get information about a specific provider."""
        return self.providers.get(provider_name)
    
    def list_devices(self, provider_name: Optional[str] = None) -> List[DeviceInfo]:
        """
        List available devices.
        
        Args:
            provider_name: Filter by provider (None for all providers)
            
        Returns:
            List of available devices
        """
        devices = []
        
        if provider_name:
            provider = self.providers.get(provider_name)
            if provider:
                devices.extend(provider.devices)
        else:
            for provider in self.providers.values():
                devices.extend(provider.devices)
        
        return devices
    
    def get_device_info(self, device_name: str) -> Optional[DeviceInfo]:
        """Get information about a specific device."""
        for provider in self.providers.values():
            for device in provider.devices:
                if device.name == device_name:
                    return device
        return None
    
    def set_active_provider(self, provider_name: str, device_name: Optional[str] = None):
        """
        Set the active provider and optionally device.
        
        Args:
            provider_name: Name of the provider
            device_name: Name of the device (optional)
        """
        if provider_name not in self.providers:
            raise ConfigurationError(
                f"Unknown provider: {provider_name}",
                user_message=f"Provider '{provider_name}' is not available"
            )
        
        provider = self.providers[provider_name]
        
        # Check if provider is available
        if provider.status == ProviderStatus.UNAVAILABLE:
            raise ConfigurationError(
                f"Provider '{provider_name}' is not available",
                user_message=f"Cannot switch to unavailable provider"
            )
        
        # Validate device if specified
        if device_name:
            device = self.get_device_info(device_name)
            if not device or device.provider != provider_name:
                raise ConfigurationError(
                    f"Device '{device_name}' not found in provider '{provider_name}'",
                    user_message=f"Invalid device selection"
                )
        else:
            # Use first available device
            if provider.devices:
                device_name = provider.devices[0].name
        
        # Update active settings
        self.active_provider = provider_name
        self.active_device = device_name
        
        # Save configuration
        self._save_configuration()
        
        logger.info(f"Switched to provider '{provider_name}', device '{device_name}'")
        
        # Notify callbacks
        self._notify_status_callbacks("provider_switched", {
            "provider": provider_name,
            "device": device_name
        })
    
    def get_active_backend(self) -> Optional[QuantumHardwareBackend]:
        """Get the currently active backend."""
        if not self.active_provider or not self.active_device:
            return None
        
        backend_key = f"{self.active_provider}:{self.active_device}"
        
        if backend_key not in self.backends:
            self.backends[backend_key] = self._create_backend(
                self.active_provider, 
                self.active_device
            )
        
        return self.backends[backend_key]
    
    def _create_backend(self, provider_name: str, device_name: str) -> QuantumHardwareBackend:
        """Create a backend for the specified provider and device."""
        if provider_name == "local":
            backend = LocalSimulatorBackend(device_name, provider_name)
            backend.initialize()
            return backend
        
        elif provider_name == "ibm":
            # Import IBM backend
            from ..hardware.backends.ibm_backend import IBMQBackend
            
            # Get credentials
            token = self._get_provider_credentials(provider_name, "token")
            if not token:
                raise ConfigurationError(
                    "IBM Quantum token not configured",
                    user_message="Please configure IBM Quantum credentials"
                )
            
            backend = IBMQBackend(device_name, provider_name, token=token)
            backend.initialize()
            return backend
        
        elif provider_name == "aws":
            # AWS Braket backend would be implemented here
            raise NotImplementedError("AWS Braket backend not yet implemented")
        
        else:
            raise ConfigurationError(
                f"Unknown provider: {provider_name}",
                user_message=f"Provider '{provider_name}' is not supported"
            )
    
    def _get_provider_credentials(self, provider_name: str, credential_type: str) -> Optional[str]:
        """Get credentials for a provider."""
        # Check environment variables first
        env_var_map = {
            "ibm": {"token": "IBM_QUANTUM_TOKEN"},
            "aws": {
                "access_key": "AWS_ACCESS_KEY_ID",
                "secret_key": "AWS_SECRET_ACCESS_KEY",
                "region": "AWS_DEFAULT_REGION"
            }
        }
        
        if provider_name in env_var_map:
            env_var = env_var_map[provider_name].get(credential_type)
            if env_var:
                return os.getenv(env_var)
        
        # Check stored credentials
        provider = self.providers.get(provider_name)
        if provider and "credentials" in provider.metadata:
            return provider.metadata["credentials"].get(credential_type)
        
        return None
    
    def set_provider_credentials(self, provider_name: str, credentials: Dict[str, str]):
        """Set credentials for a provider."""
        if provider_name not in self.providers:
            raise ConfigurationError(f"Unknown provider: {provider_name}")
        
        provider = self.providers[provider_name]
        if "credentials" not in provider.metadata:
            provider.metadata["credentials"] = {}
        
        provider.metadata["credentials"].update(credentials)
        self._save_configuration()
        
        logger.info(f"Credentials updated for provider '{provider_name}'")
    
    def discover_devices(self, provider_name: Optional[str] = None):
        """
        Discover available devices from providers.
        
        Args:
            provider_name: Specific provider to discover (None for all)
        """
        providers_to_discover = [provider_name] if provider_name else list(self.providers.keys())
        
        for provider_name in providers_to_discover:
            if provider_name == "local":
                continue  # Local devices are static
            
            try:
                self._discover_provider_devices(provider_name)
            except Exception as e:
                logger.error(f"Failed to discover devices for {provider_name}: {e}")
    
    def _discover_provider_devices(self, provider_name: str):
        """Discover devices for a specific provider."""
        provider = self.providers.get(provider_name)
        if not provider:
            return
        
        provider.status = ProviderStatus.UNKNOWN
        
        if provider_name == "ibm":
            self._discover_ibm_devices(provider)
        elif provider_name == "aws":
            self._discover_aws_devices(provider)
        
        provider.last_updated = datetime.now()
        self._save_configuration()
    
    def _discover_ibm_devices(self, provider: ProviderInfo):
        """Discover IBM Quantum devices."""
        try:
            # This would use IBM's API to discover devices
            # For now, we'll add some mock devices
            
            token = self._get_provider_credentials("ibm", "token")
            if not token:
                provider.status = ProviderStatus.ERROR
                return
            
            # Mock IBM devices
            mock_devices = [
                DeviceInfo(
                    name="ibmq_qasm_simulator",
                    provider="ibm",
                    device_type=DeviceType.SIMULATOR,
                    num_qubits=32,
                    coupling_map=[],
                    basis_gates=["u1", "u2", "u3", "cx", "id"],
                    max_shots=100000,
                    simulator=True,
                    operational=True
                ),
                DeviceInfo(
                    name="ibm_lagos",
                    provider="ibm",
                    device_type=DeviceType.SUPERCONDUCTING,
                    num_qubits=7,
                    coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
                    basis_gates=["rz", "sx", "x", "cx", "id"],
                    max_shots=20000,
                    simulator=False,
                    operational=True,
                    queue_length=5,
                    avg_wait_time=300
                )
            ]
            
            provider.devices = mock_devices
            provider.total_devices = len(mock_devices)
            provider.available_devices = sum(1 for d in mock_devices if d.operational)
            provider.status = ProviderStatus.AVAILABLE
            
            logger.info(f"Discovered {len(mock_devices)} IBM devices")
            
        except Exception as e:
            logger.error(f"Failed to discover IBM devices: {e}")
            provider.status = ProviderStatus.ERROR
    
    def _discover_aws_devices(self, provider: ProviderInfo):
        """Discover AWS Braket devices."""
        try:
            # This would use AWS Braket API to discover devices
            # For now, we'll add some mock devices
            
            access_key = self._get_provider_credentials("aws", "access_key")
            if not access_key:
                provider.status = ProviderStatus.ERROR
                return
            
            # Mock AWS devices
            mock_devices = [
                DeviceInfo(
                    name="SV1",
                    provider="aws",
                    device_type=DeviceType.SIMULATOR,
                    num_qubits=34,
                    coupling_map=[],
                    basis_gates=["x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "cnot"],
                    max_shots=100000,
                    simulator=True,
                    operational=True
                ),
                DeviceInfo(
                    name="IonQ",
                    provider="aws",
                    device_type=DeviceType.TRAPPED_ION,
                    num_qubits=11,
                    coupling_map=[],  # All-to-all connectivity
                    basis_gates=["x", "y", "z", "rx", "ry", "rz", "cnot"],
                    max_shots=10000,
                    simulator=False,
                    operational=True,
                    queue_length=12,
                    avg_wait_time=1800
                )
            ]
            
            provider.devices = mock_devices
            provider.total_devices = len(mock_devices)
            provider.available_devices = sum(1 for d in mock_devices if d.operational)
            provider.status = ProviderStatus.AVAILABLE
            
            logger.info(f"Discovered {len(mock_devices)} AWS devices")
            
        except Exception as e:
            logger.error(f"Failed to discover AWS devices: {e}")
            provider.status = ProviderStatus.ERROR
    
    def start_device_discovery(self, interval: int = 300):
        """Start periodic device discovery."""
        if self._discovery_running:
            return
        
        self._discovery_running = True
        self._discovery_thread = threading.Thread(
            target=self._discovery_loop,
            args=(interval,),
            daemon=True
        )
        self._discovery_thread.start()
        
        logger.info("Started device discovery thread")
    
    def stop_device_discovery(self):
        """Stop device discovery."""
        self._discovery_running = False
        if self._discovery_thread:
            self._discovery_thread.join(timeout=5.0)
        
        logger.info("Stopped device discovery thread")
    
    def _discovery_loop(self, interval: int):
        """Main discovery loop."""
        while self._discovery_running:
            try:
                self.discover_devices()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def add_status_callback(self, callback: Callable):
        """Add a callback for status changes."""
        self._status_callbacks.append(callback)
    
    def add_device_callback(self, callback: Callable):
        """Add a callback for device changes."""
        self._device_callbacks.append(callback)
    
    def _notify_status_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Notify status callbacks."""
        for callback in self._status_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def _notify_device_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Notify device callbacks."""
        for callback in self._device_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in device callback: {e}")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of provider and device status."""
        summary = {
            "active_provider": self.active_provider,
            "active_device": self.active_device,
            "providers": {},
            "total_devices": 0,
            "available_devices": 0
        }
        
        for name, provider in self.providers.items():
            summary["providers"][name] = {
                "status": provider.status.value,
                "total_devices": provider.total_devices,
                "available_devices": provider.available_devices,
                "last_updated": provider.last_updated.isoformat()
            }
            
            summary["total_devices"] += provider.total_devices
            summary["available_devices"] += provider.available_devices
        
        return summary


# Global provider manager instance
_provider_manager = None


def get_provider_manager(config_path: Optional[str] = None) -> ProviderManager:
    """Get or create the global provider manager."""
    global _provider_manager
    
    if _provider_manager is None:
        _provider_manager = ProviderManager(config_path)
    
    return _provider_manager


def switch_provider(provider_name: str, device_name: Optional[str] = None):
    """Convenience function to switch providers."""
    manager = get_provider_manager()
    manager.set_active_provider(provider_name, device_name)


def get_active_backend() -> Optional[QuantumHardwareBackend]:
    """Get the currently active backend."""
    manager = get_provider_manager()
    return manager.get_active_backend()


def list_available_providers() -> Dict[str, ProviderInfo]:
    """List all available providers."""
    manager = get_provider_manager()
    return manager.list_providers()


def list_available_devices(provider_name: Optional[str] = None) -> List[DeviceInfo]:
    """List all available devices."""
    manager = get_provider_manager()
    return manager.list_devices(provider_name) 