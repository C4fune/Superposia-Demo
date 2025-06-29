"""
Base Plugin Interfaces

This module defines the abstract base classes and interfaces that all
plugins must implement, providing a consistent API for plugin development.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import uuid
from datetime import datetime

from quantum_platform.compiler.ir.circuit import QuantumCircuit
from quantum_platform.compiler.ir.operation import Operation
from quantum_platform.compiler.gates.base import Gate


class PluginType(Enum):
    """Types of plugins supported by the platform."""
    COMPILER_PASS = "compiler_pass"     # Optimization/transformation passes
    GATE = "gate"                       # Custom quantum gates  
    OPTIMIZER = "optimizer"             # Circuit optimization algorithms
    EXPORTER = "exporter"               # Output format exporters
    SIMULATOR = "simulator"             # Custom simulation backends
    LANGUAGE = "language"               # DSL language extensions
    HARDWARE = "hardware"               # Hardware provider integrations
    COMPLIANCE = "compliance"           # Compliance checking rules
    UTILITY = "utility"                 # General utility functions


@dataclass
class PluginInfo:
    """Metadata about a plugin."""
    
    # Basic identification
    name: str
    version: str
    description: str
    plugin_type: PluginType
    
    # Plugin details
    author: str = "Unknown"
    email: str = ""
    license: str = "Unknown"
    homepage: str = ""
    
    # Dependencies and compatibility
    platform_version_min: str = "0.1.0"
    platform_version_max: str = "1.0.0"
    dependencies: List[str] = field(default_factory=list)
    
    # Internal tracking
    plugin_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    loaded_at: Optional[datetime] = None
    enabled: bool = True
    
    def __post_init__(self):
        """Validate plugin info."""
        if not self.name:
            raise ValueError("Plugin name cannot be empty")
        if not self.version:
            raise ValueError("Plugin version cannot be empty")


class Plugin(ABC):
    """
    Abstract base class for all plugins.
    
    All plugins must inherit from this class and implement the required
    methods to integrate with the platform.
    """
    
    def __init__(self, info: PluginInfo):
        """
        Initialize the plugin.
        
        Args:
            info: Plugin metadata
        """
        self.info = info
        self._initialized = False
        self._active = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin.
        
        This method is called when the plugin is first loaded.
        It should perform any necessary setup or validation.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def activate(self) -> bool:
        """
        Activate the plugin.
        
        This method is called to enable the plugin's functionality.
        It should register any passes, gates, or other components.
        
        Returns:
            True if activation successful, False otherwise
        """
        pass
    
    @abstractmethod
    def deactivate(self) -> bool:
        """
        Deactivate the plugin.
        
        This method should clean up and unregister any components
        that were registered during activation.
        
        Returns:
            True if deactivation successful, False otherwise
        """
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized
    
    @property
    def is_active(self) -> bool:
        """Check if plugin is active."""
        return self._active
    
    def get_config_schema(self) -> Optional[Dict[str, Any]]:
        """
        Get the configuration schema for this plugin.
        
        Returns:
            JSON schema dictionary or None if no config needed
        """
        return None
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the plugin with given settings.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration successful, False otherwise
        """
        return True
    
    def validate_dependencies(self) -> List[str]:
        """
        Validate that all dependencies are available.
        
        Returns:
            List of missing dependencies (empty if all satisfied)
        """
        missing = []
        for dep in self.info.dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        return missing
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.info.name} v{self.info.version} ({self.info.plugin_type.value})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Plugin({self.info.name}, {self.info.version}, {self.info.plugin_type.value})"


class CompilerPassPlugin(Plugin):
    """
    Base class for compiler pass plugins.
    
    These plugins provide circuit transformation and optimization passes
    that can be integrated into the compilation pipeline.
    """
    
    def __init__(self, info: PluginInfo):
        super().__init__(info)
        if info.plugin_type != PluginType.COMPILER_PASS:
            raise ValueError("Plugin type must be COMPILER_PASS")
    
    @abstractmethod
    def get_pass_function(self) -> Callable[[QuantumCircuit], QuantumCircuit]:
        """
        Get the compiler pass function.
        
        Returns:
            Function that takes a QuantumCircuit and returns a transformed QuantumCircuit
        """
        pass
    
    def get_pass_priority(self) -> int:
        """
        Get the priority of this pass in the compilation pipeline.
        
        Lower numbers run earlier. Default is 50.
        
        Returns:
            Priority value (0-100)
        """
        return 50
    
    def get_pass_requirements(self) -> List[str]:
        """
        Get list of passes that must run before this one.
        
        Returns:
            List of pass names
        """
        return []


class GatePlugin(Plugin):
    """
    Base class for gate plugins.
    
    These plugins add new quantum gates to the platform's gate set.
    """
    
    def __init__(self, info: PluginInfo):
        super().__init__(info)
        if info.plugin_type != PluginType.GATE:
            raise ValueError("Plugin type must be GATE")
    
    @abstractmethod
    def get_gates(self) -> List[Gate]:
        """
        Get the gates provided by this plugin.
        
        Returns:
            List of Gate objects to register
        """
        pass
    
    def get_gate_decompositions(self) -> Dict[str, List[Operation]]:
        """
        Get decompositions for plugin gates into standard gates.
        
        Returns:
            Dictionary mapping gate names to lists of operations
        """
        return {}


class OptimizerPlugin(Plugin):
    """
    Base class for optimizer plugins.
    
    These plugins provide circuit optimization algorithms.
    """
    
    def __init__(self, info: PluginInfo):
        super().__init__(info)
        if info.plugin_type != PluginType.OPTIMIZER:
            raise ValueError("Plugin type must be OPTIMIZER")
    
    @abstractmethod
    def optimize(self, circuit: QuantumCircuit, **kwargs) -> QuantumCircuit:
        """
        Optimize a quantum circuit.
        
        Args:
            circuit: Circuit to optimize
            **kwargs: Optimization parameters
            
        Returns:
            Optimized circuit
        """
        pass
    
    def get_optimization_options(self) -> Dict[str, Any]:
        """
        Get available optimization options and their defaults.
        
        Returns:
            Dictionary of option names and default values
        """
        return {}


class ExporterPlugin(Plugin):
    """
    Base class for exporter plugins.
    
    These plugins add support for exporting circuits to new formats.
    """
    
    def __init__(self, info: PluginInfo):
        super().__init__(info)
        if info.plugin_type != PluginType.EXPORTER:
            raise ValueError("Plugin type must be EXPORTER")
    
    @abstractmethod
    def export(self, circuit: QuantumCircuit, **kwargs) -> str:
        """
        Export a circuit to the plugin's format.
        
        Args:
            circuit: Circuit to export
            **kwargs: Export options
            
        Returns:
            Exported circuit as string
        """
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """
        Get the file extension for this format.
        
        Returns:
            File extension (e.g., ".qasm", ".json")
        """
        pass
    
    def get_export_options(self) -> Dict[str, Any]:
        """
        Get available export options and their defaults.
        
        Returns:
            Dictionary of option names and default values
        """
        return {}


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginLoadError(PluginError):
    """Exception raised when a plugin fails to load."""
    pass


class PluginActivationError(PluginError):
    """Exception raised when a plugin fails to activate."""
    pass


class PluginDependencyError(PluginError):
    """Exception raised when plugin dependencies are not satisfied."""
    pass 