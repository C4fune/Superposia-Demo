"""
Plugin System for Quantum Platform

This module provides a comprehensive plugin architecture that allows users
to extend the quantum platform with custom functionality, including:
- Algorithm plugins
- Visualization plugins
- Backend plugins
- Tool plugins
- Educational plugins
"""

import os
import sys
import importlib
import inspect
from typing import Dict, List, Optional, Any, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
import json

from quantum_platform.observability.logging import get_logger


class PluginType(Enum):
    """Types of plugins."""
    ALGORITHM = "algorithm"
    VISUALIZATION = "visualization"
    BACKEND = "backend"
    COMPILER = "compiler"
    TOOL = "tool"
    EDUCATIONAL = "educational"
    UTILITY = "utility"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    entry_point: str
    dependencies: List[str] = field(default_factory=list)
    min_platform_version: str = "1.0.0"
    max_platform_version: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    license: str = "MIT"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "entry_point": self.entry_point,
            "dependencies": self.dependencies,
            "min_platform_version": self.min_platform_version,
            "max_platform_version": self.max_platform_version,
            "tags": self.tags,
            "license": self.license
        }


class PluginInterface(ABC):
    """Base interface for all plugins."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get plugin description."""
        pass
    
    def cleanup(self):
        """Cleanup plugin resources."""
        pass


class AlgorithmPlugin(PluginInterface):
    """Base class for algorithm plugins."""
    
    @abstractmethod
    def get_algorithms(self) -> Dict[str, Callable]:
        """Get available algorithms."""
        pass
    
    @abstractmethod
    def create_circuit(self, algorithm_name: str, **kwargs):
        """Create a circuit for the specified algorithm."""
        pass


class VisualizationPlugin(PluginInterface):
    """Base class for visualization plugins."""
    
    @abstractmethod
    def get_visualizations(self) -> Dict[str, Callable]:
        """Get available visualizations."""
        pass
    
    @abstractmethod
    def visualize(self, visualization_name: str, data: Any, **kwargs):
        """Create visualization."""
        pass


class BackendPlugin(PluginInterface):
    """Base class for backend plugins."""
    
    @abstractmethod
    def get_backend(self):
        """Get backend instance."""
        pass
    
    @abstractmethod
    def execute_circuit(self, circuit, shots: int = 1000):
        """Execute a circuit."""
        pass


class ToolPlugin(PluginInterface):
    """Base class for tool plugins."""
    
    @abstractmethod
    def get_tools(self) -> Dict[str, Callable]:
        """Get available tools."""
        pass
    
    @abstractmethod
    def execute_tool(self, tool_name: str, **kwargs):
        """Execute a tool."""
        pass


@dataclass
class LoadedPlugin:
    """Represents a loaded plugin."""
    metadata: PluginMetadata
    instance: PluginInterface
    module: Any
    is_active: bool = True
    
    def activate(self):
        """Activate the plugin."""
        self.is_active = True
    
    def deactivate(self):
        """Deactivate the plugin."""
        self.is_active = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "is_active": self.is_active,
            "class_name": self.instance.__class__.__name__
        }


class PluginManager:
    """
    Manages plugin loading, activation, and lifecycle.
    
    This class provides the main interface for plugin management,
    including discovery, loading, and execution of plugins.
    """
    
    def __init__(self, plugin_directories: List[str] = None):
        """
        Initialize plugin manager.
        
        Args:
            plugin_directories: List of directories to search for plugins
        """
        self.logger = get_logger("PluginManager")
        
        # Default plugin directories
        self.plugin_directories = plugin_directories or [
            "plugins",
            "quantum_platform/plugins",
            os.path.expanduser("~/.quantum_platform/plugins")
        ]
        
        # Loaded plugins
        self.loaded_plugins: Dict[str, LoadedPlugin] = {}
        self.plugin_registry: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
        
        # Initialize plugin directories
        self._initialize_plugin_directories()
        
        # Load plugins
        self._discover_and_load_plugins()
        
        self.logger.info(f"Plugin manager initialized with {len(self.loaded_plugins)} plugins")
    
    def _initialize_plugin_directories(self):
        """Initialize plugin directories."""
        for directory in self.plugin_directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _discover_and_load_plugins(self):
        """Discover and load all plugins."""
        for directory in self.plugin_directories:
            self._discover_plugins_in_directory(directory)
    
    def _discover_plugins_in_directory(self, directory: str):
        """Discover plugins in a specific directory."""
        directory_path = Path(directory)
        if not directory_path.exists():
            return
        
        for plugin_path in directory_path.iterdir():
            if plugin_path.is_dir():
                plugin_config_file = plugin_path / "plugin.json"
                if plugin_config_file.exists():
                    try:
                        self._load_plugin_from_directory(plugin_path)
                    except Exception as e:
                        self.logger.error(f"Failed to load plugin from {plugin_path}: {e}")
    
    def _load_plugin_from_directory(self, plugin_path: Path):
        """Load a plugin from a directory."""
        # Read plugin metadata
        plugin_config_file = plugin_path / "plugin.json"
        with open(plugin_config_file, 'r') as f:
            plugin_config = json.load(f)
        
        # Create metadata
        metadata = PluginMetadata(
            name=plugin_config["name"],
            version=plugin_config["version"],
            description=plugin_config["description"],
            author=plugin_config["author"],
            plugin_type=PluginType(plugin_config["plugin_type"]),
            entry_point=plugin_config["entry_point"],
            dependencies=plugin_config.get("dependencies", []),
            min_platform_version=plugin_config.get("min_platform_version", "1.0.0"),
            max_platform_version=plugin_config.get("max_platform_version"),
            tags=plugin_config.get("tags", []),
            license=plugin_config.get("license", "MIT")
        )
        
        # Load the plugin module
        sys.path.insert(0, str(plugin_path))
        try:
            module = importlib.import_module(metadata.entry_point)
            
            # Find the plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginInterface) and 
                    obj != PluginInterface):
                    plugin_class = obj
                    break
            
            if plugin_class is None:
                raise ValueError(f"No plugin class found in {metadata.entry_point}")
            
            # Create plugin instance
            plugin_instance = plugin_class()
            
            # Initialize the plugin
            if plugin_instance.initialize():
                loaded_plugin = LoadedPlugin(
                    metadata=metadata,
                    instance=plugin_instance,
                    module=module
                )
                
                self.loaded_plugins[metadata.name] = loaded_plugin
                self.plugin_registry[metadata.plugin_type].append(metadata.name)
                
                self.logger.info(f"Loaded plugin: {metadata.name} v{metadata.version}")
            else:
                self.logger.error(f"Failed to initialize plugin: {metadata.name}")
                
        except Exception as e:
            self.logger.error(f"Failed to load plugin {metadata.name}: {e}")
        finally:
            sys.path.remove(str(plugin_path))
    
    def get_plugins(self, plugin_type: Optional[PluginType] = None) -> List[LoadedPlugin]:
        """
        Get loaded plugins.
        
        Args:
            plugin_type: Filter by plugin type
            
        Returns:
            List of loaded plugins
        """
        plugins = list(self.loaded_plugins.values())
        
        if plugin_type:
            plugins = [p for p in plugins if p.metadata.plugin_type == plugin_type]
        
        return plugins
    
    def get_plugin(self, plugin_name: str) -> Optional[LoadedPlugin]:
        """Get a specific plugin by name."""
        return self.loaded_plugins.get(plugin_name)
    
    def activate_plugin(self, plugin_name: str) -> bool:
        """Activate a plugin."""
        plugin = self.loaded_plugins.get(plugin_name)
        if plugin:
            plugin.activate()
            self.logger.info(f"Activated plugin: {plugin_name}")
            return True
        return False
    
    def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate a plugin."""
        plugin = self.loaded_plugins.get(plugin_name)
        if plugin:
            plugin.deactivate()
            plugin.instance.cleanup()
            self.logger.info(f"Deactivated plugin: {plugin_name}")
            return True
        return False
    
    def execute_algorithm(self, plugin_name: str, algorithm_name: str, **kwargs):
        """Execute an algorithm from a plugin."""
        plugin = self.loaded_plugins.get(plugin_name)
        if not plugin or not plugin.is_active:
            raise ValueError(f"Plugin {plugin_name} not found or inactive")
        
        if not isinstance(plugin.instance, AlgorithmPlugin):
            raise ValueError(f"Plugin {plugin_name} is not an algorithm plugin")
        
        return plugin.instance.create_circuit(algorithm_name, **kwargs)
    
    def create_visualization(self, plugin_name: str, visualization_name: str, data: Any, **kwargs):
        """Create a visualization from a plugin."""
        plugin = self.loaded_plugins.get(plugin_name)
        if not plugin or not plugin.is_active:
            raise ValueError(f"Plugin {plugin_name} not found or inactive")
        
        if not isinstance(plugin.instance, VisualizationPlugin):
            raise ValueError(f"Plugin {plugin_name} is not a visualization plugin")
        
        return plugin.instance.visualize(visualization_name, data, **kwargs)
    
    def get_backend(self, plugin_name: str):
        """Get a backend from a plugin."""
        plugin = self.loaded_plugins.get(plugin_name)
        if not plugin or not plugin.is_active:
            raise ValueError(f"Plugin {plugin_name} not found or inactive")
        
        if not isinstance(plugin.instance, BackendPlugin):
            raise ValueError(f"Plugin {plugin_name} is not a backend plugin")
        
        return plugin.instance.get_backend()
    
    def execute_tool(self, plugin_name: str, tool_name: str, **kwargs):
        """Execute a tool from a plugin."""
        plugin = self.loaded_plugins.get(plugin_name)
        if not plugin or not plugin.is_active:
            raise ValueError(f"Plugin {plugin_name} not found or inactive")
        
        if not isinstance(plugin.instance, ToolPlugin):
            raise ValueError(f"Plugin {plugin_name} is not a tool plugin")
        
        return plugin.instance.execute_tool(tool_name, **kwargs)
    
    def list_available_algorithms(self) -> Dict[str, List[str]]:
        """List all available algorithms from plugins."""
        algorithms = {}
        
        for plugin_name, plugin in self.loaded_plugins.items():
            if plugin.is_active and isinstance(plugin.instance, AlgorithmPlugin):
                plugin_algorithms = plugin.instance.get_algorithms()
                algorithms[plugin_name] = list(plugin_algorithms.keys())
        
        return algorithms
    
    def list_available_visualizations(self) -> Dict[str, List[str]]:
        """List all available visualizations from plugins."""
        visualizations = {}
        
        for plugin_name, plugin in self.loaded_plugins.items():
            if plugin.is_active and isinstance(plugin.instance, VisualizationPlugin):
                plugin_visualizations = plugin.instance.get_visualizations()
                visualizations[plugin_name] = list(plugin_visualizations.keys())
        
        return visualizations
    
    def list_available_tools(self) -> Dict[str, List[str]]:
        """List all available tools from plugins."""
        tools = {}
        
        for plugin_name, plugin in self.loaded_plugins.items():
            if plugin.is_active and isinstance(plugin.instance, ToolPlugin):
                plugin_tools = plugin.instance.get_tools()
                tools[plugin_name] = list(plugin_tools.keys())
        
        return tools
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a plugin."""
        plugin = self.loaded_plugins.get(plugin_name)
        if plugin:
            return plugin.to_dict()
        return None
    
    def get_plugin_statistics(self) -> Dict[str, Any]:
        """Get plugin system statistics."""
        stats = {
            "total_plugins": len(self.loaded_plugins),
            "active_plugins": len([p for p in self.loaded_plugins.values() if p.is_active]),
            "plugins_by_type": {},
            "plugin_directories": self.plugin_directories
        }
        
        for plugin_type in PluginType:
            count = len([p for p in self.loaded_plugins.values() 
                        if p.metadata.plugin_type == plugin_type])
            stats["plugins_by_type"][plugin_type.value] = count
        
        return stats 