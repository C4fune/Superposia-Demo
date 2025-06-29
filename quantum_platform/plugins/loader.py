"""
Plugin Loader

This module handles loading plugins from various sources including
Python modules, directories, and plugin files.
"""

import os
import sys
import importlib
import importlib.util
import logging
from typing import List, Optional, Dict, Any, Type
from pathlib import Path

from quantum_platform.plugins.base import Plugin, PluginInfo, PluginType, PluginLoadError


class PluginLoader:
    """
    Loader for plugins from various sources.
    
    This class can load plugins from Python modules, files, or directories
    and instantiate them for use by the plugin manager.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def load_from_module(self, module_name: str) -> List[Plugin]:
        """
        Load plugins from a Python module.
        
        Args:
            module_name: Name of module to load (e.g., 'my_plugins.optimizer')
            
        Returns:
            List of loaded plugin instances
            
        Raises:
            PluginLoadError: If module cannot be loaded or contains no plugins
        """
        try:
            module = importlib.import_module(module_name)
            return self._extract_plugins_from_module(module, module_name)
        except ImportError as e:
            raise PluginLoadError(f"Cannot import module {module_name}: {e}")
    
    def load_from_file(self, file_path: str) -> List[Plugin]:
        """
        Load plugins from a Python file.
        
        Args:
            file_path: Path to Python file containing plugins
            
        Returns:
            List of loaded plugin instances
            
        Raises:
            PluginLoadError: If file cannot be loaded or contains no plugins
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise PluginLoadError(f"Plugin file not found: {file_path}")
        
        if not file_path.suffix == '.py':
            raise PluginLoadError(f"Plugin file must be a .py file: {file_path}")
        
        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Cannot load spec from {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return self._extract_plugins_from_module(module, str(file_path))
            
        except Exception as e:
            raise PluginLoadError(f"Failed to load plugin file {file_path}: {e}")
    
    def load_from_directory(self, directory_path: str, recursive: bool = False) -> List[Plugin]:
        """
        Load all plugins from a directory.
        
        Args:
            directory_path: Path to directory containing plugin files
            recursive: Whether to search subdirectories
            
        Returns:
            List of loaded plugin instances
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise PluginLoadError(f"Plugin directory not found: {directory}")
        
        if not directory.is_dir():
            raise PluginLoadError(f"Path is not a directory: {directory}")
        
        plugins = []
        pattern = "**/*.py" if recursive else "*.py"
        
        for file_path in directory.glob(pattern):
            if file_path.name.startswith('_'):
                continue  # Skip private files
            
            try:
                file_plugins = self.load_from_file(str(file_path))
                plugins.extend(file_plugins)
                self._logger.info(f"Loaded {len(file_plugins)} plugins from {file_path}")
            except PluginLoadError as e:
                self._logger.warning(f"Failed to load plugins from {file_path}: {e}")
        
        return plugins
    
    def _extract_plugins_from_module(self, module: Any, source_name: str) -> List[Plugin]:
        """
        Extract plugin instances from a loaded module.
        
        Args:
            module: Loaded Python module
            source_name: Name/path of the source for logging
            
        Returns:
            List of plugin instances found in the module
        """
        plugins = []
        
        # Look for plugin instances
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
            
            attr = getattr(module, attr_name)
            
            # Check if it's a plugin instance
            if isinstance(attr, Plugin):
                plugins.append(attr)
                self._logger.debug(f"Found plugin instance: {attr.info.name}")
                continue
            
            # Check if it's a plugin class
            if (isinstance(attr, type) and 
                issubclass(attr, Plugin) and 
                attr != Plugin):
                
                # Try to instantiate with default parameters
                try:
                    # Look for plugin_info in the module
                    info_name = f"{attr_name.lower()}_info"
                    if hasattr(module, info_name):
                        plugin_info = getattr(module, info_name)
                        plugin_instance = attr(plugin_info)
                        plugins.append(plugin_instance)
                        self._logger.debug(f"Instantiated plugin class: {attr_name}")
                    else:
                        self._logger.warning(
                            f"Plugin class {attr_name} found but no {info_name} defined"
                        )
                except Exception as e:
                    self._logger.warning(f"Failed to instantiate plugin {attr_name}: {e}")
        
        # Look for a get_plugins() function
        if hasattr(module, 'get_plugins'):
            try:
                get_plugins = getattr(module, 'get_plugins')
                if callable(get_plugins):
                    additional_plugins = get_plugins()
                    if isinstance(additional_plugins, list):
                        plugins.extend(additional_plugins)
                        self._logger.debug(f"Got {len(additional_plugins)} plugins from get_plugins()")
            except Exception as e:
                self._logger.warning(f"Failed to call get_plugins() in {source_name}: {e}")
        
        if not plugins:
            self._logger.warning(f"No plugins found in {source_name}")
        
        return plugins
    
    def validate_plugin_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a plugin file without loading it.
        
        Args:
            file_path: Path to plugin file
            
        Returns:
            Validation results dictionary
        """
        result = {
            "valid": False,
            "file_exists": False,
            "is_python": False,
            "syntax_valid": False,
            "has_plugins": False,
            "errors": []
        }
        
        file_path = Path(file_path)
        
        # Check file existence
        if not file_path.exists():
            result["errors"].append(f"File not found: {file_path}")
            return result
        result["file_exists"] = True
        
        # Check if Python file
        if file_path.suffix != '.py':
            result["errors"].append("File must be a .py file")
            return result
        result["is_python"] = True
        
        # Check syntax
        try:
            with open(file_path, 'r') as f:
                source = f.read()
            compile(source, str(file_path), 'exec')
            result["syntax_valid"] = True
        except SyntaxError as e:
            result["errors"].append(f"Syntax error: {e}")
            return result
        except Exception as e:
            result["errors"].append(f"Error reading file: {e}")
            return result
        
        # Try to check for plugins without fully loading
        try:
            # Look for class definitions that might be plugins
            import ast
            tree = ast.parse(source)
            
            has_plugin_classes = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if class inherits from Plugin
                    for base in node.bases:
                        if (isinstance(base, ast.Name) and 
                            base.id in ['Plugin', 'CompilerPassPlugin', 'GatePlugin', 
                                       'OptimizerPlugin', 'ExporterPlugin']):
                            has_plugin_classes = True
                            break
                        elif isinstance(base, ast.Attribute):
                            if base.attr in ['Plugin', 'CompilerPassPlugin', 'GatePlugin',
                                           'OptimizerPlugin', 'ExporterPlugin']:
                                has_plugin_classes = True
                                break
            
            result["has_plugins"] = has_plugin_classes
            result["valid"] = has_plugin_classes
            
        except Exception as e:
            result["errors"].append(f"Error analyzing file: {e}")
        
        return result 