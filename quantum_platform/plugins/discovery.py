"""
Plugin Discovery

This module handles automatic discovery of plugins from standard locations
and provides mechanisms for registering plugin directories.
"""

import os
import ast
import logging
from typing import List, Dict, Set, Optional
from pathlib import Path
import json

from quantum_platform.plugins.loader import PluginLoader
from quantum_platform.plugins.base import Plugin, PluginLoadError


class PluginDiscovery:
    """
    Plugin discovery system for finding plugins automatically.
    
    This class scans standard directories and registered paths to discover
    available plugins without loading them.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._loader = PluginLoader()
        self._search_paths: Set[Path] = set()
        
        # Add default search paths
        self._add_default_paths()
    
    def _add_default_paths(self):
        """Add default plugin search paths."""
        # Platform plugins directory
        platform_dir = Path(__file__).parent.parent / "plugins"
        if platform_dir.exists():
            self._search_paths.add(platform_dir)
        
        # User plugins directory
        home = Path.home()
        user_plugins = home / ".quantum_platform" / "plugins"
        if user_plugins.exists():
            self._search_paths.add(user_plugins)
        
        # Current working directory plugins
        cwd_plugins = Path.cwd() / "plugins"
        if cwd_plugins.exists():
            self._search_paths.add(cwd_plugins)
        
        # Environment variable
        env_path = os.environ.get('QUANTUM_PLATFORM_PLUGINS')
        if env_path:
            env_plugins = Path(env_path)
            if env_plugins.exists():
                self._search_paths.add(env_plugins)
    
    def add_search_path(self, path: str) -> bool:
        """
        Add a directory to the plugin search paths.
        
        Args:
            path: Directory path to add
            
        Returns:
            True if path was added, False if invalid
        """
        plugin_path = Path(path)
        
        if not plugin_path.exists():
            self._logger.warning(f"Plugin path does not exist: {path}")
            return False
        
        if not plugin_path.is_dir():
            self._logger.warning(f"Plugin path is not a directory: {path}")
            return False
        
        self._search_paths.add(plugin_path)
        self._logger.info(f"Added plugin search path: {path}")
        return True
    
    def remove_search_path(self, path: str) -> bool:
        """
        Remove a directory from search paths.
        
        Args:
            path: Directory path to remove
            
        Returns:
            True if path was removed, False if not found
        """
        plugin_path = Path(path)
        
        if plugin_path in self._search_paths:
            self._search_paths.remove(plugin_path)
            self._logger.info(f"Removed plugin search path: {path}")
            return True
        
        return False
    
    def get_search_paths(self) -> List[str]:
        """Get list of current search paths."""
        return [str(path) for path in self._search_paths]
    
    def discover_plugins(self, recursive: bool = False) -> List[Dict[str, any]]:
        """
        Discover all available plugins in search paths.
        
        Args:
            recursive: Whether to search subdirectories
            
        Returns:
            List of plugin metadata dictionaries
        """
        discovered = []
        
        for search_path in self._search_paths:
            self._logger.debug(f"Searching for plugins in: {search_path}")
            
            try:
                path_plugins = self._discover_in_path(search_path, recursive)
                discovered.extend(path_plugins)
                self._logger.info(f"Found {len(path_plugins)} plugins in {search_path}")
                
            except Exception as e:
                self._logger.error(f"Error discovering plugins in {search_path}: {e}")
        
        return discovered
    
    def _discover_in_path(self, path: Path, recursive: bool) -> List[Dict[str, any]]:
        """Discover plugins in a specific path."""
        plugins = []
        pattern = "**/*.py" if recursive else "*.py"
        
        for file_path in path.glob(pattern):
            if file_path.name.startswith('_'):
                continue
            
            try:
                # Validate without loading
                validation = self._loader.validate_plugin_file(str(file_path))
                
                if validation["valid"]:
                    plugin_info = {
                        "file_path": str(file_path),
                        "relative_path": str(file_path.relative_to(path)),
                        "search_path": str(path),
                        "validation": validation,
                        "discovered_at": path
                    }
                    
                    # Try to extract metadata without loading
                    metadata = self._extract_metadata(file_path)
                    if metadata:
                        plugin_info.update(metadata)
                    
                    plugins.append(plugin_info)
                    
            except Exception as e:
                self._logger.debug(f"Skipping {file_path}: {e}")
        
        return plugins
    
    def _extract_metadata(self, file_path: Path) -> Optional[Dict[str, any]]:
        """
        Extract plugin metadata without loading the plugin.
        
        Args:
            file_path: Path to plugin file
            
        Returns:
            Metadata dictionary or None if extraction fails
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for plugin info definitions
            tree = ast.parse(content)
            
            metadata = {}
            
            # Look for assignments that might be plugin info
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.endswith('_info'):
                            # Try to extract basic info
                            if isinstance(node.value, ast.Call):
                                if (isinstance(node.value.func, ast.Name) and 
                                    node.value.func.id == 'PluginInfo'):
                                    
                                    # Extract arguments
                                    info = self._extract_plugin_info_args(node.value)
                                    if info:
                                        metadata.update(info)
            
            return metadata if metadata else None
            
        except Exception as e:
            self._logger.debug(f"Failed to extract metadata from {file_path}: {e}")
            return None
    
    def _extract_plugin_info_args(self, call_node: ast.Call) -> Optional[Dict[str, any]]:
        """Extract arguments from a PluginInfo constructor call."""
        info = {}
        
        try:
            # Extract positional arguments
            if len(call_node.args) >= 3:
                # name, version, description, plugin_type
                if isinstance(call_node.args[0], ast.Str):
                    info["name"] = call_node.args[0].s
                if isinstance(call_node.args[1], ast.Str):
                    info["version"] = call_node.args[1].s
                if isinstance(call_node.args[2], ast.Str):
                    info["description"] = call_node.args[2].s
            
            # Extract keyword arguments
            for keyword in call_node.keywords:
                if keyword.arg in ["author", "email", "license", "homepage"]:
                    if isinstance(keyword.value, ast.Str):
                        info[keyword.arg] = keyword.value.s
            
            return info if info else None
            
        except Exception as e:
            self._logger.debug(f"Failed to extract PluginInfo args: {e}")
            return None
    
    def load_discovered_plugins(self, filter_func: Optional[callable] = None) -> List[Plugin]:
        """
        Load all discovered plugins.
        
        Args:
            filter_func: Optional function to filter which plugins to load
            
        Returns:
            List of loaded plugin instances
        """
        discovered = self.discover_plugins()
        loaded_plugins = []
        
        for plugin_info in discovered:
            if filter_func and not filter_func(plugin_info):
                continue
            
            try:
                file_path = plugin_info["file_path"]
                plugins = self._loader.load_from_file(file_path)
                loaded_plugins.extend(plugins)
                self._logger.info(f"Loaded {len(plugins)} plugins from {file_path}")
                
            except PluginLoadError as e:
                self._logger.warning(f"Failed to load plugin {plugin_info.get('name', 'unknown')}: {e}")
        
        return loaded_plugins
    
    def save_discovery_cache(self, cache_file: str) -> bool:
        """
        Save discovery results to a cache file.
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            discovered = self.discover_plugins()
            
            # Convert Path objects to strings for JSON serialization
            cache_data = {
                "search_paths": [str(p) for p in self._search_paths],
                "plugins": []
            }
            
            for plugin_info in discovered:
                # Make a copy and ensure all values are JSON serializable
                cached_plugin = {}
                for key, value in plugin_info.items():
                    if isinstance(value, Path):
                        cached_plugin[key] = str(value)
                    elif isinstance(value, dict):
                        cached_plugin[key] = value
                    else:
                        cached_plugin[key] = str(value)
                
                cache_data["plugins"].append(cached_plugin)
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self._logger.info(f"Saved plugin discovery cache to {cache_file}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save discovery cache: {e}")
            return False
    
    def load_discovery_cache(self, cache_file: str) -> Optional[List[Dict[str, any]]]:
        """
        Load discovery results from cache file.
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            List of cached plugin info or None if load fails
        """
        try:
            if not Path(cache_file).exists():
                return None
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Validate cache format
            if "plugins" not in cache_data:
                self._logger.warning("Invalid cache file format")
                return None
            
            self._logger.info(f"Loaded plugin discovery cache from {cache_file}")
            return cache_data["plugins"]
            
        except Exception as e:
            self._logger.error(f"Failed to load discovery cache: {e}")
            return None
    
    def get_discovery_stats(self) -> Dict[str, any]:
        """Get statistics about plugin discovery."""
        discovered = self.discover_plugins()
        
        stats = {
            "search_paths": len(self._search_paths),
            "total_plugins": len(discovered),
            "by_path": {},
            "by_type": {}
        }
        
        # Count by search path
        for plugin_info in discovered:
            search_path = plugin_info.get("search_path", "unknown")
            stats["by_path"][search_path] = stats["by_path"].get(search_path, 0) + 1
        
        return stats 