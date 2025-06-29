"""Plugin Manager - Core plugin management system"""

import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict

from quantum_platform.plugins.base import Plugin, PluginType

class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._by_type: Dict[PluginType, List[Plugin]] = defaultdict(list)
        self._logger = logging.getLogger(__name__)
    
    def register(self, plugin: Plugin) -> bool:
        """Register a plugin."""
        plugin_name = plugin.info.name
        if plugin_name in self._plugins:
            return False
        
        self._plugins[plugin_name] = plugin
        self._by_type[plugin.info.plugin_type].append(plugin)
        self._logger.info(f"Registered plugin: {plugin}")
        return True
    
    def get(self, name: str) -> Optional[Plugin]:
        """Get plugin by name."""
        return self._plugins.get(name)
    
    def get_by_type(self, plugin_type: PluginType) -> List[Plugin]:
        """Get plugins by type."""
        return self._by_type[plugin_type].copy()
    
    def list_all(self) -> List[Plugin]:
        """List all plugins."""
        return list(self._plugins.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_plugins": len(self._plugins),
            "plugins_by_type": {str(plugin_type): len(plugins) 
                              for plugin_type, plugins in self._by_type.items()},
            "plugin_names": list(self._plugins.keys())
        }

class PluginManager:
    """Main plugin manager."""
    
    def __init__(self):
        self.registry = PluginRegistry()
        self._logger = logging.getLogger(__name__)
    
    def load_plugin(self, plugin: Plugin, auto_activate: bool = True) -> bool:
        """Load and optionally activate a plugin."""
        try:
            if not plugin.initialize():
                return False
            plugin._initialized = True
            
            if not self.registry.register(plugin):
                return False
            
            if auto_activate:
                if not plugin.activate():
                    return False
                plugin._active = True
            
            self._logger.info(f"Loaded plugin: {plugin}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to load plugin: {e}")
            return False
    
    def get_active_plugins(self, plugin_type: Optional[PluginType] = None) -> List[Plugin]:
        """Get active plugins."""
        if plugin_type:
            plugins = self.registry.get_by_type(plugin_type)
        else:
            plugins = self.registry.list_all()
        return [p for p in plugins if p.is_active]
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "plugin_manager_version": "1.0.0",
            "total_plugins": len(self.registry._plugins),
            "active_plugins": len(self.get_active_plugins()),
            "supported_plugin_types": [str(pt) for pt in PluginType]
        }
    
    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get plugin information by name."""
        plugin = self.registry.get(name)
        if plugin:
            return {
                "name": plugin.info.name,
                "version": plugin.info.version,
                "description": plugin.info.description,
                "type": str(plugin.info.plugin_type),
                "author": plugin.info.author,
                "initialized": plugin.is_initialized,
                "active": plugin.is_active
            }
        return None
    
    def activate_plugin(self, name: str) -> bool:
        """Activate a plugin by name."""
        plugin = self.registry.get(name)
        if plugin and plugin.is_initialized:
            try:
                if plugin.activate():
                    plugin._active = True
                    self._logger.info(f"Activated plugin: {name}")
                    return True
            except Exception as e:
                self._logger.error(f"Failed to activate plugin {name}: {e}")
        return False
    
    def deactivate_plugin(self, name: str) -> bool:
        """Deactivate a plugin by name."""
        plugin = self.registry.get(name)
        if plugin and plugin.is_active:
            try:
                if plugin.deactivate():
                    plugin._active = False
                    self._logger.info(f"Deactivated plugin: {name}")
                    return True
            except Exception as e:
                self._logger.error(f"Failed to deactivate plugin {name}: {e}")
        return False
    
    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin by name."""
        plugin = self.registry.get(name)
        if plugin:
            try:
                # Deactivate first if active
                if plugin.is_active:
                    plugin.deactivate()
                    plugin._active = False
                
                # Remove from registry
                if name in self.registry._plugins:
                    del self.registry._plugins[name]
                
                # Remove from type index
                if plugin.info.plugin_type in self.registry._by_type:
                    self.registry._by_type[plugin.info.plugin_type].remove(plugin)
                
                # Mark as uninitialized
                plugin._initialized = False
                
                self._logger.info(f"Unloaded plugin: {name}")
                return True
            except Exception as e:
                self._logger.error(f"Failed to unload plugin {name}: {e}")
        return False
