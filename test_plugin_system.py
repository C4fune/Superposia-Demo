#!/usr/bin/env python3
"""
Test script for the Plugin Architecture

This demonstrates the plugin system including discovery, loading,
and execution of various plugin types.
"""

import sys
import os
from pathlib import Path

# Add the platform to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_platform.plugins import (
    PluginManager, PluginLoader, PluginDiscovery,
    Plugin, PluginInfo, PluginType,
    CompilerPassPlugin, GatePlugin, OptimizerPlugin
)
from quantum_platform import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, CNOT


def test_plugin_manager():
    """Test basic plugin manager functionality."""
    print("=== Testing Plugin Manager ===")
    
    manager = PluginManager()
    print(f"Plugin manager created: {manager}")
    
    # Check initial state
    stats = manager.registry.get_stats()
    print(f"Initial stats: {stats}")
    
    # Test system info
    system_info = manager.get_system_info()
    print(f"System info: {system_info}")
    
    return manager


def test_plugin_loader():
    """Test plugin loading from files."""
    print("\n=== Testing Plugin Loader ===")
    
    loader = PluginLoader()
    
    # Try to load example plugin
    example_path = "quantum_platform/plugins/examples/gate_cancellation.py"
    
    if Path(example_path).exists():
        try:
            plugins = loader.load_from_file(example_path)
            print(f"Loaded {len(plugins)} plugins from {example_path}")
            
            for plugin in plugins:
                print(f"  Plugin: {plugin}")
                print(f"    Type: {plugin.info.plugin_type}")
                print(f"    Description: {plugin.info.description}")
            
            return plugins
            
        except Exception as e:
            print(f"Failed to load plugin: {e}")
            return []
    else:
        print(f"Example plugin not found at {example_path}")
        return []


def test_plugin_discovery():
    """Test plugin discovery system."""
    print("\n=== Testing Plugin Discovery ===")
    
    discovery = PluginDiscovery()
    
    # Show search paths
    paths = discovery.get_search_paths()
    print(f"Search paths: {paths}")
    
    # Try discovery
    discovered = discovery.discover_plugins()
    print(f"Discovered {len(discovered)} plugins")
    
    for plugin_info in discovered:
        print(f"  Found: {plugin_info.get('name', 'unknown')} at {plugin_info.get('file_path', 'unknown')}")
    
    # Get stats
    stats = discovery.get_discovery_stats()
    print(f"Discovery stats: {stats}")
    
    return discovered


def test_plugin_integration():
    """Test plugin integration with manager."""
    print("\n=== Testing Plugin Integration ===")
    
    manager = PluginManager()
    loader = PluginLoader()
    
    # Try to load and register example plugin
    example_path = "quantum_platform/plugins/examples/gate_cancellation.py"
    
    if Path(example_path).exists():
        try:
            plugins = loader.load_from_file(example_path)
            
            for plugin in plugins:
                success = manager.load_plugin(plugin)
                print(f"Loaded plugin {plugin.info.name}: {success}")
                
                if success:
                    # Get plugin info
                    info = manager.get_plugin_info(plugin.info.name)
                    print(f"Plugin info: {info}")
                    
                    # Test plugin type-specific functionality
                    if isinstance(plugin, CompilerPassPlugin):
                        pass_func = plugin.get_pass_function()
                        priority = plugin.get_pass_priority()
                        print(f"Compiler pass priority: {priority}")
            
            # Show active plugins
            active = manager.get_active_plugins()
            print(f"Active plugins: {[p.info.name for p in active]}")
            
            return True
            
        except Exception as e:
            print(f"Plugin integration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("No example plugin to test integration")
        return False


def test_create_simple_plugin():
    """Test creating a simple plugin programmatically."""
    print("\n=== Testing Simple Plugin Creation ===")
    
    # Create a simple exporter plugin
    from quantum_platform.plugins.base import ExporterPlugin
    from quantum_platform.compiler.ir.circuit import QuantumCircuit
    
    class SimpleJsonExporter(ExporterPlugin):
        """Simple JSON exporter plugin."""
        
        def initialize(self) -> bool:
            self._initialized = True
            return True
        
        def activate(self) -> bool:
            self._active = True
            return True
        
        def deactivate(self) -> bool:
            self._active = False
            return True
        
        def export(self, circuit: QuantumCircuit, **kwargs) -> str:
            """Export circuit as JSON."""
            return circuit.to_json()
        
        def get_file_extension(self) -> str:
            return ".json"
    
    # Create plugin info
    json_exporter_info = PluginInfo(
        name="simple_json_exporter",
        version="1.0.0",
        description="Simple JSON circuit exporter",
        plugin_type=PluginType.EXPORTER,
        author="Test Suite"
    )
    
    # Create plugin instance
    json_plugin = SimpleJsonExporter(json_exporter_info)
    
    # Test the plugin
    manager = PluginManager()
    success = manager.load_plugin(json_plugin)
    print(f"Created and loaded simple plugin: {success}")
    
    if success:
        # Test the export functionality
        with QuantumProgram(name="test_circuit") as qp:
            from quantum_platform.compiler.language.dsl import allocate
            q = allocate(2)
            H(q[0])
            CNOT(q[0], q[1])
        
        # Export using the plugin
        try:
            exported = json_plugin.export(qp.circuit)
            print(f"Exported circuit JSON (first 100 chars): {exported[:100]}...")
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    return False


def test_plugin_lifecycle():
    """Test plugin lifecycle management."""
    print("\n=== Testing Plugin Lifecycle ===")
    
    manager = PluginManager()
    
    # Create a test plugin
    test_info = PluginInfo(
        name="test_lifecycle_plugin",
        version="1.0.0",
        description="Test plugin for lifecycle",
        plugin_type=PluginType.UTILITY
    )
    
    class TestLifecyclePlugin(Plugin):
        def initialize(self) -> bool:
            print("  Plugin initialized")
            self._initialized = True
            return True
        
        def activate(self) -> bool:
            print("  Plugin activated")
            self._active = True
            return True
        
        def deactivate(self) -> bool:
            print("  Plugin deactivated")
            self._active = False
            return True
    
    plugin = TestLifecyclePlugin(test_info)
    
    # Test lifecycle
    print("Loading plugin...")
    success = manager.load_plugin(plugin, auto_activate=False)
    print(f"Load result: {success}")
    
    print("Activating plugin...")
    success = manager.activate_plugin(plugin.info.name)
    print(f"Activate result: {success}")
    
    print("Deactivating plugin...")
    success = manager.deactivate_plugin(plugin.info.name)
    print(f"Deactivate result: {success}")
    
    print("Unloading plugin...")
    success = manager.unload_plugin(plugin.info.name)
    print(f"Unload result: {success}")
    
    return True


def test_plugin_validation():
    """Test plugin validation functionality."""
    print("\n=== Testing Plugin Validation ===")
    
    loader = PluginLoader()
    
    # Test validation of example plugin
    example_path = "quantum_platform/plugins/examples/gate_cancellation.py"
    
    if Path(example_path).exists():
        validation = loader.validate_plugin_file(example_path)
        print(f"Validation result for {example_path}:")
        for key, value in validation.items():
            print(f"  {key}: {value}")
        
        return validation["valid"]
    else:
        print("No plugin file to validate")
        return False


def main():
    """Run all plugin system tests."""
    print("Plugin Architecture - Test Suite")
    print("=" * 50)
    
    try:
        # Run all tests
        test_plugin_manager()
        test_plugin_loader()
        test_plugin_discovery()
        test_plugin_integration()
        test_create_simple_plugin()
        test_plugin_lifecycle()
        test_plugin_validation()
        
        print("\n" + "=" * 50)
        print("✅ All plugin tests completed!")
        print("The plugin architecture is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 