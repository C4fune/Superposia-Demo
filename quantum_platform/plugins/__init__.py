"""
Plugin Architecture for Quantum Platform

This subsystem provides extensibility through a plugin system that allows
external contributors to add new compiler passes, gates, optimization
routines, and language features without modifying the core codebase.
"""

from quantum_platform.plugins.manager import PluginManager, PluginRegistry
from quantum_platform.plugins.base import (
    Plugin, PluginType, PluginInfo, 
    CompilerPassPlugin, GatePlugin, OptimizerPlugin, ExporterPlugin
)
from quantum_platform.plugins.loader import PluginLoader
from quantum_platform.plugins.discovery import PluginDiscovery

__all__ = [
    "PluginManager",
    "PluginRegistry", 
    "Plugin",
    "PluginType",
    "PluginInfo",
    "CompilerPassPlugin",
    "GatePlugin", 
    "OptimizerPlugin",
    "ExporterPlugin",
    "PluginLoader",
    "PluginDiscovery",
] 