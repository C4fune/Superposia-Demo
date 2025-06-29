"""
Gate Registry for managing the extensible gate set.

This module provides the central registry for all quantum gates in the platform,
enabling dynamic registration and lookup of gates.
"""

from typing import Dict, List, Optional, Any, Set, Type
from collections import defaultdict
import threading
import json

from quantum_platform.compiler.gates.base import Gate, GateType


class GateRegistry:
    """
    Central registry for all quantum gates in the platform.
    
    This provides thread-safe registration, lookup, and management
    of the gate set, enabling extensibility and plugin support.
    """
    
    def __init__(self):
        """Initialize the gate registry."""
        self._gates: Dict[str, Gate] = {}
        self._gates_by_type: Dict[GateType, Set[str]] = defaultdict(set)
        self._gates_by_qubits: Dict[int, Set[str]] = defaultdict(set)
        self._parametric_gates: Set[str] = set()
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Metadata for registry
        self._registry_version = "1.0.0"
        self._metadata: Dict[str, Any] = {}
    
    def register(self, gate: Gate, overwrite: bool = False) -> None:
        """
        Register a gate in the registry.
        
        Args:
            gate: Gate instance to register
            overwrite: Whether to overwrite existing gate with same name
            
        Raises:
            ValueError: If gate name already exists and overwrite=False
            TypeError: If gate is not a Gate instance
        """
        if not isinstance(gate, Gate):
            raise TypeError(f"Expected Gate instance, got {type(gate)}")
        
        with self._lock:
            if gate.name in self._gates and not overwrite:
                raise ValueError(f"Gate '{gate.name}' already registered. Use overwrite=True to replace.")
            
            # Remove old gate from indices if replacing
            if gate.name in self._gates:
                self._remove_from_indices(gate.name)
            
            # Add to main registry
            self._gates[gate.name] = gate
            
            # Update indices
            self._gates_by_type[gate.gate_type].add(gate.name)
            self._gates_by_qubits[gate.num_qubits].add(gate.name)
            
            if gate.is_parametric:
                self._parametric_gates.add(gate.name)
    
    def _remove_from_indices(self, gate_name: str) -> None:
        """Remove gate from all indices (internal helper)."""
        if gate_name not in self._gates:
            return
        
        gate = self._gates[gate_name]
        
        # Remove from type index
        self._gates_by_type[gate.gate_type].discard(gate_name)
        
        # Remove from qubit count index
        self._gates_by_qubits[gate.num_qubits].discard(gate_name)
        
        # Remove from parametric gates
        self._parametric_gates.discard(gate_name)
    
    def unregister(self, gate_name: str) -> bool:
        """
        Unregister a gate from the registry.
        
        Args:
            gate_name: Name of gate to remove
            
        Returns:
            True if gate was removed, False if not found
        """
        with self._lock:
            if gate_name not in self._gates:
                return False
            
            self._remove_from_indices(gate_name)
            del self._gates[gate_name]
            return True
    
    def get(self, gate_name: str) -> Optional[Gate]:
        """
        Get a gate by name.
        
        Args:
            gate_name: Name of the gate
            
        Returns:
            Gate instance or None if not found
        """
        with self._lock:
            return self._gates.get(gate_name)
    
    def get_all(self) -> Dict[str, Gate]:
        """
        Get all registered gates.
        
        Returns:
            Dictionary mapping gate names to Gate instances
        """
        with self._lock:
            return self._gates.copy()
    
    def get_by_type(self, gate_type: GateType) -> List[Gate]:
        """
        Get all gates of a specific type.
        
        Args:
            gate_type: Type of gates to retrieve
            
        Returns:
            List of gates of the specified type
        """
        with self._lock:
            gate_names = self._gates_by_type.get(gate_type, set())
            return [self._gates[name] for name in gate_names]
    
    def get_by_qubit_count(self, num_qubits: int) -> List[Gate]:
        """
        Get all gates that act on a specific number of qubits.
        
        Args:
            num_qubits: Number of qubits
            
        Returns:
            List of gates that act on num_qubits qubits
        """
        with self._lock:
            gate_names = self._gates_by_qubits.get(num_qubits, set())
            return [self._gates[name] for name in gate_names]
    
    def get_parametric_gates(self) -> List[Gate]:
        """
        Get all parametric gates.
        
        Returns:
            List of gates that have parameters
        """
        with self._lock:
            return [self._gates[name] for name in self._parametric_gates]
    
    def get_single_qubit_gates(self) -> List[Gate]:
        """Get all single-qubit gates."""
        return self.get_by_qubit_count(1)
    
    def get_two_qubit_gates(self) -> List[Gate]:
        """Get all two-qubit gates."""
        return self.get_by_qubit_count(2)
    
    def search(self, pattern: str, case_sensitive: bool = False) -> List[Gate]:
        """
        Search for gates by name pattern.
        
        Args:
            pattern: String pattern to search for
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of gates whose names match the pattern
        """
        if not case_sensitive:
            pattern = pattern.lower()
        
        with self._lock:
            matches = []
            for name, gate in self._gates.items():
                search_name = name if case_sensitive else name.lower()
                if pattern in search_name:
                    matches.append(gate)
            return matches
    
    def exists(self, gate_name: str) -> bool:
        """
        Check if a gate is registered.
        
        Args:
            gate_name: Name of the gate
            
        Returns:
            True if gate exists in registry
        """
        with self._lock:
            return gate_name in self._gates
    
    def list_names(self) -> List[str]:
        """
        Get list of all registered gate names.
        
        Returns:
            Sorted list of gate names
        """
        with self._lock:
            return sorted(self._gates.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the gate registry.
        
        Returns:
            Dictionary with registry statistics
        """
        with self._lock:
            stats = {
                "total_gates": len(self._gates),
                "parametric_gates": len(self._parametric_gates),
                "gates_by_type": {
                    gate_type.value: len(gates)
                    for gate_type, gates in self._gates_by_type.items()
                },
                "gates_by_qubit_count": {
                    str(num_qubits): len(gates)
                    for num_qubits, gates in self._gates_by_qubits.items()
                },
                "registry_version": self._registry_version
            }
            return stats
    
    def clear(self) -> None:
        """Clear all gates from the registry."""
        with self._lock:
            self._gates.clear()
            self._gates_by_type.clear()
            self._gates_by_qubits.clear()
            self._parametric_gates.clear()
    
    def export_metadata(self) -> Dict[str, Any]:
        """
        Export registry metadata (gate names and types, no implementations).
        
        Returns:
            Dictionary with registry metadata
        """
        with self._lock:
            metadata = {
                "registry_version": self._registry_version,
                "gates": {}
            }
            
            for name, gate in self._gates.items():
                metadata["gates"][name] = {
                    "name": gate.name,
                    "num_qubits": gate.num_qubits,
                    "gate_type": gate.gate_type.value,
                    "is_parametric": gate.is_parametric,
                    "parameters": gate.parameters,
                    "description": gate.description
                }
            
            return metadata
    
    def to_json(self) -> str:
        """Export registry metadata as JSON string."""
        return json.dumps(self.export_metadata(), indent=2)
    
    def __len__(self) -> int:
        """Get number of registered gates."""
        with self._lock:
            return len(self._gates)
    
    def __contains__(self, gate_name: str) -> bool:
        """Check if gate name is in registry."""
        return self.exists(gate_name)
    
    def __iter__(self):
        """Iterate over gate names."""
        with self._lock:
            return iter(list(self._gates.keys()))
    
    def __getitem__(self, gate_name: str) -> Gate:
        """Get gate by name (dict-like access)."""
        gate = self.get(gate_name)
        if gate is None:
            raise KeyError(f"Gate '{gate_name}' not found in registry")
        return gate
    
    def __repr__(self) -> str:
        """String representation of registry."""
        with self._lock:
            return f"GateRegistry({len(self._gates)} gates)"


# Global gate registry instance
_global_registry = GateRegistry()

# Global access functions
def register_gate(gate: Gate, overwrite: bool = False) -> None:
    """Register a gate in the global registry."""
    _global_registry.register(gate, overwrite)

def get_gate(gate_name: str) -> Optional[Gate]:
    """Get a gate from the global registry."""
    return _global_registry.get(gate_name)

def unregister_gate(gate_name: str) -> bool:
    """Unregister a gate from the global registry."""
    return _global_registry.unregister(gate_name)

def list_gates() -> List[str]:
    """List all gate names in the global registry."""
    return _global_registry.list_names()

def gate_exists(gate_name: str) -> bool:
    """Check if a gate exists in the global registry."""
    return _global_registry.exists(gate_name)

def get_registry_stats() -> Dict[str, Any]:
    """Get statistics about the global gate registry."""
    return _global_registry.get_statistics()

# Export the global registry as GATE_SET for backward compatibility
GATE_SET = _global_registry 