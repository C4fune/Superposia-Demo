"""
Qubit Mapping

This module provides qubit mapping functionality for mapping logical qubits
to physical qubits based on device connectivity.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import networkx as nx


@dataclass
class QubitMapping:
    """Mapping between logical and physical qubits."""
    logical_to_physical: Dict[int, int]
    physical_to_logical: Dict[int, int]
    is_mapped: bool = False
    
    def __init__(self, mapping: Optional[Dict[int, int]] = None):
        if mapping:
            self.logical_to_physical = mapping.copy()
            self.physical_to_logical = {v: k for k, v in mapping.items()}
            self.is_mapped = True
        else:
            self.logical_to_physical = {}
            self.physical_to_logical = {}
            self.is_mapped = False
    
    def map_qubit(self, logical: int, physical: int):
        """Map a logical qubit to a physical qubit."""
        self.logical_to_physical[logical] = physical
        self.physical_to_logical[physical] = logical
        self.is_mapped = True
    
    def get_physical(self, logical: int) -> Optional[int]:
        """Get physical qubit for logical qubit."""
        return self.logical_to_physical.get(logical)
    
    def get_logical(self, physical: int) -> Optional[int]:
        """Get logical qubit for physical qubit."""
        return self.physical_to_logical.get(physical)
    
    def swap_physical(self, phys1: int, phys2: int):
        """Swap the logical assignments of two physical qubits."""
        log1 = self.physical_to_logical.get(phys1)
        log2 = self.physical_to_logical.get(phys2)
        
        if log1 is not None:
            self.logical_to_physical[log1] = phys2
            self.physical_to_logical[phys2] = log1
        else:
            if phys2 in self.physical_to_logical:
                del self.physical_to_logical[phys2]
        
        if log2 is not None:
            self.logical_to_physical[log2] = phys1
            self.physical_to_logical[phys1] = log2
        else:
            if phys1 in self.physical_to_logical:
                del self.physical_to_logical[phys1]
    
    def copy(self) -> 'QubitMapping':
        """Create a copy of the mapping."""
        new_mapping = QubitMapping()
        new_mapping.logical_to_physical = self.logical_to_physical.copy()
        new_mapping.physical_to_logical = self.physical_to_logical.copy()
        new_mapping.is_mapped = self.is_mapped
        return new_mapping


class QubitMapper:
    """Qubit mapping algorithms."""
    
    def create_initial_mapping(self, num_logical: int, num_physical: int,
                             coupling_map: List[List[int]]) -> QubitMapping:
        """Create initial mapping of logical to physical qubits."""
        if num_logical > num_physical:
            raise ValueError(f"Not enough physical qubits: {num_logical} > {num_physical}")
        
        if not coupling_map:
            # All-to-all connectivity - use identity mapping
            mapping = {i: i for i in range(num_logical)}
            return QubitMapping(mapping)
        
        # For limited connectivity, use graph-based approach
        return self._graph_based_mapping(num_logical, coupling_map)
    
    def _graph_based_mapping(self, num_logical: int, 
                           coupling_map: List[List[int]]) -> QubitMapping:
        """Create mapping based on graph connectivity."""
        # Build connectivity graph
        G = nx.Graph()
        max_qubit = 0
        
        for edge in coupling_map:
            if len(edge) >= 2:
                G.add_edge(edge[0], edge[1])
                max_qubit = max(max_qubit, edge[0], edge[1])
        
        # Find a connected subgraph for our logical qubits
        if num_logical == 1:
            # Single qubit - use any qubit
            mapping = {0: 0}
        elif num_logical == 2:
            # Two qubits - find any connected pair
            if G.edges():
                edge = list(G.edges())[0]
                mapping = {0: edge[0], 1: edge[1]}
            else:
                # No connectivity info - use adjacent qubits
                mapping = {0: 0, 1: 1}
        else:
            # Multiple qubits - find connected subgraph
            mapping = self._find_connected_subgraph(G, num_logical)
        
        return QubitMapping(mapping)
    
    def _find_connected_subgraph(self, G: nx.Graph, size: int) -> Dict[int, int]:
        """Find a connected subgraph of given size."""
        # Try to find connected component of required size
        for component in nx.connected_components(G):
            if len(component) >= size:
                # Use first 'size' nodes from this component
                physical_qubits = sorted(list(component))[:size]
                return {i: physical_qubits[i] for i in range(size)}
        
        # If no suitable component found, use greedy approach
        # Start from highest degree node
        if not G.nodes():
            # No graph info - use identity mapping
            return {i: i for i in range(size)}
        
        degrees = dict(G.degree())
        start_node = max(degrees.keys(), key=lambda x: degrees[x])
        
        selected = {start_node}
        mapping = {0: start_node}
        
        # Greedily add connected nodes
        for i in range(1, size):
            candidates = set()
            for node in selected:
                candidates.update(G.neighbors(node))
            
            # Remove already selected
            candidates -= selected
            
            if candidates:
                # Choose node with highest degree among candidates
                next_node = max(candidates, key=lambda x: degrees.get(x, 0))
            else:
                # No more connected nodes - choose any unselected
                all_nodes = set(G.nodes())
                remaining = all_nodes - selected
                if remaining:
                    next_node = min(remaining)
                else:
                    # Use any index
                    next_node = max(selected) + 1
            
            selected.add(next_node)
            mapping[i] = next_node
        
        return mapping
    
    def optimize_mapping(self, mapping: QubitMapping, 
                        coupling_map: List[List[int]],
                        circuit_connections: List[Tuple[int, int]]) -> QubitMapping:
        """Optimize mapping based on circuit connectivity."""
        # Build coupling graph
        G = nx.Graph()
        for edge in coupling_map:
            if len(edge) >= 2:
                G.add_edge(edge[0], edge[1])
        
        # Count violations in current mapping
        current_violations = self._count_violations(
            mapping, circuit_connections, G
        )
        
        # Try swapping pairs to reduce violations
        best_mapping = mapping.copy()
        best_violations = current_violations
        
        physical_qubits = list(mapping.physical_to_logical.keys())
        
        for i in range(len(physical_qubits)):
            for j in range(i + 1, len(physical_qubits)):
                # Try swapping
                test_mapping = mapping.copy()
                test_mapping.swap_physical(physical_qubits[i], physical_qubits[j])
                
                violations = self._count_violations(
                    test_mapping, circuit_connections, G
                )
                
                if violations < best_violations:
                    best_mapping = test_mapping
                    best_violations = violations
        
        return best_mapping
    
    def _count_violations(self, mapping: QubitMapping,
                         circuit_connections: List[Tuple[int, int]],
                         coupling_graph: nx.Graph) -> int:
        """Count the number of connectivity violations."""
        violations = 0
        
        for log1, log2 in circuit_connections:
            phys1 = mapping.get_physical(log1)
            phys2 = mapping.get_physical(log2)
            
            if phys1 is not None and phys2 is not None:
                if not coupling_graph.has_edge(phys1, phys2):
                    violations += 1
        
        return violations


def create_initial_mapping(num_logical: int, num_physical: int,
                         coupling_map: List[List[int]]) -> QubitMapping:
    """Convenience function to create initial qubit mapping."""
    mapper = QubitMapper()
    return mapper.create_initial_mapping(num_logical, num_physical, coupling_map)


def optimize_mapping(mapping: QubitMapping, coupling_map: List[List[int]],
                    circuit_connections: List[Tuple[int, int]]) -> QubitMapping:
    """Convenience function to optimize qubit mapping."""
    mapper = QubitMapper()
    return mapper.optimize_mapping(mapping, coupling_map, circuit_connections) 