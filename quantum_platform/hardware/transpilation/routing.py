"""
Quantum Circuit Routing

This module provides routing functionality for quantum circuits on devices
with limited qubit connectivity, including SWAP gate insertion.
"""

from typing import List, Dict, Tuple, Set, Optional
from enum import Enum
import networkx as nx
from dataclasses import dataclass

from ...compiler.ir.circuit import QuantumCircuit
from ...compiler.ir.operation import Operation
from .qubit_mapping import QubitMapping


class SwapStrategy(Enum):
    """Strategy for inserting SWAP gates."""
    BASIC = "basic"          # Insert SWAPs as needed
    LOOKAHEAD = "lookahead"  # Look ahead to minimize SWAPs
    SABRE = "sabre"          # SABRE algorithm (simplified)


@dataclass
class RoutingResult:
    """Result of circuit routing."""
    routed_circuit: QuantumCircuit
    updated_mapping: QubitMapping
    swaps_inserted: int
    routing_overhead: float
    success: bool = True
    error_message: Optional[str] = None


class QuantumRouter:
    """Quantum circuit router for limited connectivity devices."""
    
    def __init__(self, strategy: SwapStrategy = SwapStrategy.BASIC):
        self.strategy = strategy
    
    def route(self, circuit: QuantumCircuit, coupling_map: List[List[int]],
              initial_mapping: QubitMapping) -> Tuple[QuantumCircuit, QubitMapping]:
        """Route circuit for device connectivity."""
        if not coupling_map:
            # All-to-all connectivity - no routing needed
            return circuit, initial_mapping
        
        # Build coupling graph
        coupling_graph = self._build_coupling_graph(coupling_map)
        
        # Route based on strategy
        if self.strategy == SwapStrategy.BASIC:
            return self._basic_routing(circuit, coupling_graph, initial_mapping)
        elif self.strategy == SwapStrategy.LOOKAHEAD:
            return self._lookahead_routing(circuit, coupling_graph, initial_mapping)
        elif self.strategy == SwapStrategy.SABRE:
            return self._sabre_routing(circuit, coupling_graph, initial_mapping)
        else:
            raise ValueError(f"Unknown routing strategy: {self.strategy}")
    
    def _build_coupling_graph(self, coupling_map: List[List[int]]) -> nx.Graph:
        """Build NetworkX graph from coupling map."""
        G = nx.Graph()
        for edge in coupling_map:
            if len(edge) >= 2:
                G.add_edge(edge[0], edge[1])
        return G
    
    def _basic_routing(self, circuit: QuantumCircuit, coupling_graph: nx.Graph,
                      mapping: QubitMapping) -> Tuple[QuantumCircuit, QubitMapping]:
        """Basic routing: insert SWAPs as needed for each gate."""
        routed_circuit = QuantumCircuit(
            name=circuit.name + "_routed",
            num_qubits=circuit.num_qubits
        )
        current_mapping = mapping.copy()
        
        for operation in circuit.operations:
            # Check if operation needs routing
            if len(operation.targets) >= 2:  # Multi-qubit gate
                routed_ops = self._route_operation(
                    operation, coupling_graph, current_mapping
                )
                for op in routed_ops:
                    routed_circuit.add_operation(op)
            else:
                # Single qubit gate - no routing needed
                routed_circuit.add_operation(operation)
        
        return routed_circuit, current_mapping
    
    def _route_operation(self, operation: Operation, coupling_graph: nx.Graph,
                        mapping: QubitMapping) -> List[Operation]:
        """Route a single multi-qubit operation."""
        if len(operation.targets) != 2:
            # Only handle 2-qubit gates for now
            return [operation]
        
        logical_q1, logical_q2 = operation.targets[0].id, operation.targets[1].id
        physical_q1 = mapping.get_physical(logical_q1)
        physical_q2 = mapping.get_physical(logical_q2)
        
        if physical_q1 is None or physical_q2 is None:
            # Unmapped qubits - shouldn't happen if mapping is correct
            return [operation]
        
        # Check if qubits are connected
        if coupling_graph.has_edge(physical_q1, physical_q2):
            # Directly connected - no SWAPs needed
            return [operation]
        
        # Need to insert SWAPs to bring qubits together
        return self._insert_swaps_for_gate(
            operation, logical_q1, logical_q2, 
            physical_q1, physical_q2, coupling_graph, mapping
        )
    
    def _insert_swaps_for_gate(self, operation: Operation, 
                              logical_q1: int, logical_q2: int,
                              physical_q1: int, physical_q2: int,
                              coupling_graph: nx.Graph, 
                              mapping: QubitMapping) -> List[Operation]:
        """Insert SWAP gates to enable a 2-qubit operation."""
        # Find shortest path between physical qubits
        try:
            path = nx.shortest_path(coupling_graph, physical_q1, physical_q2)
        except nx.NetworkXNoPath:
            # No path - device may be disconnected
            # For now, just return original operation (will likely fail)
            return [operation]
        
        if len(path) <= 2:
            # Already connected or only one hop
            return [operation]
        
        # Insert SWAPs to move q1 towards q2
        result_operations = []
        
        # Move logical_q1 along the path towards logical_q2
        current_physical = physical_q1
        
        for i in range(len(path) - 2):
            next_physical = path[i + 1]
            
            # Insert SWAP between current_physical and next_physical
            swap_op = self._create_swap_operation(current_physical, next_physical)
            result_operations.append(swap_op)
            
            # Update mapping
            mapping.swap_physical(current_physical, next_physical)
            current_physical = next_physical
        
        # Now add the original operation
        result_operations.append(operation)
        
        return result_operations
    
    def _create_swap_operation(self, phys_q1: int, phys_q2: int) -> Operation:
        """Create a SWAP operation between two physical qubits."""
        from ...compiler.ir.qubit import Qubit
        from ...compiler.language.operations import SWAP
        
        return SWAP(Qubit(phys_q1), Qubit(phys_q2))
    
    def _lookahead_routing(self, circuit: QuantumCircuit, coupling_graph: nx.Graph,
                          mapping: QubitMapping) -> Tuple[QuantumCircuit, QubitMapping]:
        """Lookahead routing: consider future gates when inserting SWAPs."""
        routed_circuit = QuantumCircuit(
            name=circuit.name + "_routed",
            num_qubits=circuit.num_qubits
        )
        current_mapping = mapping.copy()
        operations = list(circuit.operations)
        
        i = 0
        while i < len(operations):
            operation = operations[i]
            
            if len(operation.targets) >= 2:
                # Look ahead to see if better SWAP strategy exists
                lookahead_ops = self._lookahead_route_operation(
                    operation, operations[i+1:], coupling_graph, current_mapping
                )
                for op in lookahead_ops:
                    routed_circuit.add_operation(op)
            else:
                routed_circuit.add_operation(operation)
            
            i += 1
        
        return routed_circuit, current_mapping
    
    def _lookahead_route_operation(self, operation: Operation, 
                                  future_operations: List[Operation],
                                  coupling_graph: nx.Graph,
                                  mapping: QubitMapping) -> List[Operation]:
        """Route operation with lookahead."""
        # For simplicity, use basic routing for now
        # A full implementation would analyze future_operations to minimize SWAPs
        return self._route_operation(operation, coupling_graph, mapping)
    
    def _sabre_routing(self, circuit: QuantumCircuit, coupling_graph: nx.Graph,
                      mapping: QubitMapping) -> Tuple[QuantumCircuit, QubitMapping]:
        """SABRE-like routing algorithm (simplified)."""
        # This is a simplified version of the SABRE algorithm
        # The full SABRE algorithm is quite complex
        return self._basic_routing(circuit, coupling_graph, mapping)
    
    def estimate_routing_cost(self, circuit: QuantumCircuit, 
                            coupling_map: List[List[int]],
                            mapping: QubitMapping) -> int:
        """Estimate the number of SWAP gates needed for routing."""
        if not coupling_map:
            return 0  # All-to-all connectivity
        
        coupling_graph = self._build_coupling_graph(coupling_map)
        swap_count = 0
        
        for operation in circuit.operations:
            if len(operation.targets) == 2:
                logical_q1, logical_q2 = operation.targets[0].id, operation.targets[1].id
                physical_q1 = mapping.get_physical(logical_q1)
                physical_q2 = mapping.get_physical(logical_q2)
                
                if (physical_q1 is not None and physical_q2 is not None and
                    not coupling_graph.has_edge(physical_q1, physical_q2)):
                    
                    # Estimate SWAPs needed
                    try:
                        path = nx.shortest_path(coupling_graph, physical_q1, physical_q2)
                        swap_count += max(0, len(path) - 2)
                    except nx.NetworkXNoPath:
                        swap_count += 10  # Penalty for disconnected graph
        
        return swap_count
    
    def get_routing_statistics(self, original_circuit: QuantumCircuit,
                             routed_circuit: QuantumCircuit) -> Dict[str, float]:
        """Get routing statistics."""
        original_gates = len(original_circuit.operations)
        routed_gates = len(routed_circuit.operations)
        
        # Count SWAP gates
        swap_count = 0
        for operation in routed_circuit.operations:
            if operation.__class__.__name__.lower() == 'swap':
                swap_count += 1
        
        overhead = (routed_gates - original_gates) / original_gates if original_gates > 0 else 0
        
        return {
            "original_gates": original_gates,
            "routed_gates": routed_gates,
            "swap_gates": swap_count,
            "overhead_ratio": overhead,
            "overhead_percent": overhead * 100
        }


def route_circuit(circuit: QuantumCircuit, coupling_map: List[List[int]],
                 mapping: QubitMapping, 
                 strategy: SwapStrategy = SwapStrategy.BASIC) -> Tuple[QuantumCircuit, QubitMapping]:
    """Convenience function to route a circuit."""
    router = QuantumRouter(strategy)
    return router.route(circuit, coupling_map, mapping)


def insert_swaps(circuit: QuantumCircuit, coupling_map: List[List[int]],
                mapping: QubitMapping) -> QuantumCircuit:
    """Convenience function to insert SWAP gates for connectivity."""
    routed_circuit, _ = route_circuit(circuit, coupling_map, mapping)
    return routed_circuit 