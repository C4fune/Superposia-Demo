"""
Gate Factory for easy gate creation and application.

This module provides a factory class and utility functions for creating
and applying gates to circuits in a convenient way.
"""

from typing import List, Optional, Dict, Any, Union
from quantum_platform.compiler.ir.qubit import Qubit
from quantum_platform.compiler.ir.circuit import QuantumCircuit
from quantum_platform.compiler.ir.operation import GateOperation
from quantum_platform.compiler.ir.types import Parameter, ParameterDict
from quantum_platform.compiler.gates.base import Gate
from quantum_platform.compiler.gates.registry import get_gate, GATE_SET


class GateFactory:
    """
    Factory for creating and applying quantum gates.
    
    This provides a convenient interface for gate creation and application,
    with support for both individual gate operations and batch operations.
    """
    
    def __init__(self, circuit: Optional[QuantumCircuit] = None):
        """
        Initialize the gate factory.
        
        Args:
            circuit: Optional circuit to operate on
        """
        self.circuit = circuit
    
    def create_gate_operation(self, gate_name: str, targets: List[Qubit],
                            controls: Optional[List[Qubit]] = None,
                            params: Optional[ParameterDict] = None) -> GateOperation:
        """
        Create a gate operation.
        
        Args:
            gate_name: Name of the gate
            targets: Target qubits
            controls: Control qubits (optional)
            params: Gate parameters (optional)
            
        Returns:
            GateOperation instance
            
        Raises:
            ValueError: If gate is not found or parameters are invalid
        """
        gate = get_gate(gate_name)
        if gate is None:
            raise ValueError(f"Gate '{gate_name}' not found in registry")
        
        # Validate parameters if gate is parametric
        if gate.is_parametric:
            if not params:
                raise ValueError(f"Gate '{gate_name}' requires parameters: {gate.parameters}")
            gate.validate_parameters(params)
        
        # Create the operation
        return GateOperation(
            name=gate_name,
            targets=targets,
            controls=controls or [],
            params=params or {}
        )
    
    def apply_gate(self, gate_name: str, targets: List[Qubit],
                   controls: Optional[List[Qubit]] = None,
                   params: Optional[ParameterDict] = None) -> GateOperation:
        """
        Apply a gate to the circuit.
        
        Args:
            gate_name: Name of the gate
            targets: Target qubits
            controls: Control qubits (optional)
            params: Gate parameters (optional)
            
        Returns:
            The created and applied gate operation
            
        Raises:
            ValueError: If no circuit is associated or gate is invalid
        """
        if not self.circuit:
            raise ValueError("No circuit associated with factory")
        
        gate_op = self.create_gate_operation(gate_name, targets, controls, params)
        self.circuit.add_operation(gate_op)
        return gate_op
    
    def get_available_gates(self) -> List[str]:
        """Get list of all available gate names."""
        return GATE_SET.list_names()
    
    def get_gate_info(self, gate_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a gate.
        
        Args:
            gate_name: Name of the gate
            
        Returns:
            Dictionary with gate information or None if not found
        """
        gate = get_gate(gate_name)
        if gate is None:
            return None
        
        return {
            "name": gate.name,
            "num_qubits": gate.num_qubits,
            "gate_type": gate.gate_type.value,
            "is_parametric": gate.is_parametric,
            "parameters": gate.parameters,
            "description": gate.description,
            "metadata": gate.metadata
        }
    
    # Convenience methods for common gates
    def H(self, qubit: Qubit) -> GateOperation:
        """Apply Hadamard gate."""
        return self.apply_gate("H", [qubit])
    
    def X(self, qubit: Qubit) -> GateOperation:
        """Apply Pauli-X gate."""
        return self.apply_gate("X", [qubit])
    
    def Y(self, qubit: Qubit) -> GateOperation:
        """Apply Pauli-Y gate."""
        return self.apply_gate("Y", [qubit])
    
    def Z(self, qubit: Qubit) -> GateOperation:
        """Apply Pauli-Z gate."""
        return self.apply_gate("Z", [qubit])
    
    def S(self, qubit: Qubit) -> GateOperation:
        """Apply S gate."""
        return self.apply_gate("S", [qubit])
    
    def T(self, qubit: Qubit) -> GateOperation:
        """Apply T gate."""
        return self.apply_gate("T", [qubit])
    
    def RX(self, qubit: Qubit, theta: Parameter) -> GateOperation:
        """Apply X rotation gate."""
        return self.apply_gate("RX", [qubit], params={"theta": theta})
    
    def RY(self, qubit: Qubit, theta: Parameter) -> GateOperation:
        """Apply Y rotation gate."""
        return self.apply_gate("RY", [qubit], params={"theta": theta})
    
    def RZ(self, qubit: Qubit, theta: Parameter) -> GateOperation:
        """Apply Z rotation gate."""
        return self.apply_gate("RZ", [qubit], params={"theta": theta})
    
    def CNOT(self, control: Qubit, target: Qubit) -> GateOperation:
        """Apply CNOT gate."""
        return self.apply_gate("CNOT", [target], [control])
    
    def CX(self, control: Qubit, target: Qubit) -> GateOperation:
        """Apply CX gate (alias for CNOT)."""
        return self.apply_gate("CX", [target], [control])
    
    def CY(self, control: Qubit, target: Qubit) -> GateOperation:
        """Apply CY gate."""
        return self.apply_gate("CY", [target], [control])
    
    def CZ(self, control: Qubit, target: Qubit) -> GateOperation:
        """Apply CZ gate."""
        return self.apply_gate("CZ", [target], [control])
    
    def SWAP(self, qubit1: Qubit, qubit2: Qubit) -> GateOperation:
        """Apply SWAP gate."""
        return self.apply_gate("SWAP", [qubit1, qubit2])
    
    def TOFFOLI(self, control1: Qubit, control2: Qubit, target: Qubit) -> GateOperation:
        """Apply Toffoli gate."""
        return self.apply_gate("TOFFOLI", [target], [control1, control2])


# Global factory functions for convenience
def create_gate_operation(gate_name: str, targets: List[Qubit],
                         controls: Optional[List[Qubit]] = None,
                         params: Optional[ParameterDict] = None) -> GateOperation:
    """Create a gate operation without a specific circuit."""
    factory = GateFactory()
    return factory.create_gate_operation(gate_name, targets, controls, params)


def apply_gate_to_circuit(circuit: QuantumCircuit, gate_name: str, targets: List[Qubit],
                         controls: Optional[List[Qubit]] = None,
                         params: Optional[ParameterDict] = None) -> GateOperation:
    """Apply a gate to a specific circuit."""
    factory = GateFactory(circuit)
    return factory.apply_gate(gate_name, targets, controls, params) 