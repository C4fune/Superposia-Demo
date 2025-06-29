"""
Example Compiler Pass Plugin: Gate Cancellation

This plugin demonstrates how to create a compiler pass that removes
redundant gate pairs (like X followed by X, or H followed by H).
"""

from quantum_platform.plugins.base import CompilerPassPlugin, PluginInfo, PluginType
from quantum_platform.compiler.ir.circuit import QuantumCircuit
from quantum_platform.compiler.ir.operation import GateOperation
from typing import Callable


# Plugin metadata
gate_cancellation_info = PluginInfo(
    name="gate_cancellation",
    version="1.0.0", 
    description="Removes redundant gate pairs from quantum circuits",
    plugin_type=PluginType.COMPILER_PASS,
    author="Quantum Platform Team",
    email="quantum@platform.dev",
    license="MIT"
)


class GateCancellationPlugin(CompilerPassPlugin):
    """
    Compiler pass plugin that removes redundant gate pairs.
    
    This pass identifies gates that cancel each other out when applied
    consecutively to the same qubits and removes them from the circuit.
    """
    
    def __init__(self, info: PluginInfo):
        super().__init__(info)
        
        # Define which gates cancel themselves
        self.self_canceling_gates = {
            'X', 'Y', 'Z', 'H', 'S', 'T', 'CNOT', 'CX', 'CY', 'CZ'
        }
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        self._initialized = True
        return True
    
    def activate(self) -> bool:
        """Activate the plugin."""
        self._active = True
        return True
    
    def deactivate(self) -> bool:
        """Deactivate the plugin."""
        self._active = False
        return True
    
    def get_pass_function(self) -> Callable[[QuantumCircuit], QuantumCircuit]:
        """Get the gate cancellation pass function."""
        return self._cancel_redundant_gates
    
    def get_pass_priority(self) -> int:
        """Return priority for this pass (runs early)."""
        return 10
    
    def _cancel_redundant_gates(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Remove redundant gate pairs from the circuit.
        
        Args:
            circuit: Input quantum circuit
            
        Returns:
            Optimized circuit with redundant gates removed
        """
        # Create a new circuit with the same structure
        optimized = QuantumCircuit(
            name=f"{circuit.name}_gate_cancelled",
            num_qubits=circuit.num_qubits
        )
        
        # Copy classical registers
        for reg_name, reg in circuit.classical_registers.items():
            optimized.add_classical_register(reg_name, reg.size)
        
        # Copy qubits
        qubit_map = {}
        for qubit in circuit.qubits:
            new_qubit = optimized.allocate_qubit(qubit.name)
            qubit_map[qubit.id] = new_qubit
        
        # Process operations
        operations = list(circuit.operations)
        i = 0
        
        while i < len(operations):
            current_op = operations[i]
            
            # Check if this is a gate operation that might cancel
            if (isinstance(current_op, GateOperation) and 
                current_op.name in self.self_canceling_gates):
                
                # Look for the next operation
                if i + 1 < len(operations):
                    next_op = operations[i + 1]
                    
                    # Check if next operation cancels with current
                    if self._operations_cancel(current_op, next_op):
                        # Skip both operations
                        i += 2
                        continue
            
            # Add the current operation (with mapped qubits)
            self._copy_operation(current_op, optimized, qubit_map)
            i += 1
        
        return optimized
    
    def _operations_cancel(self, op1: GateOperation, op2: GateOperation) -> bool:
        """
        Check if two operations cancel each other.
        
        Args:
            op1: First operation
            op2: Second operation
            
        Returns:
            True if operations cancel, False otherwise
        """
        # Must be gate operations
        if not (isinstance(op1, GateOperation) and isinstance(op2, GateOperation)):
            return False
        
        # Must be the same gate type
        if op1.name != op2.name:
            return False
        
        # Must act on the same qubits
        if len(op1.targets) != len(op2.targets):
            return False
        
        target_ids_1 = {q.id for q in op1.targets}
        target_ids_2 = {q.id for q in op2.targets}
        
        if target_ids_1 != target_ids_2:
            return False
        
        # Check controls too
        control_ids_1 = {q.id for q in op1.controls}
        control_ids_2 = {q.id for q in op2.controls}
        
        if control_ids_1 != control_ids_2:
            return False
        
        # For self-canceling gates, same parameters mean they cancel
        if op1.name in self.self_canceling_gates:
            # For now, assume gates with same name and qubits cancel
            # In a full implementation, we'd check parameter values too
            return True
        
        return False
    
    def _copy_operation(self, operation, target_circuit, qubit_map):
        """Copy an operation to the target circuit with mapped qubits."""
        if isinstance(operation, GateOperation):
            # Map target qubits
            new_targets = [qubit_map[q.id] for q in operation.targets]
            new_controls = [qubit_map[q.id] for q in operation.controls]
            
            # Add gate operation
            target_circuit.add_gate_operation(
                gate_name=operation.name,
                targets=new_targets,
                controls=new_controls,
                params=operation.params
            )
        else:
            # For other operation types, we'd implement similar mapping
            # For now, just skip non-gate operations
            pass


# Create plugin instance
gate_cancellation_plugin = GateCancellationPlugin(gate_cancellation_info) 