"""
Quantum Circuit Transpiler

This module provides the main transpilation engine for converting abstract
quantum circuits into hardware-compatible formats.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time

from ...compiler.ir.circuit import QuantumCircuit
from ...compiler.ir.operation import Operation
from ...errors import CompilationError, handle_errors
from ..hal import DeviceInfo
from .qubit_mapping import QubitMapping, QubitMapper
from .gate_decomposition import GateDecomposer
from .routing import QuantumRouter


@dataclass
class TranspilationResult:
    """Result of circuit transpilation."""
    original_circuit: QuantumCircuit
    transpiled_circuit: QuantumCircuit
    qubit_mapping: QubitMapping
    
    # Statistics
    original_gate_count: int = 0
    transpiled_gate_count: int = 0
    original_depth: int = 0
    transpiled_depth: int = 0
    
    # Transpilation metadata
    passes_applied: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    # Detailed metrics
    swap_count: int = 0
    decomposition_count: int = 0
    optimization_savings: int = 0
    
    def get_overhead_ratio(self) -> float:
        """Get the overhead ratio (transpiled/original gate count)."""
        if self.original_gate_count == 0:
            return 0.0
        return self.transpiled_gate_count / self.original_gate_count
    
    def get_depth_ratio(self) -> float:
        """Get the depth ratio (transpiled/original depth)."""
        if self.original_depth == 0:
            return 0.0
        return self.transpiled_depth / self.original_depth


class TranspilationPass(ABC):
    """Abstract base class for transpilation passes."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def run(self, circuit: QuantumCircuit, device_info: DeviceInfo, 
            mapping: QubitMapping) -> Tuple[QuantumCircuit, QubitMapping]:
        """Run the transpilation pass."""
        pass
    
    def __str__(self):
        return self.name


class QubitMappingPass(TranspilationPass):
    """Pass for mapping logical qubits to physical qubits."""
    
    def __init__(self):
        super().__init__("qubit_mapping")
        self.mapper = QubitMapper()
    
    def run(self, circuit: QuantumCircuit, device_info: DeviceInfo, 
            mapping: QubitMapping) -> Tuple[QuantumCircuit, QubitMapping]:
        """Map logical qubits to physical qubits."""
        if mapping.is_mapped:
            return circuit, mapping
        
        # Create initial mapping
        new_mapping = self.mapper.create_initial_mapping(
            circuit.num_qubits, 
            device_info.num_qubits,
            device_info.coupling_map
        )
        
        # Apply mapping to circuit
        mapped_circuit = self._apply_mapping(circuit, new_mapping)
        
        return mapped_circuit, new_mapping
    
    def _apply_mapping(self, circuit: QuantumCircuit, mapping: QubitMapping) -> QuantumCircuit:
        """Apply qubit mapping to circuit operations."""
        # Create new circuit with physical qubits
        mapped_circuit = QuantumCircuit(
            name=circuit.name + "_mapped",
            num_qubits=max(mapping.logical_to_physical.values()) + 1
        )
        
        # Map operations
        for operation in circuit.operations:
            mapped_targets = []
            for target in operation.targets:
                physical_qubit = mapping.logical_to_physical.get(target.id)
                if physical_qubit is not None:
                    # Create new qubit reference with physical ID
                    from ...compiler.ir.qubit import Qubit
                    mapped_targets.append(Qubit(physical_qubit))
            
            # Create new operation with mapped targets
            if mapped_targets:
                new_operation = operation.__class__(
                    targets=mapped_targets,
                    parameters=operation.parameters
                )
                mapped_circuit.add_operation(new_operation)
        
        return mapped_circuit


class GateDecompositionPass(TranspilationPass):
    """Pass for decomposing gates into basis gates."""
    
    def __init__(self):
        super().__init__("gate_decomposition")
        self.decomposer = GateDecomposer()
    
    def run(self, circuit: QuantumCircuit, device_info: DeviceInfo, 
            mapping: QubitMapping) -> Tuple[QuantumCircuit, QubitMapping]:
        """Decompose gates into basis gates."""
        basis_gates = [gate.lower() for gate in device_info.basis_gates]
        decomposed_circuit = QuantumCircuit(
            name=circuit.name + "_decomposed",
            num_qubits=circuit.num_qubits
        )
        
        for operation in circuit.operations:
            gate_name = operation.__class__.__name__.lower()
            
            if gate_name in basis_gates:
                # Gate is already supported
                decomposed_circuit.add_operation(operation)
            else:
                # Decompose gate
                decomposed_ops = self.decomposer.decompose(operation, basis_gates)
                for op in decomposed_ops:
                    decomposed_circuit.add_operation(op)
        
        return decomposed_circuit, mapping


class RoutingPass(TranspilationPass):
    """Pass for routing circuits on limited connectivity devices."""
    
    def __init__(self):
        super().__init__("routing")
        self.router = QuantumRouter()
    
    def run(self, circuit: QuantumCircuit, device_info: DeviceInfo, 
            mapping: QubitMapping) -> Tuple[QuantumCircuit, QubitMapping]:
        """Route circuit to handle connectivity constraints."""
        if not device_info.coupling_map:
            # All-to-all connectivity, no routing needed
            return circuit, mapping
        
        routed_circuit, updated_mapping = self.router.route(
            circuit, device_info.coupling_map, mapping
        )
        
        return routed_circuit, updated_mapping


class OptimizationPass(TranspilationPass):
    """Pass for post-transpilation optimization."""
    
    def __init__(self):
        super().__init__("optimization")
    
    def run(self, circuit: QuantumCircuit, device_info: DeviceInfo, 
            mapping: QubitMapping) -> Tuple[QuantumCircuit, QubitMapping]:
        """Apply post-transpilation optimizations."""
        # Simple gate cancellation
        optimized_circuit = self._cancel_adjacent_gates(circuit)
        
        return optimized_circuit, mapping
    
    def _cancel_adjacent_gates(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Cancel adjacent inverse gates."""
        optimized_circuit = QuantumCircuit(
            name=circuit.name + "_optimized",
            num_qubits=circuit.num_qubits
        )
        
        operations = list(circuit.operations)
        i = 0
        
        while i < len(operations):
            current_op = operations[i]
            
            # Check if next operation cancels current one
            if i + 1 < len(operations):
                next_op = operations[i + 1]
                
                if self._are_inverse_gates(current_op, next_op):
                    # Skip both operations (they cancel)
                    i += 2
                    continue
            
            # Add current operation
            optimized_circuit.add_operation(current_op)
            i += 1
        
        return optimized_circuit
    
    def _are_inverse_gates(self, op1: Operation, op2: Operation) -> bool:
        """Check if two operations are inverses and cancel each other."""
        # Same gate type and targets
        if (op1.__class__ != op2.__class__ or 
            len(op1.targets) != len(op2.targets)):
            return False
        
        # Same target qubits
        for t1, t2 in zip(op1.targets, op2.targets):
            if t1.id != t2.id:
                return False
        
        # Check for self-inverse gates (X, Y, Z, H, CNOT)
        self_inverse_gates = ['X', 'Y', 'Z', 'H', 'CNOT']
        if op1.__class__.__name__ in self_inverse_gates:
            return True
        
        # Check for parameterized gates with opposite parameters
        if hasattr(op1, 'parameters') and hasattr(op2, 'parameters'):
            params1 = list(op1.parameters.values())
            params2 = list(op2.parameters.values())
            
            if len(params1) == len(params2) == 1:
                # Single parameter gates (RX, RY, RZ)
                return abs(params1[0] + params2[0]) < 1e-10
        
        return False


class CircuitTranspiler:
    """Main circuit transpiler coordinating all transpilation passes."""
    
    def __init__(self):
        self.passes = [
            QubitMappingPass(),
            GateDecompositionPass(),
            RoutingPass(),
            OptimizationPass()
        ]
        self._custom_passes = []
    
    def add_pass(self, transpilation_pass: TranspilationPass):
        """Add a custom transpilation pass."""
        self._custom_passes.append(transpilation_pass)
    
    @handle_errors
    def transpile(self, circuit: QuantumCircuit, device_info: DeviceInfo,
                 optimization_level: int = 1) -> TranspilationResult:
        """Transpile circuit for the given device."""
        start_time = time.time()
        
        # Initialize result
        result = TranspilationResult(
            original_circuit=circuit,
            transpiled_circuit=circuit,
            qubit_mapping=QubitMapping(),
            original_gate_count=len(circuit.operations),
            original_depth=circuit.depth
        )
        
        try:
            current_circuit = circuit
            current_mapping = QubitMapping()
            
            # Apply transpilation passes
            passes_to_apply = self.passes + self._custom_passes
            
            for transpilation_pass in passes_to_apply:
                try:
                    current_circuit, current_mapping = transpilation_pass.run(
                        current_circuit, device_info, current_mapping
                    )
                    result.passes_applied.append(transpilation_pass.name)
                    
                except Exception as e:
                    raise CompilationError(
                        f"Transpilation pass '{transpilation_pass.name}' failed: {e}",
                        user_message=f"Circuit transpilation failed at {transpilation_pass.name}"
                    )
            
            # Update result
            result.transpiled_circuit = current_circuit
            result.qubit_mapping = current_mapping
            result.transpiled_gate_count = len(current_circuit.operations)
            result.transpiled_depth = current_circuit.depth
            result.execution_time = time.time() - start_time
            result.success = True
            
            # Calculate statistics
            result.swap_count = self._count_swap_operations(current_circuit)
            result.decomposition_count = (
                result.transpiled_gate_count - result.original_gate_count - result.swap_count
            )
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.execution_time = time.time() - start_time
            raise e
        
        return result
    
    def _count_swap_operations(self, circuit: QuantumCircuit) -> int:
        """Count the number of SWAP operations in the circuit."""
        swap_count = 0
        for operation in circuit.operations:
            if operation.__class__.__name__.lower() == 'swap':
                swap_count += 1
        return swap_count
    
    def get_transpilation_preview(self, circuit: QuantumCircuit, 
                                device_info: DeviceInfo) -> Dict[str, Any]:
        """Get a preview of what transpilation would do without actually doing it."""
        preview = {
            "original_gates": len(circuit.operations),
            "original_depth": circuit.depth,
            "device_qubits": device_info.num_qubits,
            "device_gates": device_info.basis_gates,
            "connectivity_limited": bool(device_info.coupling_map),
            "estimated_overhead": 1.0
        }
        
        # Estimate overhead based on unsupported gates
        basis_gates = [gate.lower() for gate in device_info.basis_gates]
        unsupported_gates = []
        
        for operation in circuit.operations:
            gate_name = operation.__class__.__name__.lower()
            if gate_name not in basis_gates:
                unsupported_gates.append(gate_name)
        
        preview["unsupported_gates"] = list(set(unsupported_gates))
        
        # Rough overhead estimate
        if unsupported_gates:
            preview["estimated_overhead"] = 1.5 + len(unsupported_gates) * 0.2
        
        if device_info.coupling_map and circuit.num_qubits > 3:
            preview["estimated_overhead"] *= 1.2  # SWAP overhead
        
        return preview


def transpile_for_device(circuit: QuantumCircuit, device_info: DeviceInfo,
                        optimization_level: int = 1) -> TranspilationResult:
    """Convenience function to transpile a circuit for a specific device."""
    transpiler = CircuitTranspiler()
    return transpiler.transpile(circuit, device_info, optimization_level) 