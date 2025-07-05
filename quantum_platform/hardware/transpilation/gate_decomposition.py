"""
Gate Decomposition

This module provides gate decomposition functionality for breaking down
complex quantum gates into basis gate sets supported by hardware.
"""

import math
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass

from ...compiler.ir.operation import Operation
from ...compiler.ir.qubit import Qubit


@dataclass
class DecompositionRule:
    """Rule for decomposing a gate into basis gates."""
    source_gate: str
    target_gates: List[str]
    decomposition_func: Callable
    parameter_mapping: Optional[Dict[str, str]] = None
    cost: int = 1  # Number of basis gates produced


class GateDecomposer:
    """Gate decomposition engine."""
    
    def __init__(self):
        self.rules: Dict[str, DecompositionRule] = {}
        self._initialize_standard_rules()
    
    def _initialize_standard_rules(self):
        """Initialize standard decomposition rules."""
        
        # Hadamard to RZ, RX, RZ
        self.add_rule(DecompositionRule(
            source_gate="h",
            target_gates=["rz", "rx", "rz"],
            decomposition_func=self._decompose_hadamard,
            cost=3
        ))
        
        # Pauli-Y to RZ, RX
        self.add_rule(DecompositionRule(
            source_gate="y",
            target_gates=["rz", "rx", "rz"],
            decomposition_func=self._decompose_pauli_y,
            cost=3
        ))
        
        # Pauli-Z to RZ
        self.add_rule(DecompositionRule(
            source_gate="z",
            target_gates=["rz"],
            decomposition_func=self._decompose_pauli_z,
            cost=1
        ))
        
        # CNOT to CZ with Hadamards
        self.add_rule(DecompositionRule(
            source_gate="cnot",
            target_gates=["h", "cz", "h"],
            decomposition_func=self._decompose_cnot_to_cz,
            cost=3
        ))
        
        # CZ to CNOT with Hadamards  
        self.add_rule(DecompositionRule(
            source_gate="cz",
            target_gates=["h", "cnot", "h"],
            decomposition_func=self._decompose_cz_to_cnot,
            cost=3
        ))
        
        # SWAP to CNOTs
        self.add_rule(DecompositionRule(
            source_gate="swap",
            target_gates=["cnot", "cnot", "cnot"],
            decomposition_func=self._decompose_swap,
            cost=3
        ))
        
        # Toffoli to basic gates
        self.add_rule(DecompositionRule(
            source_gate="ccx",
            target_gates=["h", "cnot", "rz", "cnot", "rz", "cnot", "rz", "cnot", "rz", "h"],
            decomposition_func=self._decompose_toffoli,
            cost=10
        ))
        
        # U3 to RZ, RX, RZ (universal single-qubit)
        self.add_rule(DecompositionRule(
            source_gate="u3",
            target_gates=["rz", "rx", "rz"],
            decomposition_func=self._decompose_u3,
            cost=3
        ))
    
    def add_rule(self, rule: DecompositionRule):
        """Add a decomposition rule."""
        self.rules[rule.source_gate.lower()] = rule
    
    def decompose(self, operation: Operation, basis_gates: List[str]) -> List[Operation]:
        """Decompose an operation into basis gates."""
        gate_name = operation.__class__.__name__.lower()
        basis_gates_lower = [g.lower() for g in basis_gates]
        
        # If gate is already in basis set, return as-is
        if gate_name in basis_gates_lower:
            return [operation]
        
        # Find applicable decomposition rule
        if gate_name in self.rules:
            rule = self.rules[gate_name]
            
            # Check if target gates are in basis set
            if all(target.lower() in basis_gates_lower for target in rule.target_gates):
                return rule.decomposition_func(operation)
        
        # Try recursive decomposition
        return self._recursive_decompose(operation, basis_gates_lower)
    
    def _recursive_decompose(self, operation: Operation, basis_gates: List[str]) -> List[Operation]:
        """Recursively decompose using available rules."""
        gate_name = operation.__class__.__name__.lower()
        
        if gate_name in basis_gates:
            return [operation]
        
        # Find a rule that gets us closer to basis gates
        for rule_name, rule in self.rules.items():
            if rule_name == gate_name:
                # Apply this rule and recursively decompose results
                intermediate_ops = rule.decomposition_func(operation)
                result = []
                
                for op in intermediate_ops:
                    result.extend(self._recursive_decompose(op, basis_gates))
                
                return result
        
        # No decomposition found - return original (will likely cause error later)
        return [operation]
    
    # Decomposition functions
    
    def _decompose_hadamard(self, operation: Operation) -> List[Operation]:
        """Decompose Hadamard: H = RZ(π) RX(π/2) RZ(π)"""
        target = operation.targets[0]
        
        # Import gate classes
        from ...compiler.language.operations import RZ, RX
        
        return [
            RZ(target, math.pi),
            RX(target, math.pi/2),
            RZ(target, math.pi)
        ]
    
    def _decompose_pauli_y(self, operation: Operation) -> List[Operation]:
        """Decompose Pauli-Y: Y = RZ(π) RX(π) RZ(π)"""
        target = operation.targets[0]
        
        from ...compiler.language.operations import RZ, RX
        
        return [
            RZ(target, math.pi),
            RX(target, math.pi),
            RZ(target, math.pi)
        ]
    
    def _decompose_pauli_z(self, operation: Operation) -> List[Operation]:
        """Decompose Pauli-Z: Z = RZ(π)"""
        target = operation.targets[0]
        
        from ...compiler.language.operations import RZ
        
        return [RZ(target, math.pi)]
    
    def _decompose_cnot_to_cz(self, operation: Operation) -> List[Operation]:
        """Decompose CNOT to CZ: CNOT = H(target) CZ(control, target) H(target)"""
        control, target = operation.targets[0], operation.targets[1]
        
        from ...compiler.language.operations import H, CZ
        
        return [
            H(target),
            CZ(control, target),
            H(target)
        ]
    
    def _decompose_cz_to_cnot(self, operation: Operation) -> List[Operation]:
        """Decompose CZ to CNOT: CZ = H(target) CNOT(control, target) H(target)"""
        control, target = operation.targets[0], operation.targets[1]
        
        from ...compiler.language.operations import H, CNOT
        
        return [
            H(target),
            CNOT(control, target),
            H(target)
        ]
    
    def _decompose_swap(self, operation: Operation) -> List[Operation]:
        """Decompose SWAP: SWAP = CNOT(a,b) CNOT(b,a) CNOT(a,b)"""
        qubit_a, qubit_b = operation.targets[0], operation.targets[1]
        
        from ...compiler.language.operations import CNOT
        
        return [
            CNOT(qubit_a, qubit_b),
            CNOT(qubit_b, qubit_a),
            CNOT(qubit_a, qubit_b)
        ]
    
    def _decompose_toffoli(self, operation: Operation) -> List[Operation]:
        """Decompose Toffoli (CCX) gate using standard decomposition."""
        control1, control2, target = operation.targets[0], operation.targets[1], operation.targets[2]
        
        from ...compiler.language.operations import H, CNOT, RZ
        
        return [
            H(target),
            CNOT(control2, target),
            RZ(target, -math.pi/4),
            CNOT(control1, target),
            RZ(target, math.pi/4),
            CNOT(control2, target),
            RZ(target, -math.pi/4),
            CNOT(control1, target),
            RZ(control2, math.pi/4),
            RZ(target, math.pi/4),
            H(target),
            CNOT(control1, control2),
            RZ(control1, math.pi/4),
            RZ(control2, -math.pi/4),
            CNOT(control1, control2)
        ]
    
    def _decompose_u3(self, operation: Operation) -> List[Operation]:
        """Decompose U3 gate: U3(θ,φ,λ) = RZ(φ) RX(-π/2) RZ(θ) RX(π/2) RZ(λ)"""
        target = operation.targets[0]
        
        # Extract parameters (assuming U3 has theta, phi, lambda parameters)
        params = list(operation.parameters.values())
        if len(params) >= 3:
            theta, phi, lam = params[0], params[1], params[2]
        else:
            # Default values if not enough parameters
            theta = params[0] if len(params) > 0 else 0
            phi = params[1] if len(params) > 1 else 0
            lam = params[2] if len(params) > 2 else 0
        
        from ...compiler.language.operations import RZ, RX
        
        return [
            RZ(target, phi),
            RX(target, -math.pi/2),
            RZ(target, theta),
            RX(target, math.pi/2),
            RZ(target, lam)
        ]
    
    def get_decomposition_cost(self, gate_name: str, basis_gates: List[str]) -> int:
        """Get the cost of decomposing a gate into basis gates."""
        gate_name = gate_name.lower()
        basis_gates_lower = [g.lower() for g in basis_gates]
        
        if gate_name in basis_gates_lower:
            return 0  # No decomposition needed
        
        if gate_name in self.rules:
            rule = self.rules[gate_name]
            
            # Check if can be directly decomposed
            if all(target.lower() in basis_gates_lower for target in rule.target_gates):
                return rule.cost
            
            # Recursive cost calculation
            total_cost = 0
            for target_gate in rule.target_gates:
                total_cost += self.get_decomposition_cost(target_gate, basis_gates)
            
            return total_cost
        
        return float('inf')  # Cannot decompose
    
    def list_rules(self) -> List[str]:
        """List all available decomposition rules."""
        return list(self.rules.keys())
    
    def can_decompose(self, gate_name: str, basis_gates: List[str]) -> bool:
        """Check if a gate can be decomposed into basis gates."""
        cost = self.get_decomposition_cost(gate_name, basis_gates)
        return cost != float('inf')


def get_decomposition_rules() -> Dict[str, DecompositionRule]:
    """Get standard decomposition rules."""
    decomposer = GateDecomposer()
    return decomposer.rules


def decompose_gate(operation: Operation, basis_gates: List[str]) -> List[Operation]:
    """Convenience function to decompose a single gate."""
    decomposer = GateDecomposer()
    return decomposer.decompose(operation, basis_gates) 