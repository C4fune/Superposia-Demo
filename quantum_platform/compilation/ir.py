"""
Simple Quantum Circuit Intermediate Representation

This module provides a basic quantum circuit IR for the marketplace system.
It includes classes for representing quantum circuits, gates, and registers.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import uuid


@dataclass
class QuantumRegister:
    """Represents a quantum register."""
    name: str
    size: int
    
    def __getitem__(self, index: int) -> int:
        """Get qubit index."""
        if index >= self.size:
            raise IndexError(f"Qubit index {index} out of range for register of size {self.size}")
        return index


@dataclass
class ClassicalRegister:
    """Represents a classical register."""
    name: str
    size: int
    
    def __getitem__(self, index: int) -> int:
        """Get bit index."""
        if index >= self.size:
            raise IndexError(f"Bit index {index} out of range for register of size {self.size}")
        return index


@dataclass
class QuantumGate:
    """Represents a quantum gate."""
    name: str
    qubits: List[int]
    parameters: List[float] = field(default_factory=list)
    classical_bits: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "qubits": self.qubits,
            "parameters": self.parameters,
            "classical_bits": self.classical_bits
        }


class QuantumCircuit:
    """
    Simple quantum circuit representation.
    
    This provides a basic interface for creating and manipulating quantum circuits
    compatible with the marketplace algorithm library.
    """
    
    def __init__(self, num_qubits: int, num_classical_bits: int = 0):
        """
        Initialize quantum circuit.
        
        Args:
            num_qubits: Number of qubits
            num_classical_bits: Number of classical bits
        """
        self.num_qubits = num_qubits
        self.num_classical_bits = num_classical_bits
        self.gates: List[QuantumGate] = []
        self.id = str(uuid.uuid4())
        
        # Create default registers
        self.qreg = QuantumRegister("q", num_qubits)
        self.creg = ClassicalRegister("c", num_classical_bits) if num_classical_bits > 0 else None
    
    def h(self, qubit: int):
        """Apply Hadamard gate."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("H", [qubit]))
    
    def x(self, qubit: int):
        """Apply Pauli-X gate."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("X", [qubit]))
    
    def y(self, qubit: int):
        """Apply Pauli-Y gate."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("Y", [qubit]))
    
    def z(self, qubit: int):
        """Apply Pauli-Z gate."""
        self._validate_qubit(qubit)
        self.gates.append(QuantumGate("Z", [qubit]))
    
    def cx(self, control: int, target: int):
        """Apply CNOT gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target qubits must be different")
        self.gates.append(QuantumGate("CNOT", [control, target]))
    
    def cz(self, control: int, target: int):
        """Apply controlled-Z gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target qubits must be different")
        self.gates.append(QuantumGate("CZ", [control, target]))
    
    def ry(self, angle: Union[float, str], qubit: int):
        """Apply Y-rotation gate."""
        self._validate_qubit(qubit)
        if isinstance(angle, str):
            # Parameter placeholder
            self.gates.append(QuantumGate("RY", [qubit], parameters=[angle]))
        else:
            self.gates.append(QuantumGate("RY", [qubit], parameters=[float(angle)]))
    
    def rz(self, angle: Union[float, str], qubit: int):
        """Apply Z-rotation gate."""
        self._validate_qubit(qubit)
        if isinstance(angle, str):
            # Parameter placeholder
            self.gates.append(QuantumGate("RZ", [qubit], parameters=[angle]))
        else:
            self.gates.append(QuantumGate("RZ", [qubit], parameters=[float(angle)]))
    
    def crz(self, angle: float, control: int, target: int):
        """Apply controlled Z-rotation gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target qubits must be different")
        self.gates.append(QuantumGate("CRZ", [control, target], parameters=[float(angle)]))
    
    def swap(self, qubit1: int, qubit2: int):
        """Apply SWAP gate."""
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        if qubit1 == qubit2:
            raise ValueError("SWAP qubits must be different")
        self.gates.append(QuantumGate("SWAP", [qubit1, qubit2]))
    
    def mcx(self, control_qubits: List[int], target: int):
        """Apply multi-controlled X gate."""
        for qubit in control_qubits:
            self._validate_qubit(qubit)
        self._validate_qubit(target)
        if target in control_qubits:
            raise ValueError("Target qubit cannot be in control qubits")
        
        all_qubits = control_qubits + [target]
        self.gates.append(QuantumGate("MCX", all_qubits))
    
    def measure(self, qubit: int, classical_bit: int):
        """Add measurement."""
        self._validate_qubit(qubit)
        self._validate_classical_bit(classical_bit)
        self.gates.append(QuantumGate("MEASURE", [qubit], classical_bits=[classical_bit]))
    
    def measure_all(self):
        """Measure all qubits to corresponding classical bits."""
        for i in range(min(self.num_qubits, self.num_classical_bits)):
            self.measure(i, i)
    
    def append(self, other_circuit, qubits: List[int] = None):
        """Append another circuit to this one."""
        if qubits is None:
            qubits = list(range(other_circuit.num_qubits))
        
        if len(qubits) != other_circuit.num_qubits:
            raise ValueError("Number of qubits must match the other circuit")
        
        # Map gates from other circuit
        for gate in other_circuit.gates:
            mapped_qubits = [qubits[q] for q in gate.qubits]
            new_gate = QuantumGate(
                name=gate.name,
                qubits=mapped_qubits,
                parameters=gate.parameters.copy(),
                classical_bits=gate.classical_bits.copy()
            )
            self.gates.append(new_gate)
    
    def inverse(self):
        """Create inverse of the circuit."""
        inverse_circuit = QuantumCircuit(self.num_qubits, self.num_classical_bits)
        
        # Reverse order and invert gates
        for gate in reversed(self.gates):
            if gate.name == "MEASURE":
                # Skip measurements in inverse
                continue
            elif gate.name in ["H", "X", "Y", "Z", "CNOT", "CZ", "SWAP", "MCX"]:
                # Self-inverse gates
                inverse_circuit.gates.append(gate)
            elif gate.name in ["RY", "RZ", "CRZ"]:
                # Invert rotation angles
                inverted_params = [-p if isinstance(p, (int, float)) else f"-{p}" for p in gate.parameters]
                inverse_gate = QuantumGate(gate.name, gate.qubits, inverted_params)
                inverse_circuit.gates.append(inverse_gate)
            else:
                # Unknown gate - assume self-inverse
                inverse_circuit.gates.append(gate)
        
        return inverse_circuit
    
    def _validate_qubit(self, qubit: int):
        """Validate qubit index."""
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"Qubit index {qubit} out of range [0, {self.num_qubits})")
    
    def _validate_classical_bit(self, bit: int):
        """Validate classical bit index."""
        if not 0 <= bit < self.num_classical_bits:
            raise ValueError(f"Classical bit index {bit} out of range [0, {self.num_classical_bits})")
    
    @property
    def depth(self) -> int:
        """Get circuit depth."""
        # Simple depth calculation - could be more sophisticated
        return len(self.gates)
    
    @property
    def gate_count(self) -> int:
        """Get total gate count."""
        return len(self.gates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert circuit to dictionary."""
        return {
            "id": self.id,
            "num_qubits": self.num_qubits,
            "num_classical_bits": self.num_classical_bits,
            "gates": [gate.to_dict() for gate in self.gates],
            "depth": self.depth,
            "gate_count": self.gate_count
        }
    
    def to_qasm(self) -> str:
        """Convert circuit to OpenQASM 2.0."""
        qasm_lines = [
            "OPENQASM 2.0;",
            "include \"qelib1.inc\";",
            f"qreg q[{self.num_qubits}];"
        ]
        
        if self.num_classical_bits > 0:
            qasm_lines.append(f"creg c[{self.num_classical_bits}];")
        
        for gate in self.gates:
            if gate.name == "H":
                qasm_lines.append(f"h q[{gate.qubits[0]}];")
            elif gate.name == "X":
                qasm_lines.append(f"x q[{gate.qubits[0]}];")
            elif gate.name == "Y":
                qasm_lines.append(f"y q[{gate.qubits[0]}];")
            elif gate.name == "Z":
                qasm_lines.append(f"z q[{gate.qubits[0]}];")
            elif gate.name == "CNOT":
                qasm_lines.append(f"cx q[{gate.qubits[0]}], q[{gate.qubits[1]}];")
            elif gate.name == "CZ":
                qasm_lines.append(f"cz q[{gate.qubits[0]}], q[{gate.qubits[1]}];")
            elif gate.name == "RY":
                angle = gate.parameters[0]
                qasm_lines.append(f"ry({angle}) q[{gate.qubits[0]}];")
            elif gate.name == "RZ":
                angle = gate.parameters[0]
                qasm_lines.append(f"rz({angle}) q[{gate.qubits[0]}];")
            elif gate.name == "SWAP":
                qasm_lines.append(f"swap q[{gate.qubits[0]}], q[{gate.qubits[1]}];")
            elif gate.name == "MEASURE":
                qasm_lines.append(f"measure q[{gate.qubits[0]}] -> c[{gate.classical_bits[0]}];")
        
        return "\n".join(qasm_lines)
    
    def __str__(self) -> str:
        """String representation of the circuit."""
        return f"QuantumCircuit({self.num_qubits} qubits, {self.num_classical_bits} cbits, {self.gate_count} gates)"
    
    def __repr__(self) -> str:
        """Detailed representation of the circuit."""
        return f"QuantumCircuit(id='{self.id}', num_qubits={self.num_qubits}, num_classical_bits={self.num_classical_bits}, gates={len(self.gates)})" 