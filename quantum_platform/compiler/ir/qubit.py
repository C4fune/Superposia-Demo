"""
Qubit representation for the Quantum IR.

This module defines the Qubit class which represents quantum bits in the IR,
including allocation, lifetime management, and state tracking.
"""

from typing import Optional, Any, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid


class QubitState(Enum):
    """Represents the lifecycle state of a qubit."""
    ALLOCATED = "allocated"      # Qubit is allocated and ready for use
    IN_USE = "in_use"           # Qubit is actively being used in operations
    MEASURED = "measured"        # Qubit has been measured (state collapsed)
    FREED = "freed"             # Qubit has been deallocated and can be reused
    RESET = "reset"             # Qubit has been reset to |0> state


@dataclass
class Qubit:
    """
    Represents a quantum bit in the IR.
    
    Each qubit has a unique identifier and tracks its usage throughout
    the circuit. This enables proper lifetime management and optimization.
    
    Attributes:
        id: Unique integer identifier for this qubit (0-indexed)
        name: Optional human-readable name
        state: Current lifecycle state of the qubit
        register_name: Optional name of the register this qubit belongs to
        metadata: Additional metadata for extensions and debugging
    """
    id: int
    name: Optional[str] = None
    state: QubitState = QubitState.ALLOCATED
    register_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Internal tracking
    _uuid: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)
    _allocation_order: Optional[int] = field(default=None, init=False)
    
    def __post_init__(self):
        """Validate qubit parameters."""
        if self.id < 0:
            raise ValueError(f"Qubit ID must be non-negative, got {self.id}")
        
        # Ensure name is valid if provided
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(f"Qubit name must be string, got {type(self.name)}")
    
    @property
    def uuid(self) -> str:
        """Get the unique UUID for this qubit instance."""
        return self._uuid
    
    @property
    def is_allocated(self) -> bool:
        """Check if qubit is in allocated state."""
        return self.state == QubitState.ALLOCATED
    
    @property
    def is_in_use(self) -> bool:
        """Check if qubit is currently in use."""
        return self.state == QubitState.IN_USE
    
    @property
    def is_measured(self) -> bool:
        """Check if qubit has been measured."""
        return self.state == QubitState.MEASURED
    
    @property
    def is_freed(self) -> bool:
        """Check if qubit has been freed."""
        return self.state == QubitState.FREED
    
    @property
    def is_reset(self) -> bool:
        """Check if qubit has been reset."""
        return self.state == QubitState.RESET
    
    @property
    def can_be_reused(self) -> bool:
        """Check if qubit can be reused (freed or reset)."""
        return self.state in (QubitState.FREED, QubitState.RESET)
    
    def mark_in_use(self) -> None:
        """Mark qubit as being actively used."""
        if self.state == QubitState.FREED:
            raise RuntimeError(f"Cannot use freed qubit {self}")
        self.state = QubitState.IN_USE
    
    def mark_measured(self) -> None:
        """Mark qubit as measured (state collapsed)."""
        self.state = QubitState.MEASURED
    
    def mark_freed(self) -> None:
        """Mark qubit as freed and available for reuse."""
        self.state = QubitState.FREED
    
    def mark_reset(self) -> None:
        """Mark qubit as reset to |0> state."""
        self.state = QubitState.RESET
    
    def reset_to_allocated(self) -> None:
        """Reset qubit state back to allocated (for reuse)."""
        if not self.can_be_reused:
            raise RuntimeError(f"Cannot reset qubit {self} in state {self.state}")
        self.state = QubitState.ALLOCATED
    
    def clone_with_new_id(self, new_id: int) -> 'Qubit':
        """
        Create a copy of this qubit with a new ID.
        
        This is useful for qubit remapping during optimization.
        """
        return Qubit(
            id=new_id,
            name=self.name,
            state=self.state,
            register_name=self.register_name,
            metadata=self.metadata.copy()
        )
    
    def __hash__(self) -> int:
        """Hash based on UUID for use in sets/dicts."""
        return hash(self._uuid)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on UUID."""
        if not isinstance(other, Qubit):
            return NotImplemented
        return self._uuid == other._uuid
    
    def __lt__(self, other: 'Qubit') -> bool:
        """Comparison for sorting (by ID)."""
        return self.id < other.id
    
    def __str__(self) -> str:
        """String representation."""
        if self.name:
            return f"q{self.id}({self.name})"
        return f"q{self.id}"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        parts = [f"id={self.id}", f"state={self.state.value}"]
        if self.name:
            parts.insert(1, f"name={self.name!r}")
        if self.register_name:
            parts.append(f"register={self.register_name!r}")
        return f"Qubit({', '.join(parts)})"


class QubitRegister:
    """
    Represents a register of qubits for grouped allocation.
    
    This is useful for algorithms that work with groups of qubits
    and want to maintain logical grouping in the IR.
    """
    
    def __init__(self, name: str, size: int, start_id: int = 0):
        """
        Initialize a qubit register.
        
        Args:
            name: Name of the register
            size: Number of qubits in the register
            start_id: Starting qubit ID for this register
        """
        self.name = name
        self.size = size
        self.start_id = start_id
        
        # Create qubits for this register
        self.qubits = [
            Qubit(
                id=start_id + i,
                name=f"{name}[{i}]",
                register_name=name
            )
            for i in range(size)
        ]
    
    def __getitem__(self, index: int) -> Qubit:
        """Access qubit by index within the register."""
        if not 0 <= index < self.size:
            raise IndexError(f"Qubit index {index} out of range for register of size {self.size}")
        return self.qubits[index]
    
    def __iter__(self):
        """Iterate over qubits in the register."""
        return iter(self.qubits)
    
    def __len__(self) -> int:
        """Get number of qubits in register."""
        return self.size
    
    def __str__(self) -> str:
        return f"{self.name}[{self.size}]"
    
    def __repr__(self) -> str:
        return f"QubitRegister(name={self.name!r}, size={self.size}, start_id={self.start_id})" 