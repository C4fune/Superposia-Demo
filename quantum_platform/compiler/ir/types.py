"""
Core types for the Quantum IR system.

This module defines fundamental types used throughout the IR,
including parameter handling for variational circuits.
"""

from typing import Union, Any, Dict, Optional
from abc import ABC, abstractmethod
import sympy
from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterValue:
    """
    Represents a parameter value that can be either concrete or symbolic.
    
    This is used for parameterized gates where the parameter might be:
    - A concrete numeric value (float, int)
    - A symbolic expression (sympy.Symbol or expression)
    """
    value: Union[float, int, sympy.Basic]
    
    def __post_init__(self):
        """Validate the parameter value."""
        if not isinstance(self.value, (int, float, sympy.Basic)):
            raise TypeError(f"Parameter value must be numeric or sympy expression, got {type(self.value)}")
    
    @property
    def is_symbolic(self) -> bool:
        """Check if this parameter contains symbolic values."""
        return isinstance(self.value, sympy.Basic) and self.value.free_symbols
    
    @property
    def is_concrete(self) -> bool:
        """Check if this parameter is a concrete numeric value."""
        return isinstance(self.value, (int, float)) or (
            isinstance(self.value, sympy.Basic) and not self.value.free_symbols
        )
    
    def substitute(self, substitutions: Dict[sympy.Symbol, Union[float, int]]) -> 'ParameterValue':
        """
        Create a new ParameterValue with symbolic parameters substituted.
        
        Args:
            substitutions: Dictionary mapping symbols to concrete values
            
        Returns:
            New ParameterValue with substitutions applied
        """
        if not self.is_symbolic:
            return self
        
        if isinstance(self.value, sympy.Basic):
            new_value = self.value.subs(substitutions)
            # Try to convert to float if it's now concrete
            if not new_value.free_symbols:
                try:
                    new_value = float(new_value)
                except (TypeError, ValueError):
                    pass  # Keep as sympy expression
            return ParameterValue(new_value)
        
        return self
    
    def __float__(self) -> float:
        """Convert to float if possible."""
        if isinstance(self.value, (int, float)):
            return float(self.value)
        elif isinstance(self.value, sympy.Basic) and not self.value.free_symbols:
            return float(self.value)
        else:
            raise ValueError(f"Cannot convert symbolic parameter {self.value} to float")
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __repr__(self) -> str:
        return f"ParameterValue({self.value!r})"


class SymbolicParameter:
    """
    Factory class for creating symbolic parameters.
    
    This provides a convenient way to create sympy symbols for use in
    parameterized quantum circuits.
    """
    
    @staticmethod
    def create(name: str) -> sympy.Symbol:
        """Create a new symbolic parameter with the given name."""
        return sympy.Symbol(name, real=True)
    
    @staticmethod
    def theta() -> sympy.Symbol:
        """Create a theta parameter (commonly used for rotations)."""
        return sympy.Symbol('theta', real=True)
    
    @staticmethod
    def phi() -> sympy.Symbol:
        """Create a phi parameter (commonly used for phase gates)."""
        return sympy.Symbol('phi', real=True)
    
    @staticmethod
    def lambda_() -> sympy.Symbol:
        """Create a lambda parameter (note: lambda is a Python keyword)."""
        return sympy.Symbol('lambda', real=True)


# Type aliases for common parameter types
Parameter = Union[float, int, sympy.Basic, ParameterValue]
ParameterDict = Dict[str, Parameter] 