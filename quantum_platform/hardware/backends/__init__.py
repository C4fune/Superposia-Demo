"""
Quantum Hardware Backend Implementations

This module contains concrete implementations of quantum hardware backends
for different providers.
"""

from .local_simulator import LocalSimulatorBackend
from .ibm_backend import IBMQBackend

__all__ = [
    'LocalSimulatorBackend',
    'IBMQBackend'
] 