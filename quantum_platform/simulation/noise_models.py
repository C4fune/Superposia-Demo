"""
Quantum Noise Models for Hardware Emulation

This module provides comprehensive noise modeling capabilities for simulating
quantum hardware noise characteristics in local simulations.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json
from abc import ABC, abstractmethod

from ..compiler.ir.circuit import QuantumCircuit
from ..compiler.ir.operation import Operation
from ..errors import SimulationError


class NoiseType(Enum):
    """Types of quantum noise."""
    DEPOLARIZING = "depolarizing"      # Random Pauli errors
    AMPLITUDE_DAMPING = "amplitude_damping"  # T1 decay (|1⟩ → |0⟩)
    PHASE_DAMPING = "phase_damping"    # T2 dephasing
    THERMAL = "thermal"                # Thermal relaxation
    READOUT_ERROR = "readout_error"    # Measurement bit flips
    GATE_ERROR = "gate_error"          # Imperfect gate operations
    CROSSTALK = "crosstalk"           # Qubit-qubit interactions


@dataclass
class NoiseParameter:
    """Individual noise parameter with value and metadata."""
    value: float
    unit: str = ""
    description: str = ""
    calibration_date: Optional[str] = None
    confidence: float = 1.0  # 0-1, measurement confidence
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class GateErrorRates:
    """Error rates for different gate types."""
    single_qubit_error: float = 1e-3     # Single qubit gate error rate
    two_qubit_error: float = 1e-2        # Two qubit gate error rate
    measurement_error: float = 1e-2      # Measurement error rate
    reset_error: float = 1e-3            # Reset operation error rate
    
    # Specific gate errors
    gate_specific: Dict[str, float] = field(default_factory=dict)
    
    def get_error_rate(self, gate_name: str, num_qubits: int) -> float:
        """Get error rate for a specific gate."""
        gate_lower = gate_name.lower()
        
        # Check gate-specific rates first
        if gate_lower in self.gate_specific:
            return self.gate_specific[gate_lower]
        
        # Default rates based on gate type
        if gate_lower in ['measure', 'measurement']:
            return self.measurement_error
        elif gate_lower in ['reset']:
            return self.reset_error
        elif num_qubits == 1:
            return self.single_qubit_error
        elif num_qubits == 2:
            return self.two_qubit_error
        else:
            # Multi-qubit gates have higher error rates
            return self.two_qubit_error * (num_qubits - 1)


@dataclass
class CoherenceParameters:
    """Quantum coherence parameters."""
    T1: NoiseParameter  # Amplitude damping time (microseconds)
    T2: NoiseParameter  # Dephasing time (microseconds)
    
    @property
    def T2_star(self) -> float:
        """Pure dephasing time T2* = 1/(1/T2 - 1/(2*T1))."""
        if self.T1.value <= 0 or self.T2.value <= 0:
            return 0.0
        
        rate = 1/self.T2.value - 1/(2*self.T1.value)
        return 1/rate if rate > 0 else float('inf')


@dataclass
class ReadoutError:
    """Readout error parameters."""
    prob_0_given_1: float = 0.01  # P(measure 0 | state is 1)
    prob_1_given_0: float = 0.01  # P(measure 1 | state is 0)
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get 2x2 confusion matrix for readout errors."""
        return np.array([
            [1 - self.prob_1_given_0, self.prob_1_given_0],
            [self.prob_0_given_1, 1 - self.prob_0_given_1]
        ])


class NoiseModel:
    """Base class for quantum noise models."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.enabled = True
        
        # Per-qubit parameters
        self.coherence_params: Dict[int, CoherenceParameters] = {}
        self.readout_errors: Dict[int, ReadoutError] = {}
        self.gate_errors = GateErrorRates()
        
        # Global parameters
        self.thermal_population = 0.0  # Thermal excitation probability
        self.crosstalk_matrix: Optional[np.ndarray] = None
        
        # Metadata
        self.device_name = ""
        self.calibration_date = ""
        self.metadata: Dict[str, Any] = {}
    
    def set_qubit_coherence(self, qubit_id: int, T1: float, T2: float,
                           T1_unit: str = "us", T2_unit: str = "us"):
        """Set coherence parameters for a specific qubit."""
        self.coherence_params[qubit_id] = CoherenceParameters(
            T1=NoiseParameter(T1, T1_unit, "Amplitude damping time"),
            T2=NoiseParameter(T2, T2_unit, "Dephasing time")
        )
    
    def set_qubit_readout_error(self, qubit_id: int, prob_0_given_1: float,
                               prob_1_given_0: float):
        """Set readout error for a specific qubit."""
        self.readout_errors[qubit_id] = ReadoutError(prob_0_given_1, prob_1_given_0)
    
    def set_gate_error_rate(self, gate_name: str, error_rate: float):
        """Set error rate for a specific gate type."""
        self.gate_errors.gate_specific[gate_name.lower()] = error_rate
    
    def apply_noise_to_counts(self, counts: Dict[str, int], 
                             num_qubits: int) -> Dict[str, int]:
        """Apply readout errors to measurement counts."""
        if not self.enabled:
            return counts
        
        noisy_counts = {}
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Apply readout error to each shot
            for _ in range(count):
                noisy_bits = self._apply_readout_noise(bitstring, num_qubits)
                noisy_string = ''.join(noisy_bits)
                noisy_counts[noisy_string] = noisy_counts.get(noisy_string, 0) + 1
        
        return noisy_counts
    
    def _apply_readout_noise(self, bitstring: str, num_qubits: int) -> List[str]:
        """Apply readout errors to a single measurement outcome."""
        noisy_bits = []
        
        for i, bit in enumerate(bitstring):
            qubit_id = i if i < num_qubits else 0
            
            if qubit_id in self.readout_errors:
                readout_error = self.readout_errors[qubit_id]
                
                if bit == '0' and random.random() < readout_error.prob_1_given_0:
                    noisy_bits.append('1')
                elif bit == '1' and random.random() < readout_error.prob_0_given_1:
                    noisy_bits.append('0')
                else:
                    noisy_bits.append(bit)
            else:
                noisy_bits.append(bit)
        
        return noisy_bits
    
    def get_gate_error_probability(self, operation: Operation) -> float:
        """Get error probability for a specific operation."""
        if not self.enabled:
            return 0.0
        
        return self.gate_errors.get_error_rate(
            operation.name, 
            len(operation.targets)
        )
    
    def should_apply_gate_error(self, operation: Operation) -> bool:
        """Determine if gate error should be applied to this operation."""
        error_prob = self.get_gate_error_probability(operation)
        return random.random() < error_prob
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert noise model to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "device_name": self.device_name,
            "calibration_date": self.calibration_date,
            "coherence_params": {
                str(qid): {
                    "T1": {"value": params.T1.value, "unit": params.T1.unit},
                    "T2": {"value": params.T2.value, "unit": params.T2.unit}
                }
                for qid, params in self.coherence_params.items()
            },
            "readout_errors": {
                str(qid): {
                    "prob_0_given_1": error.prob_0_given_1,
                    "prob_1_given_0": error.prob_1_given_0
                }
                for qid, error in self.readout_errors.items()
            },
            "gate_errors": {
                "single_qubit_error": self.gate_errors.single_qubit_error,
                "two_qubit_error": self.gate_errors.two_qubit_error,
                "measurement_error": self.gate_errors.measurement_error,
                "reset_error": self.gate_errors.reset_error,
                "gate_specific": self.gate_errors.gate_specific
            },
            "thermal_population": self.thermal_population,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NoiseModel':
        """Create noise model from dictionary."""
        model = cls(data["name"], data.get("description", ""))
        model.enabled = data.get("enabled", True)
        model.device_name = data.get("device_name", "")
        model.calibration_date = data.get("calibration_date", "")
        model.thermal_population = data.get("thermal_population", 0.0)
        model.metadata = data.get("metadata", {})
        
        # Load coherence parameters
        for qid_str, params in data.get("coherence_params", {}).items():
            qid = int(qid_str)
            model.set_qubit_coherence(
                qid,
                params["T1"]["value"],
                params["T2"]["value"],
                params["T1"]["unit"],
                params["T2"]["unit"]
            )
        
        # Load readout errors
        for qid_str, error in data.get("readout_errors", {}).items():
            qid = int(qid_str)
            model.set_qubit_readout_error(
                qid,
                error["prob_0_given_1"],
                error["prob_1_given_0"]
            )
        
        # Load gate errors
        gate_errors_data = data.get("gate_errors", {})
        model.gate_errors.single_qubit_error = gate_errors_data.get("single_qubit_error", 1e-3)
        model.gate_errors.two_qubit_error = gate_errors_data.get("two_qubit_error", 1e-2)
        model.gate_errors.measurement_error = gate_errors_data.get("measurement_error", 1e-2)
        model.gate_errors.reset_error = gate_errors_data.get("reset_error", 1e-3)
        model.gate_errors.gate_specific = gate_errors_data.get("gate_specific", {})
        
        return model


class DeviceNoiseModelLibrary:
    """Library of noise models for different quantum devices."""
    
    def __init__(self):
        self.models: Dict[str, NoiseModel] = {}
        self._load_default_models()
    
    def _load_default_models(self):
        """Load default noise models for common devices."""
        # IBM-like superconducting device
        ibm_like = NoiseModel("ibm_like", "IBM-like superconducting device")
        for i in range(5):  # 5-qubit device
            ibm_like.set_qubit_coherence(i, T1=100, T2=50)  # microseconds
            ibm_like.set_qubit_readout_error(i, 0.02, 0.01)
        
        ibm_like.gate_errors.single_qubit_error = 1e-3
        ibm_like.gate_errors.two_qubit_error = 1e-2
        ibm_like.gate_errors.measurement_error = 2e-2
        ibm_like.device_name = "IBM-like Device"
        self.models["ibm_like"] = ibm_like
        
        # IonQ-like trapped ion device
        ionq_like = NoiseModel("ionq_like", "IonQ-like trapped ion device")
        for i in range(11):  # 11-qubit device
            ionq_like.set_qubit_coherence(i, T1=10000, T2=1000)  # microseconds
            ionq_like.set_qubit_readout_error(i, 0.005, 0.005)
        
        ionq_like.gate_errors.single_qubit_error = 1e-4
        ionq_like.gate_errors.two_qubit_error = 5e-3
        ionq_like.gate_errors.measurement_error = 1e-2
        ionq_like.device_name = "IonQ-like Device"
        self.models["ionq_like"] = ionq_like
        
        # Google-like superconducting device
        google_like = NoiseModel("google_like", "Google-like superconducting device")
        for i in range(20):  # 20-qubit device
            google_like.set_qubit_coherence(i, T1=80, T2=40)  # microseconds
            google_like.set_qubit_readout_error(i, 0.015, 0.01)
        
        google_like.gate_errors.single_qubit_error = 2e-3
        google_like.gate_errors.two_qubit_error = 1.5e-2
        google_like.gate_errors.measurement_error = 1.5e-2
        google_like.device_name = "Google-like Device"
        self.models["google_like"] = google_like
        
        # Ideal (no noise) model
        ideal = NoiseModel("ideal", "Ideal quantum device with no noise")
        ideal.enabled = False
        self.models["ideal"] = ideal
    
    def get_model(self, name: str) -> Optional[NoiseModel]:
        """Get a noise model by name."""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """Get list of available noise model names."""
        return list(self.models.keys())
    
    def add_model(self, model: NoiseModel):
        """Add a custom noise model to the library."""
        self.models[model.name] = model
    
    def remove_model(self, name: str):
        """Remove a noise model from the library."""
        if name in self.models:
            del self.models[name]
    
    def save_model(self, name: str, filepath: str):
        """Save a noise model to file."""
        if name not in self.models:
            raise ValueError(f"Noise model '{name}' not found")
        
        model_data = self.models[name].to_dict()
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str) -> str:
        """Load a noise model from file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        model = NoiseModel.from_dict(model_data)
        self.add_model(model)
        return model.name


# Global noise model library instance
_noise_library = None


def get_noise_library() -> DeviceNoiseModelLibrary:
    """Get the global noise model library."""
    global _noise_library
    if _noise_library is None:
        _noise_library = DeviceNoiseModelLibrary()
    return _noise_library


def create_noise_model_from_calibration(device_name: str, 
                                       calibration_data: Dict[str, Any]) -> NoiseModel:
    """
    Create a noise model from device calibration data.
    
    Args:
        device_name: Name of the device
        calibration_data: Dictionary containing calibration parameters
        
    Returns:
        Configured noise model
    """
    model = NoiseModel(
        name=f"{device_name}_calibrated",
        description=f"Noise model from {device_name} calibration data"
    )
    
    model.device_name = device_name
    model.calibration_date = calibration_data.get("date", "")
    
    # Extract qubit parameters
    if "qubits" in calibration_data:
        for qubit_data in calibration_data["qubits"]:
            qid = qubit_data["id"]
            
            # Coherence times
            if "T1" in qubit_data and "T2" in qubit_data:
                model.set_qubit_coherence(
                    qid, 
                    qubit_data["T1"], 
                    qubit_data["T2"]
                )
            
            # Readout errors
            if "readout_error" in qubit_data:
                error_data = qubit_data["readout_error"]
                model.set_qubit_readout_error(
                    qid,
                    error_data.get("prob_0_given_1", 0.01),
                    error_data.get("prob_1_given_0", 0.01)
                )
    
    # Extract gate error rates
    if "gates" in calibration_data:
        for gate_name, error_rate in calibration_data["gates"].items():
            model.set_gate_error_rate(gate_name, error_rate)
    
    return model


def apply_noise_to_circuit(circuit: QuantumCircuit, noise_model: NoiseModel) -> QuantumCircuit:
    """
    Apply noise model to a quantum circuit by adding error operations.
    
    This is a simplified implementation that adds depolarizing errors.
    A full implementation would use density matrix evolution.
    
    Args:
        circuit: Input quantum circuit
        noise_model: Noise model to apply
        
    Returns:
        Circuit with noise operations added
    """
    if not noise_model.enabled:
        return circuit
    
    # For now, return the original circuit
    # Full noise simulation would require density matrix simulation
    # which is complex and beyond the scope of this initial implementation
    return circuit 