"""
Noisy Quantum Simulator

This module provides a quantum simulator that includes realistic noise effects
using Monte Carlo simulation methods.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..compiler.ir.circuit import QuantumCircuit
from ..compiler.ir.operation import Operation
from .statevector import StateVectorSimulator, SimulationResult
from .noise_models import NoiseModel, NoiseType
from ..errors import SimulationError


@dataclass
class NoisySimulationResult(SimulationResult):
    """Result from noisy quantum simulation."""
    noise_model_name: str = ""
    error_events: List[Dict[str, Any]] = None
    ideal_counts: Optional[Dict[str, int]] = None
    noise_overhead: float = 0.0
    
    def __post_init__(self):
        if self.error_events is None:
            self.error_events = []


class NoiseSimulationEngine:
    """Engine for applying noise effects during simulation."""
    
    def __init__(self, noise_model: NoiseModel):
        self.noise_model = noise_model
        self.error_log: List[Dict[str, Any]] = []
        
    def should_apply_error(self, operation: Operation) -> bool:
        """Determine if an error should be applied to this operation."""
        if not self.noise_model.enabled:
            return False
        
        error_prob = self.noise_model.get_gate_error_probability(operation)
        return random.random() < error_prob
    
    def generate_pauli_error(self, num_qubits: int) -> List[str]:
        """Generate random Pauli error for given number of qubits."""
        pauli_ops = ['I', 'X', 'Y', 'Z']
        return [random.choice(pauli_ops) for _ in range(num_qubits)]
    
    def apply_gate_error(self, operation: Operation, state: np.ndarray) -> np.ndarray:
        """Apply gate error to quantum state."""
        if not self.should_apply_error(operation):
            return state
        
        # Simple depolarizing error model
        # In a full implementation, this would apply the specific error to the state
        error_type = random.choice(['X', 'Y', 'Z'])
        qubit_id = operation.targets[0].id if operation.targets else 0
        
        # Log error event
        self.error_log.append({
            'operation': operation.name,
            'qubit': qubit_id,
            'error_type': error_type,
            'probability': self.noise_model.get_gate_error_probability(operation)
        })
        
        # For simplicity, return unchanged state
        # Real implementation would apply Pauli matrices
        return state
    
    def apply_measurement_error(self, bitstring: str) -> str:
        """Apply measurement errors to measurement outcome."""
        if not self.noise_model.enabled:
            return bitstring
        
        noisy_bits = []
        for i, bit in enumerate(bitstring):
            qubit_id = i
            
            if qubit_id in self.noise_model.readout_errors:
                readout_error = self.noise_model.readout_errors[qubit_id]
                
                if bit == '0' and random.random() < readout_error.prob_1_given_0:
                    noisy_bits.append('1')
                    self.error_log.append({
                        'type': 'readout_error',
                        'qubit': qubit_id,
                        'original': '0',
                        'measured': '1'
                    })
                elif bit == '1' and random.random() < readout_error.prob_0_given_1:
                    noisy_bits.append('0')
                    self.error_log.append({
                        'type': 'readout_error',
                        'qubit': qubit_id,
                        'original': '1',
                        'measured': '0'
                    })
                else:
                    noisy_bits.append(bit)
            else:
                noisy_bits.append(bit)
        
        return ''.join(noisy_bits)


class NoisyQuantumSimulator:
    """Monte Carlo quantum simulator with noise effects."""
    
    def __init__(self, noise_model: Optional[NoiseModel] = None):
        self.noise_model = noise_model
        self.ideal_simulator = StateVectorSimulator()
        self._random_seed = None
        
    def set_noise_model(self, noise_model: NoiseModel):
        """Set the noise model for simulation."""
        self.noise_model = noise_model
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducible noise simulation."""
        self._random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def run(self, circuit: QuantumCircuit, shots: int = 1000,
            compare_ideal: bool = False) -> NoisySimulationResult:
        """
        Run noisy simulation of quantum circuit.
        
        Args:
            circuit: Quantum circuit to simulate
            shots: Number of measurement shots
            compare_ideal: Whether to also run ideal simulation for comparison
            
        Returns:
            Noisy simulation result
        """
        start_time = time.time()
        
        if self._random_seed is not None:
            random.seed(self._random_seed)
            np.random.seed(self._random_seed)
        
        # Run ideal simulation if requested
        ideal_result = None
        if compare_ideal or self.noise_model is None:
            ideal_result = self.ideal_simulator.run(circuit, shots)
        
        # If no noise model, return ideal result
        if self.noise_model is None or not self.noise_model.enabled:
            result = NoisySimulationResult(
                counts=ideal_result.counts,
                execution_time=ideal_result.execution_time,
                shots=shots,
                success=True,
                noise_model_name="none",
                ideal_counts=ideal_result.counts if compare_ideal else None
            )
            return result
        
        # Run noisy simulation
        noisy_counts = self._run_noisy_simulation(circuit, shots)
        execution_time = time.time() - start_time
        
        # Calculate noise overhead
        noise_overhead = 0.0
        if ideal_result:
            noise_overhead = self._calculate_noise_overhead(
                ideal_result.counts, noisy_counts
            )
        
        # Create result
        result = NoisySimulationResult(
            counts=noisy_counts,
            execution_time=execution_time,
            shots=shots,
            success=True,
            noise_model_name=self.noise_model.name,
            ideal_counts=ideal_result.counts if ideal_result else None,
            noise_overhead=noise_overhead
        )
        
        return result
    
    def _run_noisy_simulation(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Run Monte Carlo noisy simulation."""
        if shots <= 100:
            # For small shot counts, run sequentially
            return self._run_sequential_simulation(circuit, shots)
        else:
            # For large shot counts, use parallel execution
            return self._run_parallel_simulation(circuit, shots)
    
    def _run_sequential_simulation(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Run sequential Monte Carlo simulation."""
        noise_engine = NoiseSimulationEngine(self.noise_model)
        counts = {}
        
        for shot in range(shots):
            # Run single shot with noise
            bitstring = self._simulate_single_shot(circuit, noise_engine)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return counts
    
    def _run_parallel_simulation(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Run parallel Monte Carlo simulation for better performance."""
        # Split shots across threads
        num_threads = min(4, max(1, shots // 50))
        shots_per_thread = shots // num_threads
        remaining_shots = shots % num_threads
        
        counts = {}
        counts_lock = threading.Lock()
        
        def worker(thread_shots: int):
            thread_counts = self._run_sequential_simulation(circuit, thread_shots)
            with counts_lock:
                for bitstring, count in thread_counts.items():
                    counts[bitstring] = counts.get(bitstring, 0) + count
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            # Submit worker tasks
            for i in range(num_threads):
                thread_shots = shots_per_thread
                if i < remaining_shots:
                    thread_shots += 1
                
                if thread_shots > 0:
                    futures.append(executor.submit(worker, thread_shots))
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions
        
        return counts
    
    def _simulate_single_shot(self, circuit: QuantumCircuit, 
                             noise_engine: NoiseSimulationEngine) -> str:
        """Simulate a single measurement shot with noise."""
        # For simplicity, use ideal simulation then add readout errors
        # Full implementation would simulate noise throughout execution
        
        ideal_result = self.ideal_simulator.run(circuit, shots=1)
        ideal_bitstring = list(ideal_result.counts.keys())[0]
        
        # Apply readout errors
        noisy_bitstring = noise_engine.apply_measurement_error(ideal_bitstring)
        
        return noisy_bitstring
    
    def _calculate_noise_overhead(self, ideal_counts: Dict[str, int],
                                 noisy_counts: Dict[str, int]) -> float:
        """Calculate the noise overhead metric."""
        # Use total variation distance as noise overhead metric
        total_shots = sum(ideal_counts.values())
        
        if total_shots == 0:
            return 0.0
        
        # Convert to probability distributions
        ideal_probs = {k: v/total_shots for k, v in ideal_counts.items()}
        noisy_probs = {k: v/total_shots for k, v in noisy_counts.items()}
        
        # Calculate total variation distance
        all_outcomes = set(ideal_probs.keys()) | set(noisy_probs.keys())
        
        tv_distance = 0.0
        for outcome in all_outcomes:
            ideal_prob = ideal_probs.get(outcome, 0.0)
            noisy_prob = noisy_probs.get(outcome, 0.0)
            tv_distance += abs(ideal_prob - noisy_prob)
        
        return tv_distance / 2.0
    
    def run_comparative_analysis(self, circuit: QuantumCircuit, shots: int = 1000
                               ) -> Dict[str, Any]:
        """Run comparative analysis between ideal and noisy simulation."""
        if self.noise_model is None:
            raise SimulationError("No noise model set for comparative analysis")
        
        # Run ideal simulation
        ideal_result = self.ideal_simulator.run(circuit, shots)
        
        # Run noisy simulation
        noisy_result = self.run(circuit, shots, compare_ideal=False)
        
        # Calculate metrics
        fidelity = self._calculate_fidelity(ideal_result.counts, noisy_result.counts)
        hellinger_distance = self._calculate_hellinger_distance(
            ideal_result.counts, noisy_result.counts
        )
        
        return {
            'ideal_counts': ideal_result.counts,
            'noisy_counts': noisy_result.counts,
            'fidelity': fidelity,
            'hellinger_distance': hellinger_distance,
            'total_variation_distance': noisy_result.noise_overhead,
            'noise_model': self.noise_model.name,
            'execution_time_ideal': ideal_result.execution_time,
            'execution_time_noisy': noisy_result.execution_time,
            'overhead_ratio': noisy_result.execution_time / ideal_result.execution_time
        }
    
    def _calculate_fidelity(self, counts1: Dict[str, int], 
                           counts2: Dict[str, int]) -> float:
        """Calculate fidelity between two probability distributions."""
        total1 = sum(counts1.values())
        total2 = sum(counts2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        # Convert to probabilities
        probs1 = {k: v/total1 for k, v in counts1.items()}
        probs2 = {k: v/total2 for k, v in counts2.items()}
        
        # Calculate fidelity
        all_outcomes = set(probs1.keys()) | set(probs2.keys())
        fidelity = 0.0
        
        for outcome in all_outcomes:
            p1 = probs1.get(outcome, 0.0)
            p2 = probs2.get(outcome, 0.0)
            fidelity += np.sqrt(p1 * p2)
        
        return fidelity
    
    def _calculate_hellinger_distance(self, counts1: Dict[str, int],
                                     counts2: Dict[str, int]) -> float:
        """Calculate Hellinger distance between two probability distributions."""
        fidelity = self._calculate_fidelity(counts1, counts2)
        return np.sqrt(1 - fidelity)


# Factory functions for creating pre-configured noisy simulators

def create_device_simulator(device_type: str) -> NoisyQuantumSimulator:
    """
    Create a noisy simulator for a specific device type.
    
    Args:
        device_type: Type of device ('ibm_like', 'ionq_like', 'google_like', 'ideal')
        
    Returns:
        Configured noisy simulator
    """
    from .noise_models import get_noise_library
    
    library = get_noise_library()
    noise_model = library.get_model(device_type)
    
    if noise_model is None:
        raise ValueError(f"Unknown device type: {device_type}")
    
    return NoisyQuantumSimulator(noise_model)


def create_custom_simulator(noise_model: NoiseModel) -> NoisyQuantumSimulator:
    """Create a noisy simulator with a custom noise model."""
    return NoisyQuantumSimulator(noise_model)


def create_ideal_simulator() -> NoisyQuantumSimulator:
    """Create an ideal (noiseless) simulator."""
    return NoisyQuantumSimulator(noise_model=None) 