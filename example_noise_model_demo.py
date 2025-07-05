#!/usr/bin/env python3
"""
Quantum Noise Model Demonstration

This script demonstrates the local emulated quantum hardware capabilities
with realistic noise models for different quantum devices.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List

# Import quantum platform components
from quantum_platform.compiler.language.dsl import QuantumProgram, allocate
from quantum_platform.compiler.language.operations import H, X, Y, Z, RX, RY, RZ, CNOT, measure
from quantum_platform.simulation.noise_models import (
    NoiseModel, get_noise_library, create_noise_model_from_calibration
)
from quantum_platform.simulation.noisy_simulator import (
    NoisyQuantumSimulator, create_device_simulator
)
from quantum_platform.hardware.backends.noisy_simulator_backend import NoisySimulatorBackend
from quantum_platform.providers.provider_manager import get_provider_manager


def create_bell_state_circuit():
    """Create a Bell state preparation circuit."""
    with QuantumProgram(name="bell_state") as qp:
        q0 = qp.allocate_qubit("q0")
        q1 = qp.allocate_qubit("q1")
        
        H(q0)
        CNOT(q0, q1)
        
        measure(q0)
        measure(q1)
    
    return qp.circuit


def create_ghz_state_circuit(num_qubits: int = 3):
    """Create a GHZ state preparation circuit."""
    with QuantumProgram(name=f"ghz_{num_qubits}") as qp:
        qubits = [qp.allocate_qubit(f"q{i}") for i in range(num_qubits)]
        
        # Prepare GHZ state: |000⟩ + |111⟩
        H(qubits[0])
        for i in range(1, num_qubits):
            CNOT(qubits[0], qubits[i])
        
        # Measure all qubits
        for qubit in qubits:
            measure(qubit)
    
    return qp.circuit


def create_randomized_benchmarking_circuit(depth: int = 5):
    """Create a randomized benchmarking circuit."""
    with QuantumProgram(name=f"rb_depth_{depth}") as qp:
        q0 = qp.allocate_qubit("q0")
        
        # Random sequence of Clifford gates
        clifford_gates = [H, X, Y, Z]
        
        for i in range(depth):
            gate = np.random.choice(clifford_gates)
            gate(q0)
        
        measure(q0)
    
    return qp.circuit


def demonstrate_noise_models():
    """Demonstrate different noise models and their effects."""
    print("🔬 Quantum Noise Model Demonstration")
    print("=" * 50)
    
    # Get noise model library
    library = get_noise_library()
    
    print("\n📚 Available Noise Models:")
    for model_name in library.list_models():
        model = library.get_model(model_name)
        print(f"  • {model_name}: {model.description}")
    
    # Create test circuit
    circuit = create_bell_state_circuit()
    shots = 1000
    
    print(f"\n🧪 Testing with Bell State Circuit (shots={shots})")
    print(f"Expected ideal result: ~50% |00⟩, ~50% |11⟩")
    
    results = {}
    
    # Test each noise model
    for model_name in ["ideal", "ibm_like", "ionq_like", "google_like"]:
        print(f"\n🔍 Testing {model_name} device...")
        
        # Create simulator for this device type
        simulator = create_device_simulator(model_name)
        
        # Run simulation
        start_time = time.time()
        result = simulator.run(circuit, shots=shots, compare_ideal=True)
        execution_time = time.time() - start_time
        
        results[model_name] = {
            'counts': result.counts,
            'ideal_counts': result.ideal_counts,
            'noise_overhead': getattr(result, 'noise_overhead', 0.0),
            'execution_time': execution_time
        }
        
        # Display results
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Noise overhead: {results[model_name]['noise_overhead']:.4f}")
        print(f"  Noisy counts: {result.counts}")
        
        if result.ideal_counts and model_name != "ideal":
            print(f"  Ideal counts: {result.ideal_counts}")
    
    return results


def demonstrate_noise_characterization():
    """Demonstrate noise characterization and calibration."""
    print("\n🔧 Noise Model Characterization")
    print("=" * 40)
    
    # Create custom noise model from calibration data
    calibration_data = {
        "date": "2024-01-15",
        "qubits": [
            {
                "id": 0,
                "T1": 85.3,  # microseconds
                "T2": 42.1,  # microseconds
                "readout_error": {"prob_0_given_1": 0.025, "prob_1_given_0": 0.015}
            },
            {
                "id": 1,
                "T1": 78.9,
                "T2": 38.7,
                "readout_error": {"prob_0_given_1": 0.028, "prob_1_given_0": 0.018}
            }
        ],
        "gates": {
            "single_qubit": 0.0012,
            "cx": 0.0089,
            "measure": 0.018
        }
    }
    
    # Create noise model from calibration
    custom_model = create_noise_model_from_calibration("Custom_Device", calibration_data)
    
    print(f"📊 Created custom noise model: {custom_model.name}")
    print(f"   Device: {custom_model.device_name}")
    print(f"   Calibration date: {custom_model.calibration_date}")
    
    # Display noise characteristics
    print("\n   Qubit characteristics:")
    for qid, params in custom_model.coherence_params.items():
        print(f"     Qubit {qid}: T1={params.T1.value:.1f}μs, T2={params.T2.value:.1f}μs")
    
    print("\n   Gate error rates:")
    print(f"     Single qubit: {custom_model.gate_errors.single_qubit_error:.4f}")
    print(f"     Two qubit: {custom_model.gate_errors.two_qubit_error:.4f}")
    print(f"     Measurement: {custom_model.gate_errors.measurement_error:.4f}")
    
    # Test custom model
    simulator = NoisyQuantumSimulator(custom_model)
    circuit = create_bell_state_circuit()
    
    result = simulator.run(circuit, shots=1000, compare_ideal=True)
    print(f"\n   Test results with custom model:")
    print(f"     Noisy counts: {result.counts}")
    print(f"     Noise overhead: {getattr(result, 'noise_overhead', 0.0):.4f}")


def demonstrate_circuit_complexity_effects():
    """Demonstrate how noise affects circuits of different complexity."""
    print("\n📈 Circuit Complexity vs Noise Effects")
    print("=" * 40)
    
    # Test circuits of increasing complexity
    circuits = [
        ("Single qubit", create_randomized_benchmarking_circuit(1)),
        ("RB depth 3", create_randomized_benchmarking_circuit(3)),
        ("RB depth 5", create_randomized_benchmarking_circuit(5)),
        ("Bell state", create_bell_state_circuit()),
        ("GHZ-3", create_ghz_state_circuit(3))
    ]
    
    simulator = create_device_simulator("ibm_like")
    
    for name, circuit in circuits:
        print(f"\n🔬 Testing {name}:")
        
        # Run with noise
        result = simulator.run(circuit, shots=500, compare_ideal=True)
        
        # Calculate fidelity
        fidelity = simulator._calculate_fidelity(
            result.ideal_counts, result.counts
        ) if result.ideal_counts else 1.0
        
        print(f"   Qubits: {circuit.num_qubits}, Depth: {circuit.depth}")
        print(f"   Fidelity: {fidelity:.4f}")
        print(f"   Noise overhead: {getattr(result, 'noise_overhead', 0.0):.4f}")


def demonstrate_backend_integration():
    """Demonstrate integration with the provider system."""
    print("\n🔌 Backend Integration Demonstration")
    print("=" * 40)
    
    # Create noisy simulator backends
    backends = {
        "IBM-like": NoisySimulatorBackend("ibm_sim", "ibm_like", max_qubits=5),
        "IonQ-like": NoisySimulatorBackend("ionq_sim", "ionq_like", max_qubits=11),
        "Google-like": NoisySimulatorBackend("google_sim", "google_like", max_qubits=20)
    }
    
    circuit = create_bell_state_circuit()
    
    for name, backend in backends.items():
        print(f"\n🖥️ Testing {name} Backend:")
        
        # Initialize backend
        backend.initialize()
        
        # Get device info
        device_info = backend.get_device_info()
        print(f"   Device type: {device_info.device_type.value}")
        print(f"   Qubits: {device_info.num_qubits}")
        print(f"   Noise enabled: {device_info.metadata.get('noise_enabled', False)}")
        
        # Submit job
        job_handle = backend.submit_circuit(circuit, shots=100, compare_ideal=True)
        
        # Wait for completion
        while backend.get_job_status(job_handle) not in [JobStatus.COMPLETED, JobStatus.FAILED]:
            time.sleep(0.1)
        
        # Retrieve results
        result = backend.retrieve_results(job_handle)
        
        if result.status == JobStatus.COMPLETED:
            print(f"   Execution time: {result.execution_time:.1f}ms")
            print(f"   Counts: {result.counts}")
            
            if 'noise_overhead' in result.metadata:
                print(f"   Noise overhead: {result.metadata['noise_overhead']:.4f}")
        else:
            print(f"   Job failed: {result.error_message}")


def create_noise_comparison_plot(results: Dict):
    """Create visualization comparing noise effects."""
    try:
        import matplotlib.pyplot as plt
        
        print("\n📊 Creating noise comparison plot...")
        
        # Extract data for plotting
        models = list(results.keys())
        overheads = [results[model]['noise_overhead'] for model in models]
        times = [results[model]['execution_time'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Noise overhead comparison
        ax1.bar(models, overheads, color=['green', 'blue', 'orange', 'red'])
        ax1.set_ylabel('Noise Overhead')
        ax1.set_title('Noise Effect by Device Type')
        ax1.tick_params(axis='x', rotation=45)
        
        # Execution time comparison
        ax2.bar(models, times, color=['green', 'blue', 'orange', 'red'])
        ax2.set_ylabel('Execution Time (s)')
        ax2.set_title('Simulation Performance')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('noise_comparison.png', dpi=150, bbox_inches='tight')
        print("   Plot saved as 'noise_comparison.png'")
        
    except ImportError:
        print("   Matplotlib not available for plotting")


def main():
    """Main demonstration function."""
    print("🚀 Starting Quantum Noise Model Demonstration")
    print("=" * 60)
    
    try:
        # Core noise model demonstration
        results = demonstrate_noise_models()
        
        # Noise characterization
        demonstrate_noise_characterization()
        
        # Circuit complexity effects
        demonstrate_circuit_complexity_effects()
        
        # Backend integration
        demonstrate_backend_integration()
        
        # Create visualization
        create_noise_comparison_plot(results)
        
        print("\n✅ Demonstration completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Multiple device noise models (IBM, IonQ, Google)")
        print("  • Custom noise model creation from calibration data")
        print("  • Comparative ideal vs noisy simulation")
        print("  • Circuit complexity impact analysis")
        print("  • Backend integration with provider system")
        print("  • Performance metrics and noise characterization")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 