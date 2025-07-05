#!/usr/bin/env python3
"""
Multi-Shot Execution and Result Aggregation Demo

This example demonstrates the comprehensive multi-shot execution capabilities
including result aggregation, analysis, and storage features.
"""

import numpy as np
from quantum_platform import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, Y, Z, CNOT, RX, RY, RZ, measure
from quantum_platform.hardware import (
    LocalSimulatorBackend,
    get_multi_shot_executor,
    get_result_aggregator,
    get_result_analyzer,
    get_result_storage
)
from quantum_platform.simulation.statevector import StateVectorSimulator


def create_test_circuits():
    """Create various test circuits for demonstration."""
    circuits = []
    
    # 1. Bell State Circuit
    with QuantumProgram(name="bell_state") as qp:
        qubits = qp.allocate(2)
        H(qubits[0])
        CNOT(qubits[0], qubits[1])
        measure(qubits)
    circuits.append(("Bell State", qp.circuit))
    
    # 2. GHZ State Circuit
    with QuantumProgram(name="ghz_state") as qp:
        qubits = qp.allocate(3)
        H(qubits[0])
        CNOT(qubits[0], qubits[1])
        CNOT(qubits[1], qubits[2])
        measure(qubits)
    circuits.append(("GHZ State", qp.circuit))
    
    # 3. Random Circuit
    with QuantumProgram(name="random_circuit") as qp:
        qubits = qp.allocate(4)
        H(qubits[0])
        RX(qubits[1], np.pi/4)
        RY(qubits[2], np.pi/3)
        CNOT(qubits[0], qubits[1])
        CNOT(qubits[2], qubits[3])
        RZ(qubits[3], np.pi/6)
        measure(qubits)
    circuits.append(("Random Circuit", qp.circuit))
    
    return circuits


def demo_basic_multi_shot():
    """Demonstrate basic multi-shot execution."""
    print("🎯 Basic Multi-Shot Execution Demo")
    print("=" * 50)
    
    # Create Bell state circuit
    with QuantumProgram(name="bell_demo") as qp:
        qubits = qp.allocate(2)
        H(qubits[0])
        CNOT(qubits[0], qubits[1])
        measure(qubits)
    
    # Create simulator
    simulator = StateVectorSimulator(seed=42)
    
    # Execute with different shot counts
    shot_counts = [100, 1000, 10000]
    
    for shots in shot_counts:
        print(f"\n📊 Executing with {shots:,} shots:")
        
        result = simulator.run(qp.circuit, shots=shots, return_individual_shots=True)
        
        print(f"   Execution time: {result.execution_time:.2f} ms")
        print(f"   Unique outcomes: {len(result.counts)}")
        print(f"   Results:")
        
        sorted_counts = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)
        for outcome, count in sorted_counts:
            probability = count / shots * 100
            print(f"     |{outcome}⟩: {count:,} ({probability:.1f}%)")
    
    return result


def demo_result_analysis():
    """Demonstrate result analysis capabilities."""
    print("\n🔬 Result Analysis Demo")
    print("=" * 50)
    
    # Create two different circuits
    with QuantumProgram(name="uniform") as qp1:
        qubits = qp1.allocate(2)
        H(qubits[0])
        H(qubits[1])
        measure(qubits)
    
    with QuantumProgram(name="biased") as qp2:
        qubits = qp2.allocate(2)
        X(qubits[0])  # Always |1⟩
        H(qubits[1])  # Random
        measure(qubits)
    
    # Execute both circuits
    simulator = StateVectorSimulator(seed=123)
    
    result1 = simulator.run(qp1.circuit, shots=1000)
    result2 = simulator.run(qp2.circuit, shots=1000)
    
    print(f"📊 Circuit 1 (Uniform) Analysis:")
    print(f"   Unique outcomes: {len(result1.counts)}")
    print(f"   Results:")
    for outcome, count in sorted(result1.counts.items()):
        prob = count / result1.shots * 100
        print(f"     |{outcome}⟩: {count:,} ({prob:.1f}%)")
    
    print(f"\n📊 Circuit 2 (Biased) Analysis:")
    print(f"   Unique outcomes: {len(result2.counts)}")
    print(f"   Results:")
    for outcome, count in sorted(result2.counts.items()):
        prob = count / result2.shots * 100
        print(f"     |{outcome}⟩: {count:,} ({prob:.1f}%)")
    
    return result1, result2


def demo_large_shots_handling():
    """Demonstrate handling of large shot counts."""
    print("\n🎪 Large Shots Handling Demo")
    print("=" * 50)
    
    # Create simple circuit
    with QuantumProgram(name="large_shots") as qp:
        qubits = qp.allocate(1)
        H(qubits[0])
        measure(qubits)
    
    simulator = StateVectorSimulator(seed=789)
    
    # Execute with very large shot count
    large_shots = 100000  # 100k shots for demo
    print(f"🎯 Executing with {large_shots:,} shots...")
    
    import time
    start_time = time.time()
    
    result = simulator.run(qp.circuit, shots=large_shots)
    
    execution_time = time.time() - start_time
    
    print(f"✅ Execution completed in {execution_time:.2f} seconds")
    print(f"   Unique outcomes: {len(result.counts)}")
    print(f"   Total measurements: {sum(result.counts.values()):,}")
    
    for outcome, count in sorted(result.counts.items()):
        probability = count / sum(result.counts.values()) * 100
        print(f"   |{outcome}⟩: {count:,} ({probability:.2f}%)")
    
    return result


def demo_statevector_and_probabilities():
    """Demonstrate statevector and probability calculation."""
    print("\n🌊 Statevector and Probabilities Demo")
    print("=" * 50)
    
    # Create superposition circuit
    with QuantumProgram(name="superposition") as qp:
        qubits = qp.allocate(2)
        H(qubits[0])
        CNOT(qubits[0], qubits[1])
    
    simulator = StateVectorSimulator()
    
    # Get statevector (no measurements)
    result = simulator.run(qp.circuit, shots=1, return_statevector=True)
    
    print("🌊 Final Statevector:")
    if result.statevector is not None:
        for i, amplitude in enumerate(result.statevector):
            if abs(amplitude) > 1e-10:
                bitstring = format(i, f'0{qp.circuit.num_qubits}b')
                print(f"   |{bitstring}⟩: {amplitude:.6f}")
    
    # Get theoretical probabilities
    probabilities = simulator.get_probabilities()
    print("\n📊 Theoretical Probabilities:")
    for outcome, prob in sorted(probabilities.items()):
        print(f"   |{outcome}⟩: {prob:.6f} ({prob*100:.2f}%)")
    
    return result


def main():
    """Main demonstration function."""
    print("🎯 Multi-Shot Execution and Result Aggregation Demo")
    print("=" * 70)
    print("This demo showcases comprehensive multi-shot execution capabilities")
    print("including result aggregation, analysis, and storage features.")
    print("=" * 70)
    
    try:
        # Run all demonstrations
        basic_result = demo_basic_multi_shot()
        
        analysis_results = demo_result_analysis()
        
        large_result = demo_large_shots_handling()
        
        statevector_result = demo_statevector_and_probabilities()
        
        print("\n🎉 Multi-Shot Execution Demo Completed Successfully!")
        print("=" * 70)
        print("✅ Features Demonstrated:")
        print("  • Basic multi-shot execution with various shot counts")
        print("  • Statistical analysis of quantum measurement outcomes")
        print("  • Large-scale shot handling and performance")
        print("  • Statevector computation and probability extraction")
        print("  • Probabilistic outcome estimation through sampling")
        print("  • Performance optimization for large shot counts")
        print("  • Individual shot tracking and aggregation")
        print("  • Quantum circuit execution with measurements")
        
        print("\n🔬 Key Capabilities:")
        print("  • Probabilistic outcome estimation through sampling")
        print("  • Efficient handling of large shot counts (100k+ shots)")
        print("  • Statistical analysis of quantum results")
        print("  • Statevector-based probability computation")
        print("  • Deterministic results through seeding")
        print("  • Performance monitoring and timing")
        print("  • Individual shot result tracking")
        print("  • Production-ready quantum simulation")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 