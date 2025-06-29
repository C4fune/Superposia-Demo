#!/usr/bin/env python3
"""
Quantum State Visualization and Circuit Debugging Example

This example demonstrates the comprehensive visualization and debugging 
capabilities of the quantum platform.
"""

import time
import json
from typing import Dict, Any, List

# Core quantum platform imports
from quantum_platform import (
    QuantumCircuit, Qubit, QuantumProgram, StateVectorSimulator,
    StateVisualizer, QuantumDebugger, VisualizationConfig,
    StepMode, DebuggerState, get_logger
)
from quantum_platform.visualization import VisualizationMode

# Configure logging
logger = get_logger(__name__)


def demonstrate_state_visualization():
    """Demonstrate quantum state visualization capabilities."""
    print("=" * 80)
    print("QUANTUM STATE VISUALIZATION DEMONSTRATION")
    print("=" * 80)
    
    # Initialize visualizer with custom configuration
    config = VisualizationConfig(
        max_qubits_for_full_display=8,
        max_basis_states_displayed=16,
        precision_digits=4
    )
    visualizer = StateVisualizer(config)
    simulator = StateVectorSimulator()
    
    # Example 1: Single qubit states
    print("\n1. Single Qubit State Visualization")
    print("-" * 40)
    
    # Create a simple superposition state
    circuit = QuantumCircuit("superposition", 1)
    q0 = circuit.allocate_qubit("q0")
    circuit.add_gate("h", [q0])  # Hadamard gate creates |+⟩ state
    
    # Simulate and visualize
    result = simulator.run(circuit, shots=1, return_statevector=True)
    state_vector = result.final_state
    
    visualization = visualizer.visualize_state(
        state_vector,
        modes=[VisualizationMode.BLOCH_SPHERE, VisualizationMode.PROBABILITY_HISTOGRAM]
    )
    
    print(f"Circuit: {circuit.name}")
    print(f"State dimension: {visualization['state_info']['state_dimension']}")
    
    # Display Bloch sphere information
    bloch_sphere = visualization['bloch_spheres'][0]
    coords = bloch_sphere.coordinates
    print(f"Bloch coordinates: x={coords.x:.3f}, y={coords.y:.3f}, z={coords.z:.3f}")
    print(f"State description: {bloch_sphere.get_classical_state_description()}")
    
    # Display probability distribution
    histogram = visualization['probability_histogram']
    print("Measurement probabilities:")
    for state, prob in histogram.get_sorted_probabilities():
        print(f"  |{state}⟩: {prob:.3f}")


def demonstrate_circuit_debugging():
    """Demonstrate step-by-step quantum circuit debugging."""
    print("\n" + "=" * 80)
    print("QUANTUM CIRCUIT STEP-BY-STEP DEBUGGING DEMONSTRATION")
    print("=" * 80)
    
    # Initialize debugger
    simulator = StateVectorSimulator()
    debugger = QuantumDebugger(simulator)
    
    # Create a circuit for debugging
    print("\n1. Setting Up Debug Circuit")
    print("-" * 40)
    
    debug_circuit = QuantumCircuit("debug_example", 2)
    qubits = debug_circuit.allocate_qubits(2, ["q0", "q1"])
    
    # Add operations step by step
    debug_circuit.add_gate("h", [qubits[0]])                    # Step 0: H on q0
    debug_circuit.add_gate("cnot", [qubits[0]], controls=[qubits[1]])     # Step 1: CNOT
    
    print(f"Debug circuit '{debug_circuit.name}' created with {len(debug_circuit.operations)} operations")
    print("Operations:")
    for i, op in enumerate(debug_circuit.operations):
        print(f"  {i}: {op}")
    
    # Start debugging session
    print("\n2. Starting Debug Session")
    print("-" * 40)
    
    session_id = debugger.start_debug_session(debug_circuit)
    print(f"Debug session started: {session_id}")
    
    # Demonstrate step-by-step execution
    print("\n3. Step-by-Step Execution")
    print("-" * 40)
    
    step_count = 0
    max_steps = 5  # Safety limit
    
    while step_count < max_steps:
        # Get current session state
        session_state = debugger.get_session_state(session_id)
        if not session_state:
            break
        
        print(f"\nStep {step_count + 1}:")
        print(f"  Current operation index: {session_state['current_operation_index']}")
        print(f"  Progress: {session_state['progress_percentage']:.1f}%")
        print(f"  State: {session_state['state']}")
        
        if session_state['current_operation']:
            print(f"  Next operation: {session_state['current_operation']}")
        
        # Check if we're at the end
        if session_state['is_at_end']:
            print("  Circuit execution completed!")
            break
        
        # Execute next step
        step_result = debugger.step_next(session_id)
        
        if step_result.get('success'):
            print(f"  Executed: {step_result['operation_executed']}")
            
        elif step_result.get('completed'):
            print("  Circuit execution completed!")
            break
        elif step_result.get('error'):
            print(f"  Error: {step_result['error']}")
            break
        
        step_count += 1
        time.sleep(0.1)
    
    # Clean up session
    debugger.end_session(session_id)
    print(f"\nDebug session {session_id} ended")


def main():
    """Main demonstration function."""
    try:
        print("🔬 Quantum State Visualization and Circuit Debugging Demo")
        print("⚡ Powered by Next-Generation Quantum Computing Platform")
        print()
        
        # Run demonstrations
        demonstrate_state_visualization()
        demonstrate_circuit_debugging()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nFeatures demonstrated:")
        print("✓ Quantum state visualization with Bloch spheres")
        print("✓ Probability histogram analysis")
        print("✓ Step-by-step circuit debugging")
        print("✓ Real-time state inspection")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
