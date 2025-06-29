"""
Error Handling Integration Example

This example demonstrates how to integrate the comprehensive error handling
system with existing quantum platform components for better user experience.
"""

from quantum_platform import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, CNOT, measure
from quantum_platform.simulation import get_default_simulator

# Import error handling components
from quantum_platform.errors import (
    # Error reporting
    get_error_reporter, get_alert_manager,
    
    # Exceptions
    SimulationError, QubitError, CompilationError,
    create_simulation_error,
    
    # Decorators
    format_error_message
)

from quantum_platform.errors.decorator import (
    handle_errors, user_friendly_errors, safe_execute, simulation_errors
)


class EnhancedQuantumSimulator:
    """
    Enhanced quantum simulator with comprehensive error handling.
    
    This example shows how to wrap existing functionality with
    error handling for better user experience.
    """
    
    def __init__(self):
        self.simulator = get_default_simulator()
        self.error_reporter = get_error_reporter()
        self.alert_manager = get_alert_manager()
    
    @simulation_errors(operation="circuit_validation")
    def validate_circuit(self, circuit):
        """Validate circuit before simulation with error handling."""
        
        # Check for empty circuit
        if circuit.num_operations == 0:
            raise SimulationError(
                message="Circuit is empty",
                user_message="Cannot simulate an empty circuit. Please add some quantum operations."
            )
        
        # Check qubit count
        if circuit.num_qubits > 25:
            raise create_simulation_error(
                message=f"Circuit has {circuit.num_qubits} qubits, which exceeds simulation limits",
                num_qubits=circuit.num_qubits,
                memory_required=2**(circuit.num_qubits-20)  # Rough memory estimate
            )
        
        # Check for unallocated qubits
        for operation in circuit.operations:
            for qubit in operation.targets:
                if qubit.id >= circuit.num_qubits:
                    raise QubitError(
                        message=f"Operation targets qubit {qubit.id} but only {circuit.num_qubits} qubits allocated",
                        user_message=f"Qubit {qubit.id} is not allocated. Please allocate more qubits or check your circuit."
                    )
        
        return True
    
    @user_friendly_errors(
        component="Simulator",
        operation="run_simulation",
        user_message="Simulation failed. Please check your circuit and try again."
    )
    def run_with_error_handling(self, circuit, shots=1000):
        """Run simulation with comprehensive error handling."""
        
        # Validate circuit first
        self.validate_circuit(circuit)
        
        # Check shots parameter
        if shots <= 0:
            raise ValueError("Number of shots must be positive")
        
        if shots > 1000000:
            self.alert_manager.warning_alert(
                title="Large Shot Count",
                message=f"Running {shots} shots may take a long time",
                component="Simulator"
            )
        
        # Run simulation
        try:
            result = self.simulator.run(circuit, shots=shots)
            
            # Success notification
            self.alert_manager.success_alert(
                title="Simulation Complete",
                message=f"Successfully simulated {circuit.num_qubits}-qubit circuit with {shots} shots",
                component="Simulator"
            )
            
            return result
            
        except MemoryError:
            raise create_simulation_error(
                message="Insufficient memory for simulation",
                num_qubits=circuit.num_qubits,
                shots=shots,
                memory_required=shots * 2**circuit.num_qubits * 16 / 1e9  # Rough estimate
            )
        
        except Exception as e:
            # Convert unknown errors to simulation errors
            raise SimulationError(
                message=f"Simulation failed: {e}",
                user_message="An unexpected error occurred during simulation."
            )


class EnhancedCircuitBuilder:
    """
    Enhanced circuit builder with error checking.
    
    Demonstrates input validation and user-friendly error messages.
    """
    
    def __init__(self):
        self.alert_manager = get_alert_manager()
    
    @handle_errors(
        component="CircuitBuilder",
        operation="build_circuit",
        show_alert=True,
        report_error=True,
        reraise=True
    )
    def build_bell_state_circuit(self, num_pairs=1):
        """Build a Bell state circuit with error checking."""
        
        if num_pairs <= 0:
            raise ValueError("Number of Bell pairs must be positive")
        
        if num_pairs > 10:
            self.alert_manager.warning_alert(
                title="Large Circuit",
                message=f"Creating {num_pairs} Bell pairs ({num_pairs*2} qubits) may be slow to simulate",
                component="CircuitBuilder"
            )
        
        with QuantumProgram(name=f"bell_state_{num_pairs}_pairs") as qp:
            qubits = qp.allocate(num_pairs * 2)
            
            # Create Bell pairs
            for i in range(num_pairs):
                H(qubits[i*2])
                CNOT(qubits[i*2], qubits[i*2 + 1])
            
            # Measure all qubits
            measure(qubits)
        
        return qp.circuit
    
    @safe_execute(default_return=None, component="CircuitBuilder")
    def save_circuit_safe(self, circuit, filename):
        """Save circuit with safe error handling."""
        try:
            # This would normally save to file
            # For demo, just simulate potential failure
            if "invalid" in filename:
                raise IOError("Invalid filename")
            
            print(f"Circuit saved to {filename}")
            return True
            
        except Exception as e:
            # This will be caught by @safe_execute and return None
            # Error will be logged but not propagated
            raise e


def demonstrate_error_handling_integration():
    """Demonstrate the error handling system integration."""
    print("=== Error Handling Integration Demo ===\n")
    
    simulator = EnhancedQuantumSimulator()
    builder = EnhancedCircuitBuilder()
    alert_manager = get_alert_manager()
    
    print("1. Testing Successful Operations:")
    try:
        # Build a valid circuit
        circuit = builder.build_bell_state_circuit(num_pairs=2)
        print(f"   ✓ Built circuit with {circuit.num_qubits} qubits")
        
        # Run simulation
        result = simulator.run_with_error_handling(circuit, shots=100)
        print(f"   ✓ Simulation completed successfully")
        
    except Exception as e:
        formatted = format_error_message(e)
        print(f"   ✗ Error: {formatted.message}")
    
    print("\n2. Testing Error Conditions:")
    
    # Test invalid input
    try:
        circuit = builder.build_bell_state_circuit(num_pairs=-1)
    except Exception as e:
        formatted = format_error_message(e)
        print(f"   ✓ Caught invalid input: {formatted.message}")
    
    # Test large circuit warning
    try:
        circuit = builder.build_bell_state_circuit(num_pairs=15)  # Will trigger warning
        print(f"   ✓ Large circuit created with warning")
    except Exception as e:
        formatted = format_error_message(e)
        print(f"   ✗ Unexpected error: {formatted.message}")
    
    # Test simulation limits
    print("\n3. Testing Simulation Limits:")
    try:
        # Create a circuit that's too large
        with QuantumProgram(name="large_circuit") as qp:
            qubits = qp.allocate(30)  # Too many qubits
            for q in qubits:
                H(q)
        
        result = simulator.run_with_error_handling(qp.circuit)
    except Exception as e:
        formatted = format_error_message(e)
        print(f"   ✓ Caught resource limit: {formatted.title}")
        print(f"     Suggestions: {formatted.suggestions[:2]}")
    
    # Test safe operations
    print("\n4. Testing Safe Operations:")
    success = builder.save_circuit_safe(circuit, "valid_filename.qc")
    print(f"   ✓ Safe save successful: {success}")
    
    failure = builder.save_circuit_safe(circuit, "invalid_filename.qc")
    print(f"   ✓ Safe save with error (returned None): {failure}")
    
    # Show active alerts
    print("\n5. Active Alerts:")
    active_alerts = alert_manager.get_active_alerts()
    for alert in active_alerts[-5:]:  # Show last 5 alerts
        print(f"   {alert.alert_type.value.upper()}: {alert.title}")
    
    # Cleanup
    alert_manager.shutdown()
    
    print("\n=== Integration Demo Complete ===")


def demonstrate_decorator_usage():
    """Demonstrate different decorator patterns."""
    print("\n=== Decorator Usage Examples ===\n")
    
    # Example 1: Simple error handling
    @handle_errors(component="Example", show_alert=False, reraise=False)
    def simple_function_with_errors():
        raise ValueError("This will be handled gracefully")
    
    result = simple_function_with_errors()
    print(f"1. Simple error handling: {result}")
    
    # Example 2: User-friendly errors
    @user_friendly_errors(component="Example", user_message="Something went wrong in the quantum operation")
    def user_facing_function():
        raise RuntimeError("Internal quantum error")
    
    result = user_facing_function()
    print(f"2. User-friendly errors: {result}")
    
    # Example 3: Safe execution
    @safe_execute(default_return="fallback_value")
    def unreliable_function():
        import random
        if random.random() < 0.5:
            raise Exception("Random failure")
        return "success"
    
    for i in range(3):
        result = unreliable_function()
        print(f"3.{i+1} Safe execution attempt: {result}")
    
    print("\n=== Decorator Examples Complete ===")


if __name__ == "__main__":
    # Run integration demo
    demonstrate_error_handling_integration()
    
    # Show decorator usage
    demonstrate_decorator_usage()
    
    print(f"\n✅ Error handling integration examples completed successfully!")
    print(f"📊 Check the error_reports/ directory for generated error reports")
    print(f"📝 Check logs/quantum_platform.log for detailed logging") 