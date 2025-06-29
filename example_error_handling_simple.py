"""
Simple Error Handling Integration Example

This example demonstrates the core error handling features with
the existing quantum platform components.
"""

from quantum_platform.compiler.language import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, CNOT, measure

# Import error handling components
from quantum_platform.errors import (
    # Core exceptions
    QuantumPlatformError, UserError, SystemError,
    CompilationError, QubitError, SimulationError,
    
    # Error utilities
    get_error_reporter, get_alert_manager,
    format_error_message, create_user_friendly_message,
    
    # Helper functions
    create_parse_error, create_qubit_error, create_simulation_error
)

from quantum_platform.errors.decorator import (
    handle_errors, user_friendly_errors, safe_execute
)


def demonstrate_basic_error_handling():
    """Show basic error handling capabilities."""
    print("=== Basic Error Handling Demo ===\n")
    
    # 1. Exception Hierarchy
    print("1. Exception Hierarchy:")
    
    # Create different types of errors
    errors = [
        CompilationError("Failed to parse quantum program"),
        QubitError("Qubit index 5 out of range"),
        SimulationError("Insufficient memory for simulation"),
        SystemError("Internal platform error")
    ]
    
    for error in errors:
        formatted = format_error_message(error)
        print(f"   {formatted.title}: {formatted.message}")
        print(f"   → Suggestions: {formatted.suggestions[0] if formatted.suggestions else 'None'}")
    
    print()

def demonstrate_error_reporting():
    """Show error reporting functionality."""
    print("2. Error Reporting:")
    
    reporter = get_error_reporter()
    
    # Simulate various errors and report them
    try:
        raise create_qubit_error(
            "Attempted to apply gate to unallocated qubit",
            qubit_id=3,
            operation="H gate",
            suggestions=["Allocate more qubits", "Check qubit indices"]
        )
    except Exception as e:
        report = reporter.collect_error(e, user_action="Building Bell state circuit")
        print(f"   Report generated: {report.report_id}")
        print(f"   Error type: {report.error_type}")
        print(f"   User message: {report.user_message}")
        # Check if the original exception had suggestions
        if hasattr(e, 'suggestions') and e.suggestions:
            print(f"   Suggestions: {len(e.suggestions)} provided")
        else:
            print(f"   Suggestions: Available in error context")
    
    print()

def demonstrate_alert_system():
    """Show alert system functionality."""
    print("3. Alert System:")
    
    alert_manager = get_alert_manager()
    
    # Create different types of alerts
    alerts = [
        alert_manager.info_alert("Info", "Circuit compilation started"),
        alert_manager.warning_alert("Warning", "Large circuit detected - may be slow"),
        alert_manager.error_alert("Error", "Failed to allocate quantum resources", 
                                error_code="QP001", suggestions=["Reduce circuit size"])
    ]
    
    print(f"   Created {len(alerts)} alerts:")
    for alert in alerts:
        print(f"   - {alert.alert_type.value.upper()}: {alert.title}")
        if alert.metadata.get('suggestions'):
            print(f"     Suggestions: {alert.metadata['suggestions']}")
    
    print()

def demonstrate_decorators():
    """Show decorator usage."""
    print("4. Decorator Usage:")
    
    @user_friendly_errors(
        component="Demo",
        operation="test_function",
        user_message="Something went wrong in the demo"
    )
    def function_with_user_friendly_errors():
        raise ValueError("Internal error for demonstration")
    
    @safe_execute(default_return="fallback_value")
    def function_with_safe_execution():
        raise RuntimeError("This won't crash the program")
    
    @handle_errors(
        component="Demo",
        show_alert=False,
        report_error=False,
        reraise=False,
        fallback_return="handled_gracefully"
    )
    def function_with_custom_handling():
        raise Exception("Custom handled exception")
    
    # Test the decorated functions
    result1 = function_with_user_friendly_errors()
    result2 = function_with_safe_execution()
    result3 = function_with_custom_handling()
    
    print(f"   User-friendly error result: {result1}")
    print(f"   Safe execution result: {result2}")
    print(f"   Custom handling result: {result3}")
    
    print()

def demonstrate_quantum_circuit_errors():
    """Show error handling with quantum circuits."""
    print("5. Quantum Circuit Error Handling:")
    
    def build_problematic_circuit():
        try:
            with QuantumProgram(name="error_demo") as qp:
                # This will work fine
                qubits = qp.allocate(2)
                H(qubits[0])
                CNOT(qubits[0], qubits[1])
                
                # Simulate an error condition
                # In real usage, this might be trying to use an unallocated qubit
                if True:  # Simulate error condition
                    raise QubitError(
                        message="Attempted to use qubit that wasn't allocated",
                        user_message="Please allocate more qubits before using them",
                        suggestions=["Increase qubit count", "Check circuit structure"]
                    )
            
            return qp.circuit
        except Exception as e:
            # Handle the error manually for demo
            error_reporter = get_error_reporter()
            alert_manager = get_alert_manager()
            
            # Report the error
            report = error_reporter.collect_error(e, user_action="Building demo circuit")
            
            # Create alert
            formatted = format_error_message(e)
            alert_manager.create_alert(
                title=formatted.title,
                message=formatted.message,
                alert_type=formatted.level,
                component="QuantumCircuit"
            )
            
            print(f"   ✓ Handled error: {formatted.message}")
            return None
    
    circuit = build_problematic_circuit()
    print(f"   Circuit result: {circuit}")
    
    print()

def demonstrate_user_friendly_messages():
    """Show user-friendly message creation."""
    print("6. User-Friendly Messages:")
    
    # Create various user-friendly messages
    messages = [
        create_user_friendly_message(
            "Compilation Error",
            "Failed to parse quantum program",
            suggestions=["Check syntax", "Review documentation"],
            context={"line_number": 15, "component": "Parser"}
        ),
        create_user_friendly_message(
            "Resource Limit",
            "Circuit requires too much memory",
            suggestions=["Reduce qubits", "Use classical simulation"],
            context={"memory_required": "32GB", "available": "8GB"}
        )
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"   Message {i}:")
        print(f"   {message}")
        print()

def demonstrate_comprehensive_example():
    """Show a comprehensive error handling example."""
    print("7. Comprehensive Example:")
    
    class SmartQuantumCircuitBuilder:
        """Example class with comprehensive error handling."""
        
        def __init__(self):
            self.alert_manager = get_alert_manager()
            self.error_reporter = get_error_reporter()
        
        @user_friendly_errors(
            component="CircuitBuilder",
            operation="validate_and_build",
            user_message="Failed to build quantum circuit"
        )
        def validate_and_build_circuit(self, num_qubits, circuit_type="bell"):
            """Build a circuit with validation and error handling."""
            
            # Input validation
            if num_qubits <= 0:
                raise ValueError("Number of qubits must be positive")
            
            if num_qubits > 20:
                # Show warning for large circuits
                self.alert_manager.warning_alert(
                    "Large Circuit",
                    f"Circuit with {num_qubits} qubits may be slow",
                    component="CircuitBuilder"
                )
            
            # Build circuit based on type
            if circuit_type == "bell":
                return self._build_bell_circuit(num_qubits)
            elif circuit_type == "ghz":
                return self._build_ghz_circuit(num_qubits)
            else:
                raise CompilationError(
                    f"Unknown circuit type: {circuit_type}",
                    user_message=f"Circuit type '{circuit_type}' is not supported"
                )
        
        def _build_bell_circuit(self, num_qubits):
            """Build Bell state circuit."""
            if num_qubits % 2 != 0:
                raise QubitError(
                    "Bell state requires even number of qubits",
                    user_message="Bell states need pairs of qubits. Please use an even number."
                )
            
            with QuantumProgram(name="bell_state") as qp:
                qubits = qp.allocate(num_qubits)
                
                # Create Bell pairs
                for i in range(0, num_qubits, 2):
                    H(qubits[i])
                    CNOT(qubits[i], qubits[i + 1])
                
                measure(qubits)
            
            return qp.circuit
        
        def _build_ghz_circuit(self, num_qubits):
            """Build GHZ state circuit."""
            with QuantumProgram(name="ghz_state") as qp:
                qubits = qp.allocate(num_qubits)
                
                # Create GHZ state
                H(qubits[0])
                for i in range(1, num_qubits):
                    CNOT(qubits[0], qubits[i])
                
                measure(qubits)
            
            return qp.circuit
    
    # Test the comprehensive example
    builder = SmartQuantumCircuitBuilder()
    
    # Test successful operations
    try:
        circuit1 = builder.validate_and_build_circuit(4, "bell")
        print(f"   ✓ Built Bell circuit with {circuit1.num_qubits} qubits")
        
        circuit2 = builder.validate_and_build_circuit(3, "ghz")
        print(f"   ✓ Built GHZ circuit with {circuit2.num_qubits} qubits")
        
    except Exception as e:
        formatted = format_error_message(e)
        print(f"   ✗ Error: {formatted.message}")
    
    # Test error conditions
    try:
        builder.validate_and_build_circuit(3, "bell")  # Odd number for Bell state
    except Exception as e:
        formatted = format_error_message(e)
        print(f"   ✓ Caught expected error: {formatted.message}")
    
    try:
        builder.validate_and_build_circuit(2, "unknown")  # Unknown circuit type
    except Exception as e:
        formatted = format_error_message(e)
        print(f"   ✓ Caught expected error: {formatted.message}")
    
    print()

def main():
    """Run all demonstrations."""
    print("🚀 Quantum Platform Error Handling System Demo\n")
    
    # Run all demonstrations
    demonstrate_basic_error_handling()
    demonstrate_error_reporting()
    demonstrate_alert_system()
    demonstrate_decorators()
    demonstrate_quantum_circuit_errors()
    demonstrate_user_friendly_messages()
    demonstrate_comprehensive_example()
    
    # Show final statistics
    print("8. Final Statistics:")
    
    reporter = get_error_reporter()
    alert_manager = get_alert_manager()
    
    error_history = reporter.get_error_history()
    active_alerts = alert_manager.get_active_alerts()
    
    print(f"   Total errors reported: {len(error_history)}")
    print(f"   Active alerts: {len(active_alerts)}")
    print(f"   Error reports saved to: error_reports/")
    
    # Show last few errors
    if error_history:
        print(f"\n   Recent errors:")
        for error in error_history[-3:]:
            print(f"   - {error.error_type}: {error.user_message}")
    
    # Cleanup
    alert_manager.shutdown()
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📁 Check error_reports/ directory for generated reports")
    print(f"📊 Check logs/quantum_platform.log for detailed logs")


if __name__ == "__main__":
    main() 