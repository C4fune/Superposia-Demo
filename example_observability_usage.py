#!/usr/bin/env python3
"""
Example Usage of Observability Features

This example demonstrates how to use the unified logging system, performance monitoring,
and debugging capabilities of the quantum computing platform.
"""

import time
from quantum_platform.observability.logging import (
    setup_logging, get_logger, configure_logging, LogLevel, LogFormat
)
from quantum_platform.observability.monitor import get_monitor
from quantum_platform.observability.integration import initialize_observability
from quantum_platform.compiler.language.dsl import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, CNOT
from quantum_platform.simulation.statevector import StateVectorSimulator

def example_basic_logging():
    """Demonstrate basic logging capabilities."""
    print("\n=== Basic Logging Example ===")
    
    # Setup logging
    config = configure_logging(
        level=LogLevel.INFO,
        log_to_console=True,
        log_to_file=True,
        log_file_path="logs/example_usage.log"
    )
    setup_logging(config)
    
    # Get component-specific loggers
    compiler_logger = get_logger("Compiler")
    simulator_logger = get_logger("Simulator")
    
    # Log various types of messages
    compiler_logger.info("Starting quantum circuit compilation")
    compiler_logger.debug("Gate optimization pass started")
    compiler_logger.warning("Gate count is high, consider optimization")
    
    simulator_logger.info("Initializing quantum simulator")
    simulator_logger.error("Insufficient memory for large circuit simulation")
    
    print("✓ Logged messages to console and file")

def example_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n=== Performance Monitoring Example ===")
    
    monitor = get_monitor()
    logger = get_logger("PerformanceExample")
    
    # Monitor a simple operation
    with monitor.measure_operation("matrix_multiplication", "MathComponent"):
        logger.info("Starting heavy computation")
        
        # Simulate computational work
        result = 0
        for i in range(100000):
            result += i * i
            
        logger.info(f"Computation completed with result: {result}")
    
    # Set custom metrics
    monitor.increment_counter("operations_completed")
    monitor.set_gauge("cpu_temperature", 65.5)
    monitor.set_gauge("memory_usage_percent", 78.2)
    
    # Get system summary
    summary = monitor.get_system_summary()
    print(f"✓ Monitored {summary['performance']['total_operations_5min']} operations")
    print(f"✓ Average operation duration: {summary['performance']['avg_duration_5min']:.3f}s")

def example_quantum_circuit_with_observability():
    """Demonstrate observability integration with quantum circuits."""
    print("\n=== Quantum Circuit with Observability ===")
    
    # Initialize comprehensive observability
    integration = initialize_observability(
        log_level=LogLevel.INFO,
        enable_performance_monitoring=True
    )
    
    logger = get_logger("QuantumExample")
    monitor = get_monitor()
    
    # Create and execute quantum circuit with full observability
    with monitor.measure_operation("quantum_bell_circuit", "QuantumExample"):
        logger.info("Creating Bell state quantum circuit")
        
        with QuantumProgram() as program:
            # Allocate qubits with logging
            logger.debug("Allocating 2 qubits")
            qubits = program.allocate(2)
            
            # Apply gates with logging
            logger.debug("Applying Hadamard gate to qubit 0")
            H(qubits[0])
            
            logger.debug("Applying CNOT gate between qubits 0 and 1")
            CNOT(qubits[0], qubits[1])
            
            # Compile circuit
            circuit = program.compile()
            logger.info(f"Compiled circuit with {len(circuit.operations)} operations")
    
    # Simulate with monitoring
    with monitor.measure_operation("quantum_simulation", "QuantumExample"):
        logger.info("Starting quantum simulation")
        
        simulator = StateVectorSimulator()
        result = simulator.run(circuit, shots=1000)
        
        logger.info("Simulation completed successfully")
        logger.info(f"Measurement results: {dict(result.measurement_counts)}")
    
    print("✓ Quantum circuit executed with full observability")

def example_error_handling_with_logging():
    """Demonstrate error handling and logging."""
    print("\n=== Error Handling with Logging ===")
    
    logger = get_logger("ErrorExample")
    monitor = get_monitor()
    
    # Demonstrate successful operation logging
    with monitor.measure_operation("successful_operation", "ErrorExample"):
        logger.info("Starting operation that will succeed")
        time.sleep(0.1)
        logger.info("Operation completed successfully")
    
    # Demonstrate error logging
    try:
        with monitor.measure_operation("failing_operation", "ErrorExample"):
            logger.info("Starting operation that will fail")
            logger.warning("Potential issue detected")
            raise ValueError("This is an intentional error for demonstration")
    except ValueError as e:
        logger.error(f"Operation failed as expected: {e}")
        print("✓ Error properly logged and handled")
    
    # System continues to work after error
    logger.info("System recovered and continues to function")

def example_concurrent_operations():
    """Demonstrate logging in concurrent operations."""
    print("\n=== Concurrent Operations Example ===")
    
    import threading
    
    def worker_task(worker_id: int):
        logger = get_logger(f"Worker{worker_id}")
        monitor = get_monitor()
        
        for i in range(3):
            with monitor.measure_operation(f"task_{i}", f"Worker{worker_id}"):
                logger.info(f"Worker {worker_id} executing task {i}")
                time.sleep(0.05)  # Simulate work
                
                if i == 1:  # Log a warning on second task
                    logger.warning(f"Worker {worker_id} encountered minor issue in task {i}")
                
                logger.debug(f"Worker {worker_id} completed task {i}")
    
    # Start multiple workers
    threads = []
    for worker_id in range(3):
        thread = threading.Thread(target=worker_task, args=(worker_id,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    print("✓ All concurrent operations completed with proper logging")

def example_log_analysis():
    """Demonstrate log analysis capabilities."""
    print("\n=== Log Analysis Example ===")
    
    monitor = get_monitor()
    
    # Generate some varied operations for analysis
    logger = get_logger("AnalysisExample")
    
    operations = [
        ("fast_operation", 0.01),
        ("medium_operation", 0.05),
        ("slow_operation", 0.15),
        ("fast_operation", 0.02),
        ("medium_operation", 0.06)
    ]
    
    for op_name, duration in operations:
        with monitor.measure_operation(op_name, "AnalysisExample"):
            logger.info(f"Executing {op_name}")
            time.sleep(duration)
    
    # Get and display analysis
    component_stats = monitor.get_component_stats("AnalysisExample")
    if component_stats:
        print(f"✓ Analysis complete:")
        print(f"  - Total operations: {component_stats['total_operations']}")
        print(f"  - Average duration: {component_stats['avg_duration']:.3f}s")
        print(f"  - Success rate: {component_stats['success_count']}/{component_stats['total_operations']}")
        print(f"  - Last operation: {component_stats['last_operation']}")
    
    # Export metrics for further analysis
    metrics_data = monitor.export_metrics("dict")
    print(f"✓ Exported {len(metrics_data['performance_metrics'])} performance metrics for analysis")

def main():
    """Run all observability examples."""
    print("Quantum Platform Observability Examples")
    print("=" * 50)
    
    try:
        example_basic_logging()
        example_performance_monitoring()
        example_quantum_circuit_with_observability()
        example_error_handling_with_logging()
        example_concurrent_operations()
        example_log_analysis()
        
        print("\n" + "=" * 50)
        print("🎉 All observability examples completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• Unified logging across all components")
        print("• Performance monitoring and metrics")
        print("• Integration with quantum circuits")
        print("• Error handling and recovery")
        print("• Concurrent operation logging")
        print("• Log analysis and reporting")
        print("\nCheck 'logs/example_usage.log' for detailed log output.")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 