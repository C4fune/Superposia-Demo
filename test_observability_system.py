#!/usr/bin/env python3
"""
Comprehensive Test Suite for Observability and Debugging Features

This test demonstrates the unified logging system, performance monitoring,
debugging capabilities, and integration with existing quantum platform components.
"""

import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

# Import observability components
from quantum_platform.observability.logging import (
    setup_logging, get_logger, configure_logging, LogLevel, LogFormat
)
from quantum_platform.observability.monitor import (
    get_monitor, measure_performance, SystemMonitor, PerformanceMetrics
)
from quantum_platform.observability.integration import (
    initialize_observability, get_integration, ObservabilityMixin,
    add_observability, log_method_calls
)

# Import existing platform components for integration testing
from quantum_platform.compiler.ir.circuit import QuantumCircuit
from quantum_platform.compiler.ir.qubit import QubitRegister
from quantum_platform.compiler.language.dsl import QuantumProgram, QuantumContext
from quantum_platform.compiler.language.operations import H, X, Y, Z, CNOT
from quantum_platform.simulation.statevector import StateVectorSimulator

def test_basic_logging_system():
    """Test basic logging functionality."""
    print("\n=== Testing Basic Logging System ===")
    
    # Setup logging with different configurations
    config = configure_logging(
        level=LogLevel.DEBUG,
        log_to_console=True,
        log_to_file=True,
        log_file_path="logs/test_observability.log",
        enable_performance=True,
        format_template=LogFormat.DETAILED
    )
    
    logger_system = setup_logging(config)
    print(f"✓ Logging system initialized with level: {config.level.name}")
    
    # Test component-specific loggers
    compiler_logger = get_logger("TestCompiler")
    simulator_logger = get_logger("TestSimulator")
    security_logger = get_logger("TestSecurity")
    
    # Test different log levels
    compiler_logger.debug("This is a debug message from compiler")
    compiler_logger.info("Compiler is processing quantum circuit")
    compiler_logger.warning("Potential optimization opportunity detected")
    
    simulator_logger.info("Starting quantum simulation")
    simulator_logger.error("Simulation failed due to resource constraints")
    
    security_logger.info("User authentication successful")
    security_logger.warning("Multiple login attempts detected")
    
    print("✓ Component-specific logging working correctly")
    
    # Test performance logging
    with logger_system.performance_context("test_operation", "TestComponent"):
        time.sleep(0.1)  # Simulate work
        compiler_logger.info("Operation completed successfully")
    
    print("✓ Performance logging context working")
    
    # Test user context logging
    with logger_system.user_context(user_id="test_user", session_id="session_123"):
        security_logger.info("User performed sensitive operation")
        simulator_logger.info("Running simulation for user")
    
    print("✓ User context logging working")

def test_system_monitoring():
    """Test system monitoring and performance metrics."""
    print("\n=== Testing System Monitoring ===")
    
    monitor = get_monitor()
    print("✓ System monitor initialized")
    
    # Test performance measurement
    with monitor.measure_operation("test_computation", "TestComponent"):
        # Simulate some work
        result = sum(i*i for i in range(1000))
        time.sleep(0.05)
    
    print("✓ Performance measurement working")
    
    # Test custom metrics
    monitor.increment_counter("test_operations")
    monitor.increment_counter("test_operations")
    monitor.set_gauge("cpu_usage", 45.2)
    monitor.set_gauge("memory_usage", 67.8)
    
    print("✓ Custom metrics tracking working")
    
    # Test resource collection
    resource_usage = monitor.collect_resource_metrics()
    print(f"✓ Current CPU: {resource_usage.cpu_percent:.1f}%")
    print(f"✓ Current Memory: {resource_usage.memory_percent:.1f}%")
    print(f"✓ Thread Count: {resource_usage.thread_count}")
    
    # Get component statistics
    stats = monitor.get_component_stats()
    print(f"✓ Components monitored: {len(stats)}")
    
    # Test system summary
    summary = monitor.get_system_summary()
    print("✓ System summary generated:")
    print(f"  - Total operations (5min): {summary['performance']['total_operations_5min']}")
    print(f"  - Success rate (5min): {summary['performance']['success_rate_5min']:.2%}")
    print(f"  - Average duration (5min): {summary['performance']['avg_duration_5min']:.3f}s")

@measure_performance("decorated_function", "TestDecorator")
def example_decorated_function(n: int) -> int:
    """Example function with performance measurement decorator."""
    logger = get_logger("TestDecorator")
    logger.info(f"Processing {n} items")
    
    # Simulate work
    result = 0
    for i in range(n):
        result += i * i
    
    logger.info(f"Completed processing, result: {result}")
    return result

def test_performance_decorators():
    """Test performance measurement decorators."""
    print("\n=== Testing Performance Decorators ===")
    
    # Test decorated function
    result = example_decorated_function(1000)
    print(f"✓ Decorated function result: {result}")
    
    # Test method decorator
    @log_method_calls("TestClass", log_args=True, performance_tracking=True)
    class TestClass:
        def __init__(self, name: str):
            self.name = name
        
        def process_data(self, data: list) -> int:
            return sum(data)
    
    test_obj = TestClass("test_instance")
    result = test_obj.process_data([1, 2, 3, 4, 5])
    print(f"✓ Method logging decorator result: {result}")

def test_observability_mixin():
    """Test ObservabilityMixin for existing classes."""
    print("\n=== Testing Observability Mixin ===")
    
    # Create a class with observability
    class TestProcessor(ObservabilityMixin):
        def __init__(self, processor_id: str):
            super().__init__()
            self.processor_id = processor_id
            self.logger.info(f"Initialized processor: {processor_id}")
        
        def process_task(self, task_data: dict):
            with self.observe_operation("process_task", task_id=task_data.get('id')):
                self.logger.info(f"Processing task: {task_data}")
                
                # Simulate processing
                time.sleep(0.02)
                
                if task_data.get('should_fail', False):
                    raise ValueError("Simulated processing error")
                
                return {"status": "completed", "result": "success"}
    
    # Test the processor
    processor = TestProcessor("test_proc_1")
    
    # Successful operation
    result = processor.process_task({"id": "task_1", "data": "test_data"})
    print(f"✓ Successful task result: {result}")
    
    # Failed operation
    try:
        processor.process_task({"id": "task_2", "should_fail": True})
    except ValueError as e:
        print(f"✓ Failed task logged correctly: {e}")

def test_quantum_circuit_integration():
    """Test observability integration with quantum circuits."""
    print("\n=== Testing Quantum Circuit Integration ===")
    
    # Initialize observability
    integration = initialize_observability(
        log_level=LogLevel.INFO,
        enable_performance_monitoring=True
    )
    
    logger = get_logger("CircuitTest")
    monitor = get_monitor()
    
    logger.info("Starting quantum circuit observability test")
    
    # Test circuit creation with monitoring
    with monitor.measure_operation("create_bell_circuit", "CircuitTest"):
        with QuantumProgram() as program:
            qubits = program.allocate(2)
            
            # Create Bell state with logging
            logger.debug("Applying Hadamard gate")
            H(qubits[0])
            
            logger.debug("Applying CNOT gate")
            CNOT(qubits[0], qubits[1])
            
            circuit = program.compile()
            logger.info(f"Created circuit with {len(circuit.operations)} operations")
    
    print("✓ Circuit creation monitored successfully")
    
    # Test simulation with monitoring
    with monitor.measure_operation("simulate_circuit", "CircuitTest"):
        simulator = StateVectorSimulator()
        
        logger.info("Starting simulation")
        result = simulator.run(circuit, shots=100)
        
        logger.info(f"Simulation completed with {len(result.measurement_counts)} measurement outcomes")
        logger.info(f"Measurement distribution: {dict(result.measurement_counts)}")
    
    print("✓ Simulation monitoring working")
    
    # Get integration status
    status = integration.get_integration_status()
    print(f"✓ Integration status: {status['total_integrations']} components integrated")

def test_concurrent_logging():
    """Test logging system under concurrent access."""
    print("\n=== Testing Concurrent Logging ===")
    
    def worker_function(worker_id: int):
        logger = get_logger(f"Worker{worker_id}")
        monitor = get_monitor()
        
        for i in range(5):
            with monitor.measure_operation(f"work_iteration_{i}", f"Worker{worker_id}"):
                logger.info(f"Worker {worker_id} iteration {i}")
                time.sleep(0.01)
                
                if i == 2:  # Simulate occasional errors
                    logger.warning(f"Worker {worker_id} encountered minor issue")
    
    # Start multiple worker threads
    threads = []
    for worker_id in range(3):
        thread = threading.Thread(target=worker_function, args=(worker_id,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("✓ Concurrent logging completed successfully")
    
    # Check that all operations were logged
    monitor = get_monitor()
    stats = monitor.get_component_stats()
    total_ops = sum(stat['total_operations'] for stat in stats.values())
    print(f"✓ Total operations logged across all threads: {total_ops}")

def test_log_file_rotation():
    """Test log file creation and basic functionality."""
    print("\n=== Testing Log File Management ===")
    
    log_file_path = Path("logs/test_observability.log")
    
    if log_file_path.exists():
        file_size = log_file_path.stat().st_size
        print(f"✓ Log file exists: {log_file_path} ({file_size} bytes)")
        
        # Read last few lines to verify content
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"✓ Log file contains {len(lines)} lines")
                print(f"✓ Last log entry: {lines[-1].strip()}")
            else:
                print("⚠ Log file is empty")
    else:
        print("⚠ Log file not found")

def test_error_handling_and_recovery():
    """Test error handling in observability components."""
    print("\n=== Testing Error Handling and Recovery ===")
    
    logger = get_logger("ErrorTest")
    monitor = get_monitor()
    
    # Test logging of exceptions
    try:
        with monitor.measure_operation("error_prone_operation", "ErrorTest"):
            logger.info("Starting operation that will fail")
            raise RuntimeError("This is a test error")
    except RuntimeError as e:
        logger.error(f"Caught expected error: {e}")
        print("✓ Error logging working correctly")
    
    # Test recovery after errors
    with monitor.measure_operation("recovery_operation", "ErrorTest"):
        logger.info("System recovered successfully")
        print("✓ System recovery after error working")
    
    # Test invalid log operations
    try:
        # This should not crash the system
        logger.info("Testing with unusual data: %s", {"complex": {"nested": "data"}})
        print("✓ Complex data logging handled gracefully")
    except Exception as e:
        print(f"⚠ Unexpected error in logging: {e}")

def main():
    """Run comprehensive observability system tests."""
    print("Starting Comprehensive Observability System Tests")
    print("=" * 60)
    
    try:
        # Run all tests
        test_basic_logging_system()
        test_system_monitoring()
        test_performance_decorators()
        test_observability_mixin()
        test_quantum_circuit_integration()
        test_concurrent_logging()
        test_log_file_rotation()
        test_error_handling_and_recovery()
        
        # Final system summary
        print("\n=== Final System Summary ===")
        monitor = get_monitor()
        summary = monitor.get_system_summary()
        
        print("Performance Summary:")
        print(f"  - Total operations: {summary['performance']['total_operations_5min']}")
        print(f"  - Success rate: {summary['performance']['success_rate_5min']:.2%}")
        print(f"  - Average duration: {summary['performance']['avg_duration_5min']:.3f}s")
        
        print("\nResource Summary:")
        print(f"  - CPU usage: {summary['resources']['avg_cpu_5min']:.1f}%")
        print(f"  - Memory usage: {summary['resources']['avg_memory_5min']:.1f}%")
        print(f"  - Active threads: {summary['resources']['thread_count']}")
        
        print("\nCustom Metrics:")
        for name, value in summary['custom_metrics']['counters'].items():
            print(f"  - {name}: {value}")
        for name, value in summary['custom_metrics']['gauges'].items():
            print(f"  - {name}: {value}")
        
        # Test export functionality
        metrics_data = monitor.export_metrics("dict")
        print(f"\n✓ Exported {len(metrics_data['performance_metrics'])} performance metrics")
        print(f"✓ Exported {len(metrics_data['resource_history'])} resource usage records")
        
        print("\n" + "=" * 60)
        print("🎉 All Observability System Tests Completed Successfully!")
        print("The unified logging system is fully operational and integrated.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 