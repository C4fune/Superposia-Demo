#!/usr/bin/env python3
"""
Hardware Execution Example

This example demonstrates the Real Hardware Execution features of the
quantum platform, including Hardware Abstraction Layer (HAL), transpilation,
and job management.
"""

import time
import asyncio
from quantum_platform import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, CNOT, measure
from quantum_platform.hardware import (
    LocalSimulatorBackend, get_backend_registry, get_job_manager,
    transpile_for_device, JobPriority
)
from quantum_platform.errors import handle_errors, get_error_reporter


@handle_errors
def create_sample_circuit():
    """Create a sample quantum circuit for testing."""
    print("📋 Creating Sample Quantum Circuit")
    print("=" * 50)
    
    with QuantumProgram(name="hardware_test_circuit") as qp:
        # Allocate qubits
        qubits = qp.allocate(3)
        
        # Create Bell state + extra qubit
        H(qubits[0])
        CNOT(qubits[0], qubits[1])
        X(qubits[2])
        
        # Measure all qubits
        measure(qubits)
    
    print(f"✅ Created circuit with {qp.circuit.num_qubits} qubits")
    print(f"   Operations: {qp.circuit.num_operations}")
    print(f"   Circuit depth: {qp.circuit.depth}")
    print()
    
    return qp.circuit


@handle_errors
def demonstrate_hal():
    """Demonstrate Hardware Abstraction Layer."""
    print("🔧 Hardware Abstraction Layer Demo")
    print("=" * 50)
    
    # Get backend registry
    registry = get_backend_registry()
    
    # Register local simulator backend
    simulator = LocalSimulatorBackend(name="local_sim", max_qubits=30)
    registry.register_backend(LocalSimulatorBackend, "LocalSimulator")
    
    # Create backend instance
    backend = registry.create_backend("LocalSimulator", "test_simulator")
    
    print(f"✅ Registered backend: {backend.name}")
    print(f"   Provider: {backend.provider}")
    print(f"   Initialized: {backend._initialized}")
    
    # Get device information
    device_info = backend.get_device_info()
    print(f"\n📊 Device Information:")
    print(f"   Name: {device_info.name}")
    print(f"   Type: {device_info.device_type.value}")
    print(f"   Qubits: {device_info.num_qubits}")
    print(f"   Basis gates: {device_info.basis_gates[:5]}...")  # Show first 5
    print(f"   Simulator: {device_info.simulator}")
    print(f"   Operational: {device_info.operational}")
    print()
    
    return backend


@handle_errors
def demonstrate_transpilation(circuit, backend):
    """Demonstrate circuit transpilation."""
    print("🔄 Circuit Transpilation Demo")
    print("=" * 50)
    
    # Get device info
    device_info = backend.get_device_info()
    
    # Get transpilation preview
    from quantum_platform.hardware.transpilation import CircuitTranspiler
    transpiler = CircuitTranspiler()
    preview = transpiler.get_transpilation_preview(circuit, device_info)
    
    print("📊 Transpilation Preview:")
    print(f"   Original gates: {preview['original_gates']}")
    print(f"   Original depth: {preview['original_depth']}")
    print(f"   Device qubits: {preview['device_qubits']}")
    print(f"   Connectivity limited: {preview['connectivity_limited']}")
    print(f"   Unsupported gates: {preview['unsupported_gates']}")
    print(f"   Estimated overhead: {preview['estimated_overhead']:.2f}x")
    
    # Perform transpilation
    print("\n🔄 Transpiling circuit...")
    result = transpile_for_device(circuit, device_info)
    
    print(f"✅ Transpilation completed:")
    print(f"   Success: {result.success}")
    print(f"   Execution time: {result.execution_time:.3f}s")
    print(f"   Passes applied: {', '.join(result.passes_applied)}")
    print(f"   Gate count: {result.original_gate_count} → {result.transpiled_gate_count}")
    print(f"   Circuit depth: {result.original_depth} → {result.transpiled_depth}")
    print(f"   Overhead ratio: {result.get_overhead_ratio():.2f}x")
    print(f"   SWAP insertions: {result.swap_count}")
    print()
    
    return result.transpiled_circuit


@handle_errors
def demonstrate_job_submission(circuit, backend):
    """Demonstrate job submission and management."""
    print("📤 Job Submission & Management Demo")
    print("=" * 50)
    
    # Get job manager
    job_manager = get_job_manager()
    
    # Register backend with job manager
    job_manager.register_backend(backend.name, backend)
    
    # Start job manager
    job_manager.start()
    print("✅ Job manager started")
    
    # Add job callbacks
    def on_job_submit(job):
        print(f"🚀 Job submitted: {job.job_id}")
    
    def on_job_start(job):
        print(f"▶️  Job started: {job.job_id}")
    
    def on_job_complete(job):
        print(f"✅ Job completed: {job.job_id}")
        if job.result:
            print(f"   Execution time: {job.result.execution_time:.2f}ms")
            print(f"   Top results: {dict(list(job.result.counts.items())[:3])}")
    
    def on_job_error(job):
        print(f"❌ Job failed: {job.job_id}")
        print(f"   Error: {job.error_message}")
    
    job_manager.add_callback('on_submit', on_job_submit)
    job_manager.add_callback('on_start', on_job_start)
    job_manager.add_callback('on_complete', on_job_complete)
    job_manager.add_callback('on_error', on_job_error)
    
    # Submit multiple jobs with different priorities
    jobs = []
    
    # High priority job
    job_id_1 = job_manager.submit_job(
        circuit, backend.name, shots=1000, 
        priority=JobPriority.HIGH,
        user_id="user_123",
        tags=["demo", "high_priority"]
    )
    jobs.append(job_id_1)
    
    # Normal priority job
    job_id_2 = job_manager.submit_job(
        circuit, backend.name, shots=5000,
        priority=JobPriority.NORMAL,
        user_id="user_123",
        tags=["demo", "normal_priority"]
    )
    jobs.append(job_id_2)
    
    # Low priority job
    job_id_3 = job_manager.submit_job(
        circuit, backend.name, shots=2000,
        priority=JobPriority.LOW,
        user_id="user_456",
        tags=["demo", "low_priority"]
    )
    jobs.append(job_id_3)
    
    print(f"📤 Submitted {len(jobs)} jobs")
    
    # Monitor job progress
    print("\n⏳ Monitoring job progress...")
    completed_jobs = 0
    timeout = 30  # seconds
    start_time = time.time()
    
    while completed_jobs < len(jobs) and (time.time() - start_time) < timeout:
        for job_id in jobs:
            status = job_manager.get_job_status(job_id)
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                completed_jobs += 1
        
        time.sleep(1)
    
    # Show final results
    print(f"\n📊 Final Results:")
    for job_id in jobs:
        result = job_manager.get_job_result(job_id)
        if result:
            print(f"   {job_id}: {result.status.value}")
            if result.counts:
                top_count = max(result.counts.values())
                top_state = [k for k, v in result.counts.items() if v == top_count][0]
                print(f"     Most probable: |{top_state}⟩ ({top_count}/{result.shots})")
    
    # Show queue statistics
    stats = job_manager.get_queue_stats()
    print(f"\n📈 Queue Statistics:")
    print(f"   Active jobs: {stats['active_jobs']}")
    print(f"   Pending jobs: {stats['pending_jobs']}")
    print(f"   Completed jobs: {stats['completed_jobs']}")
    print(f"   Total processed: {stats['total_jobs']}")
    
    # Stop job manager
    job_manager.stop()
    print("🛑 Job manager stopped")
    print()
    
    return jobs


@handle_errors
def demonstrate_direct_execution(circuit, backend):
    """Demonstrate direct circuit execution."""
    print("⚡ Direct Circuit Execution Demo")
    print("=" * 50)
    
    # Validate circuit
    try:
        backend.validate_circuit(circuit)
        print("✅ Circuit validation passed")
    except Exception as e:
        print(f"❌ Circuit validation failed: {e}")
        return
    
    # Submit and wait for result
    print("🚀 Submitting circuit for direct execution...")
    result = backend.submit_and_wait(circuit, shots=1000, timeout=30)
    
    print(f"✅ Execution completed:")
    print(f"   Status: {result.status.value}")
    print(f"   Shots: {result.shots}")
    print(f"   Execution time: {result.execution_time:.2f}ms")
    print(f"   Queue time: {result.queue_time:.2f}s")
    
    if result.counts:
        print(f"   Results:")
        sorted_counts = sorted(result.counts.items(), 
                             key=lambda x: x[1], reverse=True)
        for state, count in sorted_counts[:5]:  # Show top 5
            probability = count / result.shots * 100
            print(f"     |{state}⟩: {count:4d} ({probability:5.1f}%)")
    
    print()
    return result


@handle_errors
def demonstrate_error_handling():
    """Demonstrate error handling in hardware execution."""
    print("🚨 Error Handling Demo")
    print("=" * 50)
    
    # Get error reporter
    error_reporter = get_error_reporter()
    
    try:
        # Try to create invalid circuit
        with QuantumProgram(name="invalid_circuit") as qp:
            qubits = qp.allocate(50)  # Too many qubits
            H(qubits[0])
            measure(qubits)
        
        # Try to execute on limited backend
        backend = LocalSimulatorBackend(max_qubits=10)  # Limited qubits
        backend.initialize()
        
        # This should fail
        backend.validate_circuit(qp.circuit)
        
    except Exception as e:
        print(f"✅ Expected error caught: {type(e).__name__}")
        print(f"   Message: {e}")
        
        # Generate error report
        report = error_reporter.collect_error(e, user_action="hardware_demo")
        print(f"   Error report generated: {report.report_id}")
    
    # Try invalid backend
    try:
        registry = get_backend_registry()
        backend = registry.get_backend("nonexistent_backend")
        if not backend:
            raise HardwareError("Backend not found")
            
    except Exception as e:
        print(f"✅ Backend error handled: {type(e).__name__}")
    
    print()


def main():
    """Main demonstration function."""
    print("🚀 Quantum Hardware Execution Features Demo")
    print("=" * 60)
    print("This demo showcases the Real Hardware Execution capabilities")
    print("of the quantum platform with comprehensive error handling.")
    print("=" * 60)
    print()
    
    try:
        # Create sample circuit
        circuit = create_sample_circuit()
        
        # Demonstrate HAL
        backend = demonstrate_hal()
        
        # Demonstrate transpilation
        transpiled_circuit = demonstrate_transpilation(circuit, backend)
        
        # Demonstrate job management
        jobs = demonstrate_job_submission(transpiled_circuit, backend)
        
        # Demonstrate direct execution
        result = demonstrate_direct_execution(circuit, backend)
        
        # Demonstrate error handling
        demonstrate_error_handling()
        
        print("🎉 Hardware Execution Demo Completed Successfully!")
        print("=" * 60)
        print("Key features demonstrated:")
        print("✅ Hardware Abstraction Layer (HAL)")
        print("✅ Circuit Transpilation")
        print("✅ Job Queue Management")
        print("✅ Direct Circuit Execution")
        print("✅ Comprehensive Error Handling")
        print("✅ Backend Registration & Discovery")
        print("✅ Device Information & Capabilities")
        print("✅ Real-time Job Monitoring")
        print()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 