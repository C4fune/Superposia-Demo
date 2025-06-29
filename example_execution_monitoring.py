#!/usr/bin/env python3
"""
Real-Time Execution Monitoring Dashboard Demo

This script demonstrates the comprehensive real-time execution monitoring
capabilities including job management, progress tracking, status monitoring,
and the dashboard interface for quantum program executions.
"""

import time
import threading
from typing import List, Dict, Any

# Quantum Platform Components
from quantum_platform.compiler.language.dsl import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, Y, Z, CNOT, RX, RY, RZ
from quantum_platform.simulation.executor import (
    MonitoredSimulationExecutor, execute_quantum_circuit, execute_and_wait
)

# Execution Monitoring Components
from quantum_platform.execution.job_manager import get_job_manager, JobType
from quantum_platform.execution.progress_tracker import create_simulation_tracker
from quantum_platform.execution.status_monitor import (
    get_status_monitor, monitor_hardware_job, HardwareJobInfo
)
from quantum_platform.execution.dashboard import (
    get_dashboard, start_dashboard_server
)

# Observability Components
from quantum_platform.observability.logging import get_logger
from quantum_platform.observability.monitor import get_monitor
from quantum_platform.observability.debug import get_debugger

def create_bell_state_circuit():
    """Create a simple Bell state circuit."""
    with QuantumProgram("Bell State Circuit") as qp:
        q0, q1 = qp.allocate(2)
        
        H(q0)
        CNOT(q0, q1)
        
        qp.measure(q0, 0)
        qp.measure(q1, 1)
    
    return qp.circuit

def create_variational_circuit():
    """Create a variational quantum circuit with parameters."""
    with QuantumProgram("Variational Circuit") as qp:
        qubits = qp.allocate(4)
        
        # Layer 1
        for q in qubits:
            RY(q, 0.5)
        
        # Entangling layer
        for i in range(len(qubits) - 1):
            CNOT(qubits[i], qubits[i + 1])
        
        # Layer 2
        for q in qubits:
            RX(q, 0.3)
            RZ(q, 0.7)
        
        # Final entangling
        CNOT(qubits[0], qubits[3])
        CNOT(qubits[1], qubits[2])
        
        # Measurements
        for i, q in enumerate(qubits):
            qp.measure(q, i)
    
    return qp.circuit

def create_large_grover_circuit():
    """Create a larger Grover's algorithm circuit for demonstration."""
    with QuantumProgram("Grover's Algorithm Demo") as qp:
        qubits = qp.allocate(6)  # 6-qubit Grover
        
        # Initialize superposition
        for q in qubits:
            H(q)
        
        # Grover iterations
        for iteration in range(3):
            # Oracle (marking |101010⟩ as an example)
            for i in [0, 2, 4]:  # Apply X to even qubits
                X(qubits[i])
            
            # Multi-controlled Z
            # This is a simplified oracle implementation
            for q in qubits[1:]:
                CNOT(qubits[0], q)
            Z(qubits[-1])
            for q in reversed(qubits[1:]):
                CNOT(qubits[0], q)
            
            for i in [0, 2, 4]:  # Undo X gates
                X(qubits[i])
            
            # Diffusion operator
            for q in qubits:
                H(q)
                X(q)
            
            # Multi-controlled Z for diffusion
            for q in qubits[1:]:
                CNOT(qubits[0], q)
            Z(qubits[-1])
            for q in reversed(qubits[1:]):
                CNOT(qubits[0], q)
            
            for q in qubits:
                X(q)
                H(q)
        
        # Measurements
        for i, q in enumerate(qubits):
            qp.measure(q, i)
    
    return qp.circuit

def demonstrate_progress_tracking():
    """Demonstrate real-time progress tracking for simulations."""
    logger = get_logger("DashboardDemo")
    logger.info("=== Demonstrating Progress Tracking ===")
    
    # Create executor with monitoring enabled
    executor = MonitoredSimulationExecutor(enable_monitoring=True)
    
    # Create circuits of different complexities
    circuits = [
        ("Bell State", create_bell_state_circuit(), 1000),
        ("Variational Circuit", create_variational_circuit(), 2000),
        ("Grover Algorithm", create_large_grover_circuit(), 3000)
    ]
    
    jobs = []
    
    # Submit multiple jobs
    for name, circuit, shots in circuits:
        logger.info(f"Submitting job: {name} ({shots} shots)")
        
        job = executor.execute_circuit(
            circuit=circuit,
            shots=shots,
            job_name=f"Demo: {name}",
            enable_progress_tracking=True
        )
        
        jobs.append((name, job))
        
        # Brief delay between submissions
        time.sleep(0.5)
    
    # Monitor job progress
    logger.info("Monitoring job progress...")
    completed_jobs = []
    
    while len(completed_jobs) < len(jobs):
        for name, job in jobs:
            if job.job_id not in [j.job_id for _, j in completed_jobs]:
                if job.is_finished:
                    status = "✓ COMPLETED" if job.status.value == "completed" else f"✗ {job.status.value.upper()}"
                    logger.info(f"{status}: {name} (Progress: {job.progress:.1f}%)")
                    completed_jobs.append((name, job))
                else:
                    logger.info(f"⏳ RUNNING: {name} (Progress: {job.progress:.1f}%)")
        
        if len(completed_jobs) < len(jobs):
            time.sleep(2)  # Check every 2 seconds
    
    logger.info(f"All {len(jobs)} jobs completed!")
    return [job for _, job in completed_jobs]

def demonstrate_hardware_monitoring():
    """Demonstrate hardware job status monitoring."""
    logger = get_logger("DashboardDemo")
    logger.info("=== Demonstrating Hardware Job Monitoring ===")
    
    status_monitor = get_status_monitor()
    
    # Simulate hardware jobs (in real implementation, these would be actual hardware submissions)
    hardware_jobs = [
        ("IBM Quantum", "ibmq_mumbai", "sim_job_001"),
        ("Google Quantum", "sycamore_23", "sim_job_002"),
        ("IonQ Quantum", "ionq_11", "sim_job_003")
    ]
    
    monitored_jobs = []
    
    for provider, device, provider_job_id in hardware_jobs:
        logger.info(f"Starting to monitor {provider} job {provider_job_id} on {device}")
        
        # Create hardware job info
        job_info = HardwareJobInfo(
            job_id=f"hw_{len(monitored_jobs) + 1}",
            provider_job_id=provider_job_id,
            provider_name=provider,
            device_name=device,
            status="queued",
            queue_position=len(monitored_jobs) + 5  # Simulate queue
        )
        
        # Start monitoring
        job_id = status_monitor.add_hardware_job(job_info)
        monitored_jobs.append(job_id)
        
        time.sleep(0.5)
    
    # Monitor for a while to see status changes
    logger.info("Monitoring hardware jobs for 30 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 30:
        logger.info("--- Hardware Job Status ---")
        
        for job_id in monitored_jobs:
            job_info = status_monitor.get_hardware_job_status(job_id)
            if job_info:
                status_msg = f"{job_info.provider_name} ({job_info.device_name}): {job_info.status}"
                if job_info.queue_position:
                    status_msg += f" [Queue: {job_info.queue_position}]"
                logger.info(status_msg)
        
        time.sleep(5)
    
    # Clean up monitoring
    for job_id in monitored_jobs:
        status_monitor.remove_hardware_job(job_id)
    
    logger.info("Hardware monitoring demonstration completed")

def demonstrate_dashboard_api():
    """Demonstrate dashboard API functionality."""
    logger = get_logger("DashboardDemo")
    logger.info("=== Demonstrating Dashboard API ===")
    
    dashboard = get_dashboard()
    api = dashboard.api
    
    # Get current dashboard state
    state = api.get_dashboard_state()
    logger.info(f"Dashboard State:")
    logger.info(f"  Active Jobs: {len(state['active_jobs'])}")
    logger.info(f"  Hardware Jobs: {len(state['hardware_jobs'])}")
    logger.info(f"  System Status: {state['system_status']}")
    logger.info(f"  Notifications: {len(state['notifications'])}")
    
    # Get statistics
    stats = api.get_statistics()
    logger.info(f"Execution Statistics:")
    logger.info(f"  Total Jobs: {stats['jobs'].get('total_jobs', 0)}")
    logger.info(f"  Completed: {stats['jobs'].get('total_completed', 0)}")
    logger.info(f"  Failed: {stats['jobs'].get('total_failed', 0)}")
    
    # Export dashboard data
    export_data = dashboard.export_dashboard_data("dict")
    logger.info(f"Dashboard export contains {len(export_data.keys())} sections")
    
    return state

def demonstrate_resource_monitoring():
    """Demonstrate system resource monitoring."""
    logger = get_logger("DashboardDemo")
    logger.info("=== Demonstrating Resource Monitoring ===")
    
    monitor = get_monitor()
    
    # Create some load with concurrent simulations
    executor = MonitoredSimulationExecutor()
    circuit = create_variational_circuit()
    
    # Submit multiple concurrent jobs
    jobs = []
    for i in range(3):
        job = executor.execute_circuit(
            circuit=circuit,
            shots=1500,
            job_name=f"Resource Test {i+1}",
            enable_progress_tracking=True
        )
        jobs.append(job)
    
    # Monitor resources while jobs run
    logger.info("Monitoring system resources during execution...")
    for _ in range(10):
        # Get current resource usage
        resource_summary = monitor.get_system_summary()
        logger.info(f"System Load - Memory: {resource_summary.get('memory_usage_mb', 'N/A')}MB, "
                   f"Active Operations: {resource_summary.get('active_operations', 0)}")
        
        time.sleep(2)
    
    # Wait for jobs to complete
    for job in jobs:
        while not job.is_finished:
            time.sleep(0.1)
    
    logger.info("Resource monitoring demonstration completed")

def run_dashboard_server_demo():
    """Run the web dashboard server for interactive demonstration."""
    logger = get_logger("DashboardDemo")
    logger.info("=== Starting Dashboard Web Server ===")
    
    try:
        # Start the dashboard server
        server = start_dashboard_server(port=8080)
        logger.info("Dashboard server started at http://localhost:8080")
        logger.info("You can now open a web browser and view the real-time dashboard")
        
        # Keep the server running for the demo
        logger.info("Server will run for 60 seconds during this demo...")
        
        # Generate some activity for the dashboard
        def generate_activity():
            executor = MonitoredSimulationExecutor()
            circuits = [
                ("Bell State", create_bell_state_circuit(), 800),
                ("Variational", create_variational_circuit(), 1200)
            ]
            
            for name, circuit, shots in circuits:
                job = executor.execute_circuit(
                    circuit=circuit,
                    shots=shots,
                    job_name=f"Dashboard Demo: {name}"
                )
                time.sleep(15)  # Space out job submissions
        
        # Start activity in background
        activity_thread = threading.Thread(target=generate_activity, daemon=True)
        activity_thread.start()
        
        # Keep server running
        time.sleep(60)
        
        # Stop the server
        server.stop()
        logger.info("Dashboard server stopped")
        
    except Exception as e:
        logger.error(f"Failed to start dashboard server: {e}")
        logger.info("Dashboard server demo skipped (this is normal if HTTP modules aren't available)")

def main():
    """Main demonstration function."""
    logger = get_logger("DashboardDemo")
    logger.info("🚀 Starting Real-Time Execution Monitoring Dashboard Demo")
    
    try:
        # 1. Progress Tracking Demo
        completed_jobs = demonstrate_progress_tracking()
        
        # Brief pause
        time.sleep(2)
        
        # 2. Hardware Monitoring Demo
        demonstrate_hardware_monitoring()
        
        # Brief pause
        time.sleep(2)
        
        # 3. Dashboard API Demo
        dashboard_state = demonstrate_dashboard_api()
        
        # Brief pause
        time.sleep(2)
        
        # 4. Resource Monitoring Demo
        demonstrate_resource_monitoring()
        
        # Brief pause
        time.sleep(2)
        
        # 5. Web Dashboard Demo (optional)
        logger.info("Starting web dashboard demo (optional)...")
        run_dashboard_server_demo()
        
        # Final summary
        logger.info("=== Demo Summary ===")
        logger.info(f"✅ Completed {len(completed_jobs)} simulation jobs with progress tracking")
        logger.info("✅ Demonstrated hardware job status monitoring")
        logger.info("✅ Showcased dashboard API functionality")
        logger.info("✅ Monitored system resources during execution")
        logger.info("✅ Ran interactive web dashboard (if HTTP available)")
        
        # Show final dashboard state
        dashboard = get_dashboard()
        final_state = dashboard.get_current_state()
        logger.info(f"Final Dashboard State: {len(final_state.active_jobs)} active jobs, "
                   f"{len(final_state.notifications)} notifications")
        
        logger.info("🎉 Real-Time Execution Monitoring Dashboard Demo Completed Successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 