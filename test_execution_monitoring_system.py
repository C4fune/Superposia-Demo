#!/usr/bin/env python3
"""
Test Suite for Real-Time Execution Monitoring Dashboard

This test suite validates all components of the execution monitoring system
including job management, progress tracking, status monitoring, and dashboard
functionality.
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Quantum Platform Components
from quantum_platform.compiler.language.dsl import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, CNOT
from quantum_platform.simulation.executor import MonitoredSimulationExecutor

# Execution Monitoring Components
from quantum_platform.execution.job_manager import (
    JobManager, ExecutionJob, JobStatus, JobType, get_job_manager
)
from quantum_platform.execution.progress_tracker import (
    ProgressTracker, SimulationProgress, ProgressType,
    create_simulation_tracker, MultiStageProgressTracker
)
from quantum_platform.execution.status_monitor import (
    StatusMonitor, HardwareJobMonitor, HardwareJobInfo, StatusUpdate,
    get_status_monitor
)
from quantum_platform.execution.dashboard import (
    ExecutionDashboard, DashboardAPI, DashboardNotification,
    get_dashboard
)

class TestJobManager(unittest.TestCase):
    """Test cases for JobManager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.job_manager = JobManager(max_concurrent_jobs=2, cleanup_interval=3600)
    
    def tearDown(self):
        """Clean up after tests."""
        self.job_manager.shutdown()
    
    def test_create_job(self):
        """Test job creation."""
        job = self.job_manager.create_job(
            job_type=JobType.SIMULATION,
            name="Test Job",
            circuit_name="test_circuit",
            backend_name="statevector",
            shots=1000
        )
        
        self.assertIsInstance(job, ExecutionJob)
        self.assertEqual(job.job_type, JobType.SIMULATION)
        self.assertEqual(job.name, "Test Job")
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertEqual(job.shots, 1000)
        self.assertIsNotNone(job.job_id)
        self.assertIsNotNone(job.created_at)
    
    def test_job_lifecycle(self):
        """Test complete job lifecycle."""
        job = self.job_manager.create_job(
            job_type=JobType.SIMULATION,
            name="Lifecycle Test"
        )
        
        # Test initial state
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertFalse(job.is_active)
        self.assertFalse(job.is_finished)
        
        # Test starting
        job.start()
        self.assertEqual(job.status, JobStatus.RUNNING)
        self.assertTrue(job.is_active)
        self.assertFalse(job.is_finished)
        self.assertIsNotNone(job.started_at)
        
        # Test progress update
        job.update_progress(50.0, "Half done")
        self.assertEqual(job.progress, 50.0)
        
        # Test completion
        job.complete("Test result")
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertFalse(job.is_active)
        self.assertTrue(job.is_finished)
        self.assertEqual(job.result, "Test result")
        self.assertIsNotNone(job.completed_at)
    
    def test_job_cancellation(self):
        """Test job cancellation."""
        job = self.job_manager.create_job(
            job_type=JobType.SIMULATION,
            name="Cancel Test"
        )
        
        # Cancel job
        success = self.job_manager.cancel_job(job.job_id)
        self.assertTrue(success)
        self.assertEqual(job.status, JobStatus.CANCELLED)
        self.assertTrue(job.is_cancelled())
    
    def test_job_submission_and_execution(self):
        """Test job submission and execution."""
        job = self.job_manager.create_job(
            job_type=JobType.SIMULATION,
            name="Execution Test"
        )
        
        # Mock executor function
        def mock_executor(j):
            time.sleep(0.1)  # Simulate work
            return "Mock result"
        
        # Submit job
        success = self.job_manager.submit_job(job, mock_executor)
        self.assertTrue(success)
        
        # Wait for completion
        timeout = time.time() + 5.0
        while not job.is_finished and time.time() < timeout:
            time.sleep(0.01)
        
        self.assertTrue(job.is_finished)
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertEqual(job.result, "Mock result")
    
    def test_job_statistics(self):
        """Test job statistics tracking."""
        # Create some jobs
        for i in range(3):
            job = self.job_manager.create_job(
                job_type=JobType.SIMULATION,
                name=f"Stats Test {i}"
            )
            if i == 0:
                job.complete()
            elif i == 1:
                job.fail("Test failure")
        
        stats = self.job_manager.get_job_statistics()
        self.assertGreaterEqual(stats['total_created'], 3)
        self.assertGreaterEqual(stats['total_completed'], 1)
        self.assertGreaterEqual(stats['total_failed'], 1)

class TestProgressTracker(unittest.TestCase):
    """Test cases for ProgressTracker functionality."""
    
    def test_progress_tracker_creation(self):
        """Test progress tracker creation and basic functionality."""
        tracker = ProgressTracker("Test Operation", ProgressType.PERCENTAGE)
        
        self.assertEqual(tracker.operation_name, "Test Operation")
        self.assertEqual(tracker.progress_type, ProgressType.PERCENTAGE)
        self.assertFalse(tracker.is_active)
    
    def test_progress_updates(self):
        """Test progress updates."""
        tracker = ProgressTracker("Test Operation")
        
        # Start tracking
        tracker.start()
        self.assertTrue(tracker.is_active)
        
        # Update progress
        tracker.update_percentage(25.0, "Quarter done")
        self.assertEqual(tracker.progress.percentage, 25.0)
        self.assertEqual(tracker.progress.message, "Quarter done")
        
        # Complete
        tracker.complete("Done")
        self.assertFalse(tracker.is_active)
        self.assertEqual(tracker.progress.percentage, 100.0)
        self.assertEqual(tracker.progress.message, "Done")
    
    def test_simulation_progress_tracker(self):
        """Test simulation-specific progress tracker."""
        tracker = create_simulation_tracker("Test Simulation", 1000)
        
        tracker.start_simulation()
        tracker.update_shot_progress(250, "Quarter shots done")
        
        self.assertEqual(tracker.progress.current_shot, 250)
        self.assertEqual(tracker.progress.total_shots, 1000)
        self.assertEqual(tracker.progress.percentage, 25.0)
    
    def test_multi_stage_progress_tracker(self):
        """Test multi-stage progress tracker."""
        stages = ["Initialize", "Compute", "Finalize"]
        tracker = MultiStageProgressTracker("Multi-stage Op", stages)
        
        tracker.start_multi_stage()
        
        # First stage
        tracker.update_stage_progress(50.0)
        self.assertAlmostEqual(tracker.progress.percentage, 50.0 / 3, places=1)
        
        # Complete first stage
        tracker.complete_stage()
        tracker.update_stage_progress(0.0)  # Start second stage
        
        # Complete all stages
        tracker.complete_stage()
        tracker.complete_stage()
        
        self.assertFalse(tracker.is_active)
    
    def test_progress_callbacks(self):
        """Test progress update callbacks."""
        tracker = ProgressTracker("Callback Test")
        callback_called = []
        
        def test_callback(progress):
            callback_called.append(progress.percentage)
        
        tracker.add_callback(test_callback)
        tracker.start()
        tracker.update_percentage(50.0)
        tracker.complete()
        
        self.assertGreater(len(callback_called), 0)
        self.assertIn(50.0, callback_called)

class TestStatusMonitor(unittest.TestCase):
    """Test cases for StatusMonitor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.status_monitor = StatusMonitor()
    
    def tearDown(self):
        """Clean up after tests."""
        self.status_monitor.shutdown()
    
    def test_hardware_job_monitoring(self):
        """Test hardware job monitoring."""
        job_info = HardwareJobInfo(
            job_id="test_hw_job",
            provider_job_id="provider_123",
            provider_name="Test Provider",
            device_name="test_device",
            status="queued",
            queue_position=5
        )
        
        # Start monitoring
        job_id = self.status_monitor.add_hardware_job(job_info)
        self.assertEqual(job_id, "test_hw_job")
        
        # Check status
        retrieved_job = self.status_monitor.get_hardware_job_status(job_id)
        self.assertIsNotNone(retrieved_job)
        self.assertEqual(retrieved_job.provider_name, "Test Provider")
        self.assertEqual(retrieved_job.queue_position, 5)
        
        # Remove monitoring
        success = self.status_monitor.remove_hardware_job(job_id)
        self.assertTrue(success)
        
        # Verify removal
        retrieved_job = self.status_monitor.get_hardware_job_status(job_id)
        self.assertIsNone(retrieved_job)
    
    def test_status_updates(self):
        """Test status update recording and callbacks."""
        callback_called = []
        
        def test_callback(update):
            callback_called.append(update)
        
        self.status_monitor.add_global_callback(test_callback)
        
        # Add a hardware job to generate updates
        job_info = HardwareJobInfo(
            job_id="test_updates",
            provider_job_id="provider_456",
            provider_name="Update Test",
            device_name="test_device"
        )
        
        job_id = self.status_monitor.add_hardware_job(job_info)
        
        # Allow some time for status updates
        time.sleep(0.5)
        
        # Check that callbacks were called
        self.assertGreater(len(callback_called), 0)
        
        # Clean up
        self.status_monitor.remove_hardware_job(job_id)
    
    def test_monitoring_statistics(self):
        """Test monitoring statistics."""
        # Add some hardware jobs
        for i in range(3):
            job_info = HardwareJobInfo(
                job_id=f"stats_test_{i}",
                provider_job_id=f"provider_{i}",
                provider_name="Stats Provider",
                device_name="stats_device"
            )
            self.status_monitor.add_hardware_job(job_info)
        
        stats = self.status_monitor.get_monitoring_statistics()
        self.assertEqual(stats['active_hardware_monitors'], 3)
        self.assertGreaterEqual(stats['hardware_jobs_added'], 3)
        
        # Clean up
        for i in range(3):
            self.status_monitor.remove_hardware_job(f"stats_test_{i}")

class TestExecutionDashboard(unittest.TestCase):
    """Test cases for ExecutionDashboard functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.dashboard = ExecutionDashboard(update_interval=0.1)
    
    def tearDown(self):
        """Clean up after tests."""
        self.dashboard.shutdown()
    
    def test_dashboard_state(self):
        """Test dashboard state management."""
        state = self.dashboard.get_current_state()
        
        self.assertIsInstance(state.active_jobs, list)
        self.assertIsInstance(state.hardware_jobs, list)
        self.assertIsInstance(state.system_status, dict)
        self.assertIsInstance(state.notifications, list)
        self.assertIsNotNone(state.last_updated)
    
    def test_dashboard_api(self):
        """Test dashboard API functionality."""
        api = self.dashboard.api
        
        # Test getting dashboard state
        state = api.get_dashboard_state()
        self.assertIn('active_jobs', state)
        self.assertIn('system_status', state)
        self.assertIn('notifications', state)
        
        # Test getting statistics
        stats = api.get_statistics()
        self.assertIn('jobs', stats)
        self.assertIn('monitoring', stats)
        self.assertIn('system', stats)
    
    def test_notifications(self):
        """Test notification system."""
        # Add a notification
        self.dashboard._add_notification(
            "Test Notification",
            "This is a test message",
            "info"
        )
        
        # Check recent notifications
        notifications = self.dashboard.get_recent_notifications(1)
        self.assertGreater(len(notifications), 0)
        
        # Find our test notification
        test_notif = None
        for notif in notifications:
            if notif['title'] == "Test Notification":
                test_notif = notif
                break
        
        self.assertIsNotNone(test_notif)
        self.assertEqual(test_notif['message'], "This is a test message")
        self.assertEqual(test_notif['type'], "info")
    
    def test_export_functionality(self):
        """Test dashboard data export."""
        export_data = self.dashboard.export_dashboard_data("dict")
        
        self.assertIn('current_state', export_data)
        self.assertIn('notifications', export_data)
        self.assertIn('export_timestamp', export_data)
        
        # Test JSON export
        json_export = self.dashboard.export_dashboard_data("json")
        self.assertIsInstance(json_export, str)

class TestMonitoredSimulationExecutor(unittest.TestCase):
    """Test cases for MonitoredSimulationExecutor."""
    
    def setUp(self):
        """Set up test environment."""
        self.executor = MonitoredSimulationExecutor(enable_monitoring=True)
    
    def test_circuit_execution_with_monitoring(self):
        """Test circuit execution with monitoring enabled."""
        # Create a simple test circuit
        with QuantumProgram("Test Circuit") as qp:
            q0, q1 = qp.allocate(2)
            H(q0)
            CNOT(q0, q1)
            qp.measure(q0, 0)
            qp.measure(q1, 1)
        
        # Execute with monitoring
        job = self.executor.execute_circuit(
            circuit=qp.circuit,
            shots=100,  # Small number for quick test
            job_name="Monitor Test",
            enable_progress_tracking=False  # Disable for quick test
        )
        
        self.assertIsInstance(job, ExecutionJob)
        self.assertEqual(job.job_type, JobType.SIMULATION)
        
        # Wait for completion
        result = self.executor.wait_for_job(job, timeout=10.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.shots, 100)
    
    def test_executor_without_monitoring(self):
        """Test executor with monitoring disabled."""
        executor = MonitoredSimulationExecutor(enable_monitoring=False)
        
        with QuantumProgram("No Monitor Test") as qp:
            q = qp.allocate(1)
            H(q)
            qp.measure(q, 0)
        
        # Should return result directly
        result = executor.execute_circuit(
            circuit=qp.circuit,
            shots=50
        )
        
        # Should be a SimulationResult, not ExecutionJob
        from quantum_platform.simulation.base import SimulationResult
        self.assertIsInstance(result, SimulationResult)
    
    def test_job_cancellation(self):
        """Test job cancellation functionality."""
        with QuantumProgram("Cancel Test") as qp:
            qubits = qp.allocate(3)
            for q in qubits:
                H(q)
            for i, q in enumerate(qubits):
                qp.measure(q, i)
        
        # Submit a longer job
        job = self.executor.execute_circuit(
            circuit=qp.circuit,
            shots=5000,  # Larger number to allow cancellation
            job_name="Cancellation Test"
        )
        
        # Brief delay to let job start
        time.sleep(0.1)
        
        # Cancel the job
        success = self.executor.cancel_job(job.job_id)
        self.assertTrue(success)
        
        # Verify cancellation
        time.sleep(0.1)
        self.assertTrue(job.is_cancelled() or job.status == JobStatus.CANCELLED)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete monitoring system."""
    
    def test_end_to_end_monitoring(self):
        """Test complete end-to-end monitoring workflow."""
        # Get global instances
        job_manager = get_job_manager()
        status_monitor = get_status_monitor()
        dashboard = get_dashboard()
        
        # Create and execute a monitored job
        executor = MonitoredSimulationExecutor()
        
        with QuantumProgram("Integration Test") as qp:
            q0, q1 = qp.allocate(2)
            H(q0)
            CNOT(q0, q1)
            qp.measure(q0, 0)
            qp.measure(q1, 1)
        
        job = executor.execute_circuit(
            circuit=qp.circuit,
            shots=200,
            job_name="Integration Test Job"
        )
        
        # Verify job is tracked by job manager
        tracked_job = job_manager.get_job(job.job_id)
        self.assertIsNotNone(tracked_job)
        self.assertEqual(tracked_job.job_id, job.job_id)
        
        # Wait for job completion
        result = executor.wait_for_job(job, timeout=10.0)
        self.assertIsNotNone(result)
        
        # Verify dashboard state reflects completion
        dashboard_state = dashboard.get_current_state()
        # Note: Job might not be in active_jobs anymore if completed quickly
        
        # Verify statistics are updated
        stats = job_manager.get_job_statistics()
        self.assertGreater(stats['total_completed'], 0)
    
    def test_concurrent_job_execution(self):
        """Test multiple concurrent jobs."""
        executor = MonitoredSimulationExecutor()
        
        # Create multiple simple circuits
        jobs = []
        for i in range(3):
            with QuantumProgram(f"Concurrent Test {i}") as qp:
                q = qp.allocate(1)
                H(q)
                qp.measure(q, 0)
            
            job = executor.execute_circuit(
                circuit=qp.circuit,
                shots=100,
                job_name=f"Concurrent Job {i}"
            )
            jobs.append(job)
        
        # Wait for all jobs to complete
        results = []
        for job in jobs:
            result = executor.wait_for_job(job, timeout=10.0)
            results.append(result)
        
        # Verify all jobs completed successfully
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNotNone(result)
            self.assertEqual(result.shots, 100)

def run_performance_test():
    """Run a basic performance test of the monitoring system."""
    print("\n=== Performance Test ===")
    
    executor = MonitoredSimulationExecutor()
    
    # Create a moderately complex circuit
    with QuantumProgram("Performance Test") as qp:
        qubits = qp.allocate(4)
        
        # Create some complexity
        for q in qubits:
            H(q)
        
        for i in range(len(qubits) - 1):
            CNOT(qubits[i], qubits[i + 1])
        
        for q in qubits:
            H(q)
        
        for i, q in enumerate(qubits):
            qp.measure(q, i)
    
    # Test execution time with monitoring
    start_time = time.time()
    job = executor.execute_circuit(
        circuit=qp.circuit,
        shots=1000,
        job_name="Performance Test"
    )
    result = executor.wait_for_job(job)
    end_time = time.time()
    
    print(f"Monitored execution time: {end_time - start_time:.3f}s")
    print(f"Simulation completed with {result.shots} shots")
    print(f"Job status: {job.status.value}")
    print(f"Final progress: {job.progress:.1f}%")

if __name__ == "__main__":
    # Run the test suite
    print("🧪 Running Real-Time Execution Monitoring Test Suite")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestJobManager,
        TestProgressTracker,
        TestStatusMonitor,
        TestExecutionDashboard,
        TestMonitoredSimulationExecutor,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance test
    try:
        run_performance_test()
    except Exception as e:
        print(f"Performance test failed: {e}")
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed.'}")
    
    exit(0 if success else 1) 