#!/usr/bin/env python3
"""
Comprehensive Tests for Performance Profiling and Benchmarking System

This test suite validates the complete performance profiling and benchmarking
functionality, including integration with existing quantum platform components.

Run with: python -m pytest test_performance_profiling_system.py -v
"""

import pytest
import time
import asyncio
import tempfile
import os
from datetime import datetime, timedelta

# Test quantum platform components
from quantum_platform.compiler.ir.circuit import QuantumCircuit
from quantum_platform.simulation.statevector import StateVectorSimulator
from quantum_platform.execution.job_manager import JobManager, JobType

# Test profiling components
try:
    from quantum_platform.profiling import (
        QuantumProfiler, ProfilerConfig, ProfilerMode,
        SimulationProfiler, HardwareProfiler, CompilerProfiler, MemoryProfiler,
        QuantumBenchmark, BenchmarkSuite, BenchmarkResult, ScalingAnalysis,
        ProfileReportGenerator, ReportFormat,
        get_profiler, configure_profiler, profile_execution
    )
    PROFILING_AVAILABLE = True
except ImportError as e:
    print(f"Profiling components not available: {e}")
    PROFILING_AVAILABLE = False

# Test observability integration
from quantum_platform.observability.logging import get_logger


@pytest.mark.skipif(not PROFILING_AVAILABLE, reason="Profiling components not available")
class TestQuantumProfilerCore:
    """Test core profiler functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.config = ProfilerConfig(
            mode=ProfilerMode.BASIC,
            profile_simulation=True,
            track_gate_timing=True,
            track_memory_usage=True
        )
        self.profiler = QuantumProfiler(self.config)
    
    def test_profiler_initialization(self):
        """Test profiler initialization with different configurations."""
        # Test basic initialization
        assert self.profiler.config.mode == ProfilerMode.BASIC
        assert self.profiler.config.profile_simulation is True
        assert len(self.profiler.get_active_profiles()) == 0
        assert len(self.profiler.get_profile_history()) == 0
        
        # Test detailed mode
        detailed_config = ProfilerConfig(mode=ProfilerMode.DETAILED)
        detailed_profiler = QuantumProfiler(detailed_config)
        assert detailed_profiler.config.mode == ProfilerMode.DETAILED
        
        # Test disabled mode
        disabled_config = ProfilerConfig(mode=ProfilerMode.DISABLED)
        disabled_profiler = QuantumProfiler(disabled_config)
        assert disabled_profiler.config.mode == ProfilerMode.DISABLED
    
    def test_basic_profiling_session(self):
        """Test basic profiling session lifecycle."""
        profile_id = "test_basic_session"
        
        # Start profiling
        profile_data = self.profiler.start_profile(
            profile_id,
            circuit_name="test_circuit",
            backend_name="test_backend"
        )
        
        assert profile_data is not None
        assert profile_data.profile_id == profile_id
        assert profile_data.circuit_name == "test_circuit"
        assert profile_data.backend_name == "test_backend"
        assert profile_id in self.profiler.get_active_profiles()
        
        # Simulate some work
        time.sleep(0.1)
        
        # Stop profiling
        completed_profile = self.profiler.stop_profile(profile_id)
        
        assert completed_profile is not None
        assert completed_profile.total_duration >= 0.1
        assert profile_id not in self.profiler.get_active_profiles()
        assert len(self.profiler.get_profile_history()) == 1
    
    def test_disabled_profiler(self):
        """Test that disabled profiler doesn't collect data."""
        disabled_config = ProfilerConfig(mode=ProfilerMode.DISABLED)
        disabled_profiler = QuantumProfiler(disabled_config)
        
        # Start profiling (should return None)
        profile_data = disabled_profiler.start_profile("disabled_test")
        assert profile_data is None
        
        # Stop profiling (should return None)
        completed_profile = disabled_profiler.stop_profile("disabled_test")
        assert completed_profile is None


class TestBasicQuantumOperations:
    """Test basic quantum operations without profiling dependencies."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.simulator = StateVectorSimulator(max_qubits=10)
    
    def test_basic_circuit_execution(self):
        """Test basic quantum circuit execution."""
        # Create a simple circuit
        circuit = QuantumCircuit("test_circuit", num_qubits=2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        # Execute circuit
        result = self.simulator.run(circuit, shots=1000)
        
        assert result.success is True
        assert result.circuit_name == "test_circuit"
        assert result.shots == 1000
        assert result.execution_time > 0
        assert len(result.measurement_counts) > 0
    
    def test_job_manager_basic_functionality(self):
        """Test basic job manager functionality."""
        job_manager = JobManager()
        
        try:
            # Create a job
            job = job_manager.create_job(
                job_type=JobType.SIMULATION,
                name="Test Job",
                circuit_name="test_circuit",
                shots=100
            )
            
            assert job.name == "Test Job"
            assert job.job_type == JobType.SIMULATION
            assert job.circuit_name == "test_circuit"
            assert job.shots == 100
            
            # Test job execution
            def simple_executor(execution_job):
                """Simple job executor."""
                execution_job.update_progress(50, "Running")
                time.sleep(0.1)
                execution_job.update_progress(100, "Completed")
                return "success"
            
            success = job_manager.submit_job(job, simple_executor)
            assert success is True
            
            # Wait for completion
            timeout = time.time() + 5
            while job.is_active and time.time() < timeout:
                time.sleep(0.1)
            
            assert job.is_finished
            assert job.progress == 100.0
            
        finally:
            job_manager.shutdown()


@pytest.mark.skipif(not PROFILING_AVAILABLE, reason="Profiling components not available")
class TestSimulationProfiling:
    """Test simulation-specific profiling functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.config = ProfilerConfig(track_gate_timing=True, track_memory_usage=True)
        self.simulation_profiler = SimulationProfiler(self.config)
        self.simulator = StateVectorSimulator(max_qubits=10)
    
    def test_simulation_profiling_session(self):
        """Test complete simulation profiling session."""
        profile_id = "test_simulation"
        
        # Start profiling
        profile = self.simulation_profiler.start_profile(profile_id)
        assert profile.profile_id == profile_id
        
        # Set circuit information
        self.simulation_profiler.set_circuit_info(
            circuit_name="test_circuit",
            num_qubits=3,
            total_gates=5,
            circuit_depth=3
        )
        
        # Simulate phases
        self.simulation_profiler.start_phase("initialization")
        time.sleep(0.01)
        self.simulation_profiler.end_phase("initialization")
        
        self.simulation_profiler.start_phase("gate_application")
        time.sleep(0.02)
        self.simulation_profiler.end_phase("gate_application")
        
        # Record memory usage
        self.simulation_profiler.record_memory_usage(100.0)
        
        # Stop profiling
        completed_profile = self.simulation_profiler.stop_profile(profile_id)
        
        assert completed_profile is not None
        assert completed_profile.circuit_name == "test_circuit"
        assert completed_profile.num_qubits == 3
        assert completed_profile.total_gates == 5
        assert completed_profile.peak_memory_mb == 100.0


@pytest.mark.skipif(not PROFILING_AVAILABLE, reason="Profiling components not available")
class TestBenchmarking:
    """Test quantum benchmarking functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.simulator = StateVectorSimulator(max_qubits=10)
        self.benchmark = QuantumBenchmark(
            "test_benchmark",
            "Test Benchmark",
            "A test benchmark for validation"
        )
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        assert self.benchmark.benchmark_id == "test_benchmark"
        assert self.benchmark.name == "Test Benchmark"
        assert self.benchmark.description == "A test benchmark for validation"
        assert len(self.benchmark.results) == 0
    
    def test_single_benchmark_run(self):
        """Test single benchmark execution."""
        def test_function(num_qubits: int, shots: int):
            """Simple test function."""
            circuit = QuantumCircuit("test", num_qubits=num_qubits)
            for i in range(num_qubits):
                circuit.h(i)
            circuit.measure_all()
            
            result = self.simulator.run(circuit, shots=shots)
            return {
                'success': result.success,
                'metrics': {
                    'execution_time': result.execution_time,
                    'memory_used': result.memory_used
                }
            }
        
        # Run benchmark
        result = self.benchmark.run_single(
            test_function,
            {'num_qubits': 3, 'shots': 100},
            circuit_name="test_circuit"
        )
        
        assert result.success is True
        assert result.execution_time > 0
        assert result.circuit_name == "test_circuit"
        assert len(self.benchmark.results) == 1
        
        # Test summary statistics
        summary = self.benchmark.get_summary_statistics()
        assert summary['total_runs'] == 1
        assert summary['successful_runs'] == 1
        assert summary['success_rate'] == 1.0


@pytest.mark.skipif(not PROFILING_AVAILABLE, reason="Profiling components not available")
class TestIntegration:
    """Test integration with existing quantum platform components."""
    
    def setup_method(self):
        """Setup for integration tests."""
        # Configure profiler
        config = ProfilerConfig(
            mode=ProfilerMode.BASIC,
            profile_simulation=True,
            track_memory_usage=True
        )
        configure_profiler(config)
        
        self.profiler = get_profiler()
        self.simulator = StateVectorSimulator(max_qubits=10)
        self.job_manager = JobManager()
    
    def teardown_method(self):
        """Cleanup after integration tests."""
        self.job_manager.shutdown()
    
    def test_profiling_with_circuit_execution(self):
        """Test profiling integration with quantum circuit execution."""
        # Create a test circuit
        circuit = QuantumCircuit("integration_test", num_qubits=3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.measure_all()
        
        # Execute with profiling
        with profile_execution("integration_test", circuit_name=circuit.name) as profile_data:
            result = self.simulator.run(circuit, shots=1000)
            
            # Verify execution succeeded
            assert result.success is True
            assert result.circuit_name == "integration_test"
        
        # Verify profiling data was collected
        history = self.profiler.get_profile_history(limit=1)
        assert len(history) == 1
        
        profile = history[0]
        assert profile.circuit_name == "integration_test"
        assert profile.total_duration > 0


def test_quantum_platform_imports():
    """Test that core quantum platform components can be imported."""
    # Test core circuit components
    from quantum_platform.compiler.ir.circuit import QuantumCircuit
    from quantum_platform.simulation.statevector import StateVectorSimulator
    from quantum_platform.execution.job_manager import JobManager
    
    # Test observability components
    from quantum_platform.observability.logging import get_logger
    
    # Verify imports work
    assert QuantumCircuit is not None
    assert StateVectorSimulator is not None
    assert JobManager is not None
    assert get_logger is not None


def test_profiling_availability():
    """Test profiling component availability."""
    try:
        from quantum_platform.profiling import QuantumProfiler, ProfilerConfig
        profiling_available = True
    except ImportError:
        profiling_available = False
    
    # This test just documents the availability
    print(f"Profiling components available: {profiling_available}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"]) 