#!/usr/bin/env python3
"""
Quantum Platform Performance Profiling and Benchmarking Example

This example demonstrates the comprehensive performance profiling and benchmarking
capabilities of the quantum platform, including:

1. Real-time execution profiling
2. Hardware timing analysis
3. Memory usage tracking
4. Compilation profiling
5. Benchmarking and scaling studies
6. Performance report generation

Run with: python example_performance_profiling.py
"""

import time
import asyncio
from datetime import datetime, timedelta

# Quantum Platform imports
from quantum_platform.compiler.ir.circuit import QuantumCircuit
from quantum_platform.simulation.statevector import StateVectorSimulator
from quantum_platform.execution.job_manager import JobManager, JobType

# Profiling imports
from quantum_platform.profiling import (
    QuantumProfiler, ProfilerConfig, ProfilerMode,
    SimulationProfiler, HardwareProfiler, CompilerProfiler, MemoryProfiler,
    QuantumBenchmark, BenchmarkSuite, BenchmarkResult,
    ProfileReportGenerator, ReportFormat,
    get_profiler, configure_profiler, profile_execution
)

# Observability imports
from quantum_platform.observability.logging import get_logger


class PerformanceProfilingDemo:
    """Demonstration of quantum performance profiling capabilities."""
    
    def __init__(self):
        """Initialize the profiling demo."""
        self.logger = get_logger(__name__)
        
        # Configure profiler for detailed analysis
        profiler_config = ProfilerConfig(
            mode=ProfilerMode.DETAILED,
            profile_simulation=True,
            profile_hardware=True,
            profile_compilation=True,
            profile_memory=True,
            track_gate_timing=True,
            track_memory_usage=True,
            memory_sample_interval=0.1,
            max_overhead_percent=10.0
        )
        configure_profiler(profiler_config)
        
        # Initialize components
        self.simulator = StateVectorSimulator(max_qubits=20)
        self.job_manager = JobManager()
        self.profiler = get_profiler()
        self.report_generator = ProfileReportGenerator()
        
        self.logger.info("Performance Profiling Demo initialized")
    
    def create_test_circuits(self) -> dict:
        """Create various test circuits for profiling."""
        circuits = {}
        
        # Small circuit for basic profiling
        small_circuit = QuantumCircuit("small_test", num_qubits=3)
        small_circuit.h(0)
        small_circuit.cx(0, 1)
        small_circuit.cx(1, 2)
        small_circuit.measure_all()
        circuits['small'] = small_circuit
        
        # Medium circuit for scaling analysis
        medium_circuit = QuantumCircuit("medium_test", num_qubits=8)
        for i in range(8):
            medium_circuit.h(i)
        for i in range(7):
            medium_circuit.cx(i, i+1)
        for i in range(8):
            medium_circuit.rz(0.5, i)
        medium_circuit.measure_all()
        circuits['medium'] = medium_circuit
        
        # Large circuit for stress testing
        large_circuit = QuantumCircuit("large_test", num_qubits=15)
        # Create a more complex circuit
        for layer in range(5):
            for i in range(15):
                large_circuit.h(i)
            for i in range(0, 14, 2):
                large_circuit.cx(i, i+1)
            for i in range(1, 14, 2):
                large_circuit.cx(i, i+1)
            for i in range(15):
                large_circuit.rz(0.1 * layer, i)
        large_circuit.measure_all()
        circuits['large'] = large_circuit
        
        return circuits
    
    async def demonstrate_basic_profiling(self):
        """Demonstrate basic execution profiling."""
        print("\n" + "="*60)
        print("BASIC EXECUTION PROFILING DEMONSTRATION")
        print("="*60)
        
        circuits = self.create_test_circuits()
        
        # Profile a simple circuit execution
        with profile_execution("basic_demo", circuit_name="small_test") as profile_data:
            self.logger.info("Executing small circuit with profiling...")
            
            circuit = circuits['small']
            result = self.simulator.run(
                circuit, 
                shots=1000, 
                return_statevector=True
            )
            
            print(f"✓ Circuit executed: {result.circuit_name}")
            print(f"  Shots: {result.shots}")
            print(f"  Execution time: {result.execution_time:.3f}s")
            print(f"  Memory used: {result.memory_used / (1024**2):.1f}MB")
        
        # Generate and display report
        completed_profile = self.profiler.stop_profile("basic_demo")
        if completed_profile:
            report = self.profiler.generate_report(completed_profile)
            print("\n" + report.get_formatted_summary())
    
    async def demonstrate_simulation_profiling(self):
        """Demonstrate detailed simulation profiling."""
        print("\n" + "="*60)
        print("DETAILED SIMULATION PROFILING DEMONSTRATION")
        print("="*60)
        
        circuits = self.create_test_circuits()
        simulation_profiler = SimulationProfiler(self.profiler.config)
        
        # Profile simulation with gate-level timing
        for size, circuit in circuits.items():
            print(f"\nProfiling {size} circuit ({circuit.num_qubits} qubits)...")
            
            profile_id = f"sim_profile_{size}"
            sim_profile = simulation_profiler.start_profile(profile_id)
            simulation_profiler.set_circuit_info(
                circuit_name=circuit.name,
                num_qubits=circuit.num_qubits,
                total_gates=len(circuit.operations),
                circuit_depth=circuit.depth()
            )
            
            # Simulate with profiling
            simulation_profiler.start_phase("initialization")
            time.sleep(0.01)  # Simulate initialization overhead
            simulation_profiler.end_phase("initialization")
            
            simulation_profiler.start_phase("gate_application")
            result = self.simulator.run(circuit, shots=500)
            simulation_profiler.end_phase("gate_application")
            
            simulation_profiler.start_phase("measurement")
            time.sleep(0.005)  # Simulate measurement overhead
            simulation_profiler.end_phase("measurement")
            
            # Record memory usage
            simulation_profiler.record_memory_usage(result.memory_used / (1024**2))
            
            completed_profile = simulation_profiler.stop_profile(profile_id)
            
            if completed_profile:
                print(f"  Total time: {completed_profile.total_time:.3f}s")
                print(f"  Gates/second: {completed_profile.gates_per_second:.1f}")
                print(f"  Peak memory: {completed_profile.peak_memory_mb:.1f}MB")
                
                breakdown = completed_profile.execution_breakdown
                print(f"  Phase breakdown:")
                print(f"    Initialization: {breakdown.initialization_time:.3f}s")
                print(f"    Gate application: {breakdown.gate_application_time:.3f}s")
                print(f"    Measurement: {breakdown.measurement_time:.3f}s")
    
    async def demonstrate_memory_profiling(self):
        """Demonstrate memory usage profiling."""
        print("\n" + "="*60)
        print("MEMORY USAGE PROFILING DEMONSTRATION")
        print("="*60)
        
        circuits = self.create_test_circuits()
        memory_profiler = MemoryProfiler(self.profiler.config)
        
        # Profile memory usage for different circuit sizes
        for size, circuit in circuits.items():
            print(f"\nMemory profiling {size} circuit...")
            
            profile_id = f"memory_profile_{size}"
            memory_profile = memory_profiler.start_profile(
                profile_id,
                operation_type="simulation",
                num_qubits=circuit.num_qubits
            )
            
            # Simulate execution with memory tracking
            result = self.simulator.run(circuit, shots=1000)
            
            # Record quantum-specific memory usage
            state_vector_mb = (2 ** circuit.num_qubits) * 16 / (1024**2)  # Complex128
            memory_profiler.record_quantum_memory(
                state_vector_mb=state_vector_mb,
                circuit_mb=0.1,  # Estimated circuit representation
                result_mb=0.05   # Estimated result storage
            )
            
            completed_profile = memory_profiler.stop_profile(profile_id)
            
            if completed_profile:
                usage = completed_profile.memory_usage
                print(f"  Theoretical memory: {completed_profile.theoretical_memory_mb:.1f}MB")
                print(f"  Peak memory: {usage.peak_process_memory_mb:.1f}MB")
                print(f"  Memory efficiency: {usage.memory_efficiency:.1%}")
                print(f"  State vector memory: {usage.peak_state_vector_memory_mb:.1f}MB")
                
                if completed_profile.memory_warnings:
                    print(f"  Warnings: {len(completed_profile.memory_warnings)}")
                    for warning in completed_profile.memory_warnings:
                        print(f"    ⚠️  {warning}")
    
    def create_benchmark_suite(self) -> BenchmarkSuite:
        """Create a comprehensive benchmark suite."""
        suite = BenchmarkSuite(
            "quantum_performance_suite", 
            "Comprehensive quantum performance benchmarks"
        )
        
        # Circuit execution benchmark
        execution_benchmark = QuantumBenchmark(
            "circuit_execution",
            "Circuit Execution Performance",
            "Measures basic circuit execution performance"
        )
        suite.add_benchmark(execution_benchmark)
        
        # Memory efficiency benchmark
        memory_benchmark = QuantumBenchmark(
            "memory_efficiency",
            "Memory Usage Efficiency",
            "Measures memory efficiency for different circuit sizes"
        )
        suite.add_benchmark(memory_benchmark)
        
        # Scaling performance benchmark
        scaling_benchmark = QuantumBenchmark(
            "scaling_performance",
            "Scaling Performance Analysis",
            "Analyzes performance scaling with circuit parameters"
        )
        suite.add_benchmark(scaling_benchmark)
        
        return suite
    
    async def demonstrate_benchmarking(self):
        """Demonstrate comprehensive benchmarking."""
        print("\n" + "="*60)
        print("QUANTUM BENCHMARKING DEMONSTRATION")
        print("="*60)
        
        circuits = self.create_test_circuits()
        suite = self.create_benchmark_suite()
        
        # Define test functions for benchmarks
        def circuit_execution_test(circuit_name: str, shots: int = 1000):
            """Test function for circuit execution benchmark."""
            circuit = circuits[circuit_name]
            result = self.simulator.run(circuit, shots=shots)
            return {
                'success': result.success,
                'metrics': {
                    'execution_time': result.execution_time,
                    'memory_used': result.memory_used,
                    'measurement_count': len(result.measurement_counts)
                }
            }
        
        def memory_efficiency_test(circuit_name: str):
            """Test function for memory efficiency benchmark."""
            circuit = circuits[circuit_name]
            result = self.simulator.run(circuit, shots=100)
            
            theoretical_memory = (2 ** circuit.num_qubits) * 16  # bytes
            actual_memory = result.memory_used
            efficiency = theoretical_memory / actual_memory if actual_memory > 0 else 0
            
            return {
                'success': result.success,
                'metrics': {
                    'memory_efficiency': efficiency,
                    'theoretical_memory': theoretical_memory,
                    'actual_memory': actual_memory
                }
            }
        
        # Run benchmark suite
        test_functions = {
            'circuit_execution': circuit_execution_test,
            'memory_efficiency': memory_efficiency_test
        }
        
        parameters = {
            'circuit_execution': {'circuit_name': 'medium', 'shots': 1000},
            'memory_efficiency': {'circuit_name': 'large'}
        }
        
        print("Running benchmark suite...")
        suite_results = suite.run_suite(test_functions, parameters)
        
        # Display results
        print(f"\nBenchmark Suite Results: {suite_results['suite_name']}")
        print(f"Execution time: {suite_results['suite_summary']['total_execution_time']:.3f}s")
        print(f"Success rate: {suite_results['suite_summary']['suite_success_rate']:.1%}")
        
        for bench_id, bench_result in suite_results['benchmark_results'].items():
            print(f"\n  {bench_result['benchmark_name']}:")
            summary = bench_result['summary']
            print(f"    Success rate: {summary['success_rate']:.1%}")
            print(f"    Avg execution time: {summary['avg_execution_time']:.3f}s")
            print(f"    Peak memory: {summary.get('peak_memory_usage_mb', 0):.1f}MB")
    
    async def demonstrate_scaling_analysis(self):
        """Demonstrate performance scaling analysis."""
        print("\n" + "="*60)
        print("PERFORMANCE SCALING ANALYSIS DEMONSTRATION")  
        print("="*60)
        
        scaling_benchmark = QuantumBenchmark(
            "qubit_scaling",
            "Qubit Count Scaling Analysis",
            "Analyzes how performance scales with qubit count"
        )
        
        def qubit_scaling_test(num_qubits: int):
            """Test function for qubit scaling analysis."""
            # Create circuit with specified qubit count
            circuit = QuantumCircuit(f"scaling_test_{num_qubits}", num_qubits=num_qubits)
            
            # Add gates proportional to qubit count
            for i in range(num_qubits):
                circuit.h(i)
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
            circuit.measure_all()
            
            # Execute with timing
            start_time = time.time()
            result = self.simulator.run(circuit, shots=100)
            execution_time = time.time() - start_time
            
            return {
                'success': result.success,
                'metrics': {
                    'execution_time': execution_time,
                    'memory_used': result.memory_used,
                    'gate_count': len(circuit.operations)
                }
            }
        
        # Run scaling study
        print("Analyzing performance scaling with qubit count...")
        qubit_values = [3, 5, 7, 9, 11, 13]
        
        scaling_analysis = scaling_benchmark.run_scaling_study(
            qubit_scaling_test,
            'num_qubits',
            qubit_values
        )
        
        # Display scaling analysis results
        print(f"\nScaling Analysis Results:")
        print(f"Parameter: {scaling_analysis.parameter_name}")
        print(f"Time complexity: {scaling_analysis.time_complexity}")
        print(f"Correlation coefficient: {scaling_analysis.correlation_coefficient:.3f}")
        
        if scaling_analysis.performance_cliff:
            print(f"Performance cliff at: {scaling_analysis.performance_cliff}")
        
        if scaling_analysis.optimal_range:
            print(f"Optimal range: {scaling_analysis.optimal_range[0]} - {scaling_analysis.optimal_range[1]}")
        
        if scaling_analysis.scaling_warnings:
            print("Scaling warnings:")
            for warning in scaling_analysis.scaling_warnings:
                print(f"  ⚠️  {warning}")
        
        # Show performance data
        print(f"\nPerformance Data:")
        for i, (qubits, time_taken) in enumerate(zip(qubit_values, scaling_analysis.execution_times)):
            memory = scaling_analysis.memory_usage[i] if i < len(scaling_analysis.memory_usage) else 0
            print(f"  {qubits} qubits: {time_taken:.3f}s, {memory:.1f}MB")
    
    async def demonstrate_report_generation(self):
        """Demonstrate comprehensive report generation."""
        print("\n" + "="*60)
        print("PERFORMANCE REPORT GENERATION DEMONSTRATION")
        print("="*60)
        
        # Collect some profile data for reporting
        circuits = self.create_test_circuits()
        profile_data_list = []
        
        print("Collecting profile data for report generation...")
        
        for i, (size, circuit) in enumerate(circuits.items()):
            profile_id = f"report_data_{i}"
            
            with profile_execution(profile_id, 
                                 circuit_name=circuit.name,
                                 backend_name="StateVectorSimulator",
                                 num_qubits=circuit.num_qubits,
                                 gate_count=len(circuit.operations)) as profile_data:
                
                result = self.simulator.run(circuit, shots=500)
                
                # Add some simulated timing data
                profile_data.compilation_time = 0.1 + circuit.num_qubits * 0.01
                profile_data.simulation_time = result.execution_time
                profile_data.peak_memory_mb = result.memory_used / (1024**2)
            
            # Get completed profile
            completed_profile = self.profiler.get_profile_history(limit=1)[0]
            profile_data_list.append(completed_profile)
        
        # Generate comprehensive performance report
        print("\nGenerating performance reports...")
        
        # Text format report
        text_report = self.report_generator.generate_performance_report(
            profile_data_list,
            title="Quantum Platform Performance Analysis",
            format=ReportFormat.TEXT
        )
        
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS REPORT (TEXT FORMAT)")
        print("="*60)
        print(text_report)
        
        # JSON format report
        json_report = self.report_generator.generate_performance_report(
            profile_data_list,
            title="Quantum Platform Performance Analysis",
            format=ReportFormat.JSON
        )
        
        # Save JSON report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"performance_report_{timestamp}.json"
        
        try:
            with open(json_filename, 'w') as f:
                f.write(json_report)
            print(f"\n✓ JSON report saved to: {json_filename}")
        except Exception as e:
            print(f"⚠️  Could not save JSON report: {e}")
        
        # Markdown format report
        markdown_report = self.report_generator.generate_performance_report(
            profile_data_list,
            title="Quantum Platform Performance Analysis",
            format=ReportFormat.MARKDOWN
        )
        
        markdown_filename = f"performance_report_{timestamp}.md"
        try:
            with open(markdown_filename, 'w') as f:
                f.write(markdown_report)
            print(f"✓ Markdown report saved to: {markdown_filename}")
        except Exception as e:
            print(f"⚠️  Could not save Markdown report: {e}")
    
    async def demonstrate_integration_with_monitoring(self):
        """Demonstrate integration with existing monitoring systems."""
        print("\n" + "="*60)
        print("INTEGRATION WITH EXECUTION MONITORING DEMONSTRATION")
        print("="*60)
        
        circuits = self.create_test_circuits()
        
        # Create a job that includes profiling
        from quantum_platform.execution.job_manager import ExecutionJob
        
        job = self.job_manager.create_job(
            job_type=JobType.SIMULATION,
            name="Profiled Simulation Job",
            circuit_name="medium_test",
            backend_name="StateVectorSimulator",
            shots=1000,
            metadata={'profiling_enabled': True}
        )
        
        print(f"Created job: {job.name} (ID: {job.job_id})")
        
        def execute_with_profiling(execution_job):
            """Execute job with integrated profiling."""
            circuit = circuits['medium']
            
            # Start profiling for this job
            with profile_execution(f"job_{execution_job.job_id}",
                                 circuit_name=circuit.name,
                                 job_id=execution_job.job_id) as profile_data:
                
                # Update job progress
                execution_job.update_progress(10, "Starting simulation")
                
                # Execute circuit
                execution_job.update_progress(50, "Running simulation")
                result = self.simulator.run(circuit, shots=1000)
                
                execution_job.update_progress(90, "Processing results")
                time.sleep(0.1)  # Simulate post-processing
                
                execution_job.update_progress(100, "Completed")
                
                return result
        
        # Submit and execute job
        success = self.job_manager.submit_job(job, execute_with_profiling)
        
        if success:
            print(f"✓ Job submitted successfully")
            
            # Wait for completion
            while job.is_active:
                print(f"  Progress: {job.progress:.1f}% - {job.metadata.get('status', 'Running')}")
                await asyncio.sleep(0.5)
            
            print(f"✓ Job completed: {job.status.value}")
            print(f"  Duration: {job.duration}")
            
            # Get profiling results
            recent_profiles = self.profiler.get_profile_history(limit=1)
            if recent_profiles:
                profile = recent_profiles[0]
                report = self.profiler.generate_report(profile)
                print(f"\nIntegrated Profiling Results:")
                print(f"  Total time: {profile.total_duration:.3f}s")
                print(f"  Peak memory: {profile.peak_memory_mb:.1f}MB")
                print(f"  Circuit: {profile.circuit_name}")
                print(f"  Backend: {profile.backend_name}")
        else:
            print("❌ Job submission failed")
    
    async def run_full_demonstration(self):
        """Run the complete performance profiling demonstration."""
        print("🚀 QUANTUM PLATFORM PERFORMANCE PROFILING & BENCHMARKING DEMO")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Run all demonstrations
            await self.demonstrate_basic_profiling()
            await self.demonstrate_simulation_profiling()
            await self.demonstrate_memory_profiling()
            await self.demonstrate_benchmarking()
            await self.demonstrate_scaling_analysis()
            await self.demonstrate_report_generation()
            await self.demonstrate_integration_with_monitoring()
            
            print("\n" + "="*80)
            print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
            print("="*80)
            
            # Final summary
            profile_history = self.profiler.get_profile_history()
            print(f"\nFinal Summary:")
            print(f"  Total profiles created: {len(profile_history)}")
            print(f"  Active profiles: {len(self.profiler.get_active_profiles())}")
            
            if profile_history:
                total_time = sum(p.total_duration for p in profile_history)
                avg_time = total_time / len(profile_history)
                print(f"  Total profiled time: {total_time:.3f}s")
                print(f"  Average execution time: {avg_time:.3f}s")
                
                max_memory = max(p.peak_memory_mb for p in profile_history if p.peak_memory_mb > 0)
                print(f"  Peak memory usage: {max_memory:.1f}MB")
        
        except Exception as e:
            print(f"\n❌ Demonstration failed: {e}")
            self.logger.error(f"Demo execution failed: {e}", exc_info=True)
        
        finally:
            # Cleanup
            self.job_manager.shutdown()
            print(f"\nDemo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


async def main():
    """Main demonstration entry point."""
    print("Initializing Quantum Platform Performance Profiling Demo...")
    
    try:
        demo = PerformanceProfilingDemo()
        await demo.run_full_demonstration()
    except Exception as e:
        print(f"Failed to run demonstration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    
    # Run the demonstration
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1) 