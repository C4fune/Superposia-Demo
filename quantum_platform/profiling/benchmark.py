"""
Quantum Benchmarking Suite

This module provides comprehensive benchmarking tools for quantum operations,
including performance scaling analysis and comparative studies.
"""

import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import numpy as np
import statistics

from quantum_platform.observability.logging import get_logger
from quantum_platform.profiling.profiler import get_profiler, ProfilerConfig, ProfilerMode


class BenchmarkType(Enum):
    """Types of benchmarks."""
    SCALING = "scaling"           # Performance vs parameter scaling
    COMPARISON = "comparison"     # Compare different implementations
    REGRESSION = "regression"     # Performance regression testing
    STRESS = "stress"            # Stress testing with high loads
    ACCURACY = "accuracy"        # Accuracy vs performance tradeoffs


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    benchmark_id: str
    run_id: str
    timestamp: datetime
    
    # Test parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    
    # Quality metrics
    success: bool = True
    error_rate: float = 0.0
    accuracy_score: Optional[float] = None
    
    # Resource metrics
    gates_per_second: Optional[float] = None
    shots_per_second: Optional[float] = None
    
    # Additional metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    backend_name: str = ""
    circuit_name: str = ""
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'benchmark_id': self.benchmark_id,
            'run_id': self.run_id,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters,
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_utilization': self.cpu_utilization,
            'success': self.success,
            'error_rate': self.error_rate,
            'accuracy_score': self.accuracy_score,
            'gates_per_second': self.gates_per_second,
            'shots_per_second': self.shots_per_second,
            'custom_metrics': self.custom_metrics,
            'backend_name': self.backend_name,
            'circuit_name': self.circuit_name,
            'error_message': self.error_message
        }


@dataclass
class ScalingAnalysis:
    """Analysis of performance scaling characteristics."""
    parameter_name: str
    parameter_values: List[Any] = field(default_factory=list)
    
    # Performance scaling
    execution_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    
    # Statistical analysis
    time_complexity: Optional[str] = None  # O(n), O(n^2), etc.
    memory_complexity: Optional[str] = None
    correlation_coefficient: float = 0.0
    
    # Scaling predictions
    predicted_scaling: Dict[str, float] = field(default_factory=dict)
    performance_cliff: Optional[Any] = None  # Parameter value where performance degrades
    
    # Recommendations
    optimal_range: Optional[Tuple[Any, Any]] = None
    scaling_warnings: List[str] = field(default_factory=list)
    
    def analyze_scaling(self):
        """Analyze scaling patterns and generate insights."""
        if len(self.parameter_values) < 3 or len(self.execution_times) < 3:
            return
        
        # Convert parameter values to numeric for analysis
        try:
            numeric_params = [float(p) for p in self.parameter_values]
            
            # Calculate correlation
            if len(numeric_params) == len(self.execution_times):
                self.correlation_coefficient = np.corrcoef(numeric_params, self.execution_times)[0, 1]
            
            # Detect scaling patterns
            self._detect_complexity_class(numeric_params, self.execution_times)
            self._find_performance_cliff(numeric_params, self.execution_times)
            self._generate_recommendations(numeric_params, self.execution_times)
            
        except (ValueError, TypeError):
            # Non-numeric parameters, skip analysis
            pass
    
    def _detect_complexity_class(self, params: List[float], times: List[float]):
        """Detect time complexity class."""
        if len(params) < 3:
            return
        
        # Test various complexity classes
        param_ratios = [params[i]/params[0] for i in range(1, len(params))]
        time_ratios = [times[i]/times[0] for i in range(1, len(times))]
        
        # Linear: O(n)
        linear_expected = param_ratios
        linear_error = sum(abs(a - b) for a, b in zip(time_ratios, linear_expected)) / len(time_ratios)
        
        # Quadratic: O(n^2)
        quad_expected = [r**2 for r in param_ratios]
        quad_error = sum(abs(a - b) for a, b in zip(time_ratios, quad_expected)) / len(time_ratios)
        
        # Exponential: O(2^n)
        exp_expected = [2**(r-1) for r in param_ratios]
        exp_error = sum(abs(a - b) for a, b in zip(time_ratios, exp_expected)) / len(time_ratios)
        
        # Choose best fit
        errors = [('O(n)', linear_error), ('O(n^2)', quad_error), ('O(2^n)', exp_error)]
        best_fit = min(errors, key=lambda x: x[1])
        
        if best_fit[1] < 0.5:  # Reasonable fit threshold
            self.time_complexity = best_fit[0]
    
    def _find_performance_cliff(self, params: List[float], times: List[float]):
        """Find parameter value where performance degrades significantly."""
        if len(params) < 4:
            return
        
        # Look for sudden increases in execution time
        for i in range(1, len(times)):
            if times[i] > times[i-1] * 2:  # 2x increase
                self.performance_cliff = params[i]
                self.scaling_warnings.append(
                    f"Performance cliff detected at {self.parameter_name}={params[i]}"
                )
                break
    
    def _generate_recommendations(self, params: List[float], times: List[float]):
        """Generate performance recommendations."""
        if len(params) < 3:
            return
        
        # Find optimal range (where performance is reasonable)
        reasonable_times = [t for t in times if t < max(times) * 0.1]  # Within 10% of fastest
        if reasonable_times:
            optimal_indices = [i for i, t in enumerate(times) if t in reasonable_times]
            if optimal_indices:
                self.optimal_range = (params[min(optimal_indices)], params[max(optimal_indices)])
        
        # Generate warnings
        if self.correlation_coefficient > 0.9:
            self.scaling_warnings.append("Strong positive correlation - performance degrades significantly with parameter increase")
        
        if times[-1] > times[0] * 10:
            self.scaling_warnings.append("Performance degrades by 10x+ across parameter range")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'parameter_name': self.parameter_name,
            'parameter_values': self.parameter_values,
            'execution_times': self.execution_times,
            'memory_usage': self.memory_usage,
            'time_complexity': self.time_complexity,
            'memory_complexity': self.memory_complexity,
            'correlation_coefficient': self.correlation_coefficient,
            'performance_cliff': self.performance_cliff,
            'optimal_range': self.optimal_range,
            'scaling_warnings': self.scaling_warnings
        }


class QuantumBenchmark:
    """
    Individual quantum benchmark for specific operations or circuits.
    """
    
    def __init__(self, benchmark_id: str, name: str, description: str = ""):
        """
        Initialize a quantum benchmark.
        
        Args:
            benchmark_id: Unique identifier
            name: Human-readable name
            description: Description of what is being benchmarked
        """
        self.benchmark_id = benchmark_id
        self.name = name
        self.description = description
        self.logger = get_logger(__name__)
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        self.scaling_analyses: Dict[str, ScalingAnalysis] = {}
        
        # Benchmark configuration
        self.warmup_runs = 1
        self.measurement_runs = 3
        self.timeout_seconds = 300
        
    def run_single(self, test_function: Callable[..., Any], 
                  parameters: Dict[str, Any], **kwargs) -> BenchmarkResult:
        """
        Run a single benchmark instance.
        
        Args:
            test_function: Function to benchmark
            parameters: Parameters to pass to the function
            **kwargs: Additional benchmark options
            
        Returns:
            BenchmarkResult with performance metrics
        """
        run_id = f"{self.benchmark_id}_{len(self.results)}"
        
        result = BenchmarkResult(
            benchmark_id=self.benchmark_id,
            run_id=run_id,
            timestamp=datetime.now(),
            parameters=parameters.copy(),
            backend_name=kwargs.get('backend_name', ''),
            circuit_name=kwargs.get('circuit_name', '')
        )
        
        # Setup profiling
        profiler = get_profiler()
        
        try:
            # Warmup runs
            for _ in range(self.warmup_runs):
                try:
                    test_function(**parameters)
                except Exception:
                    pass  # Ignore warmup failures
            
            # Measurement runs
            execution_times = []
            memory_usages = []
            
            for run in range(self.measurement_runs):
                profile_id = f"{run_id}_run_{run}"
                
                # Start profiling
                profile_data = profiler.start_profile(
                    profile_id, 
                    circuit_name=result.circuit_name,
                    backend_name=result.backend_name,
                    **parameters
                )
                
                # Execute benchmark
                start_time = time.time()
                try:
                    test_result = test_function(**parameters)
                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)
                    
                    # Extract custom metrics if returned
                    if isinstance(test_result, dict) and 'metrics' in test_result:
                        result.custom_metrics.update(test_result['metrics'])
                
                except Exception as e:
                    execution_time = time.time() - start_time
                    result.success = False
                    result.error_message = str(e)
                    self.logger.error(f"Benchmark run failed: {e}")
                
                # Stop profiling and collect metrics
                profile_result = profiler.stop_profile(profile_id)
                if profile_result:
                    memory_usages.append(profile_result.peak_memory_mb)
                    result.cpu_utilization = max(result.cpu_utilization, profile_result.avg_cpu_percent)
            
            # Aggregate results
            if execution_times:
                result.execution_time = statistics.median(execution_times)
            if memory_usages:
                result.memory_usage_mb = max(memory_usages)
            
            # Calculate derived metrics
            if 'shots' in parameters and result.execution_time > 0:
                result.shots_per_second = parameters['shots'] / result.execution_time
            
            if 'gate_count' in parameters and result.execution_time > 0:
                result.gates_per_second = parameters['gate_count'] / result.execution_time
        
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.logger.error(f"Benchmark execution failed: {e}")
        
        self.results.append(result)
        return result
    
    def run_scaling_study(self, test_function: Callable[..., Any], 
                         parameter_name: str, parameter_values: List[Any],
                         base_parameters: Optional[Dict[str, Any]] = None) -> ScalingAnalysis:
        """
        Run a scaling study varying a single parameter.
        
        Args:
            test_function: Function to benchmark
            parameter_name: Name of parameter to vary
            parameter_values: List of values for the parameter
            base_parameters: Base parameters for the test
            
        Returns:
            ScalingAnalysis with scaling characteristics
        """
        if base_parameters is None:
            base_parameters = {}
        
        analysis = ScalingAnalysis(parameter_name=parameter_name)
        
        for param_value in parameter_values:
            # Prepare parameters for this run
            run_parameters = base_parameters.copy()
            run_parameters[parameter_name] = param_value
            
            # Run benchmark
            result = self.run_single(test_function, run_parameters)
            
            if result.success:
                analysis.parameter_values.append(param_value)
                analysis.execution_times.append(result.execution_time)
                analysis.memory_usage.append(result.memory_usage_mb)
                
                self.logger.info(f"Scaling study {parameter_name}={param_value}: "
                               f"{result.execution_time:.3f}s, {result.memory_usage_mb:.1f}MB")
            else:
                self.logger.warning(f"Scaling study failed at {parameter_name}={param_value}: "
                                  f"{result.error_message}")
        
        # Analyze scaling patterns
        analysis.analyze_scaling()
        self.scaling_analyses[parameter_name] = analysis
        
        return analysis
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all benchmark runs."""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            return {'success_rate': 0.0, 'total_runs': len(self.results)}
        
        execution_times = [r.execution_time for r in successful_results]
        memory_usages = [r.memory_usage_mb for r in successful_results]
        
        return {
            'total_runs': len(self.results),
            'successful_runs': len(successful_results),
            'success_rate': len(successful_results) / len(self.results),
            'avg_execution_time': statistics.mean(execution_times),
            'median_execution_time': statistics.median(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'std_execution_time': statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0,
            'avg_memory_usage_mb': statistics.mean(memory_usages),
            'peak_memory_usage_mb': max(memory_usages)
        }


class BenchmarkSuite:
    """
    Collection of related benchmarks for comprehensive performance analysis.
    """
    
    def __init__(self, suite_name: str, description: str = ""):
        """
        Initialize benchmark suite.
        
        Args:
            suite_name: Name of the benchmark suite
            description: Description of the suite
        """
        self.suite_name = suite_name
        self.description = description
        self.logger = get_logger(__name__)
        
        # Benchmark collection
        self.benchmarks: Dict[str, QuantumBenchmark] = {}
        self.suite_results: List[Dict[str, Any]] = []
        
        # Suite configuration
        self.parallel_execution = False
        self.continue_on_failure = True
        
    def add_benchmark(self, benchmark: QuantumBenchmark):
        """Add a benchmark to the suite."""
        self.benchmarks[benchmark.benchmark_id] = benchmark
        self.logger.info(f"Added benchmark '{benchmark.name}' to suite '{self.suite_name}'")
    
    def run_suite(self, test_functions: Dict[str, Callable], 
                 parameters: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run all benchmarks in the suite.
        
        Args:
            test_functions: Dictionary mapping benchmark IDs to test functions
            parameters: Dictionary mapping benchmark IDs to their parameters
            
        Returns:
            Suite execution results
        """
        suite_start_time = time.time()
        suite_results = {
            'suite_name': self.suite_name,
            'start_time': datetime.now().isoformat(),
            'benchmark_results': {},
            'suite_summary': {}
        }
        
        self.logger.info(f"Starting benchmark suite: {self.suite_name}")
        
        for benchmark_id, benchmark in self.benchmarks.items():
            if benchmark_id not in test_functions:
                self.logger.warning(f"No test function provided for benchmark {benchmark_id}")
                continue
            
            try:
                benchmark_params = parameters.get(benchmark_id, {})
                test_function = test_functions[benchmark_id]
                
                self.logger.info(f"Running benchmark: {benchmark.name}")
                result = benchmark.run_single(test_function, benchmark_params)
                
                suite_results['benchmark_results'][benchmark_id] = {
                    'benchmark_name': benchmark.name,
                    'result': result.to_dict(),
                    'summary': benchmark.get_summary_statistics()
                }
                
                if not result.success and not self.continue_on_failure:
                    self.logger.error(f"Benchmark {benchmark_id} failed, stopping suite")
                    break
                    
            except Exception as e:
                self.logger.error(f"Failed to run benchmark {benchmark_id}: {e}")
                if not self.continue_on_failure:
                    break
        
        # Calculate suite summary
        suite_execution_time = time.time() - suite_start_time
        suite_results['suite_summary'] = self._calculate_suite_summary(suite_execution_time)
        suite_results['end_time'] = datetime.now().isoformat()
        
        self.suite_results.append(suite_results)
        self.logger.info(f"Completed benchmark suite: {self.suite_name} in {suite_execution_time:.2f}s")
        
        return suite_results
    
    def _calculate_suite_summary(self, execution_time: float) -> Dict[str, Any]:
        """Calculate summary statistics for the entire suite."""
        total_benchmarks = len(self.benchmarks)
        successful_benchmarks = sum(
            1 for b in self.benchmarks.values() 
            if b.results and any(r.success for r in b.results)
        )
        
        return {
            'total_benchmarks': total_benchmarks,
            'successful_benchmarks': successful_benchmarks,
            'suite_success_rate': successful_benchmarks / total_benchmarks if total_benchmarks > 0 else 0.0,
            'total_execution_time': execution_time,
            'avg_benchmark_time': execution_time / total_benchmarks if total_benchmarks > 0 else 0.0
        } 