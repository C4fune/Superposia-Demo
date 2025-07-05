"""
Hybrid Quantum-Classical Executor

This module provides the core execution framework for hybrid quantum-classical algorithms,
including parameter binding, synchronous execution, and result management.
"""

import threading
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from enum import Enum
import hashlib
import json
import sympy

from ..compiler.ir.circuit import QuantumCircuit
from ..compiler.ir.types import Parameter, ParameterDict, ParameterValue
from ..hardware.hal import QuantumHardwareBackend, HardwareResult
from ..hardware.backends import LocalSimulatorBackend
from ..errors import ExecutionError, ParameterError, handle_errors
from ..observability import get_logger

logger = get_logger(__name__)


class ExecutionMode(Enum):
    """Execution modes for hybrid algorithms."""
    SYNCHRONOUS = "synchronous"        # Block until results are available
    ASYNCHRONOUS = "asynchronous"      # Return immediately with job handle
    BATCH = "batch"                    # Execute multiple parameter sets at once
    STREAMING = "streaming"            # Continuous execution with callbacks


@dataclass
class ExecutionContext:
    """Context for hybrid algorithm execution."""
    backend: QuantumHardwareBackend
    shots: int = 1000
    timeout: Optional[float] = None
    mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    
    # Caching options
    enable_caching: bool = True
    cache_threshold: float = 1e-6  # Parameter difference threshold for cache hits
    
    # Monitoring options
    enable_monitoring: bool = True
    callback_interval: int = 1  # Call progress callback every N executions
    
    # Error handling
    retry_count: int = 3
    backoff_factor: float = 2.0
    
    # Provider switching
    fallback_backend: Optional[QuantumHardwareBackend] = None
    auto_switch_on_error: bool = True


@dataclass
class HybridResult:
    """Result from hybrid algorithm execution."""
    execution_id: str
    parameters: Dict[str, float]
    expectation_value: float
    raw_result: HardwareResult
    execution_time: float
    
    # Metadata
    iteration: int = 0
    backend_used: str = ""
    cache_hit: bool = False
    shots_used: int = 0
    
    # Error information
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Additional metrics
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParameterBinder:
    """Handles parameter binding for quantum circuits."""
    
    def __init__(self, circuit: QuantumCircuit):
        """
        Initialize parameter binder.
        
        Args:
            circuit: Quantum circuit with symbolic parameters
        """
        self.circuit = circuit
        self.parameter_names = self._extract_parameter_names()
        self._cached_circuits: Dict[str, QuantumCircuit] = {}
        self._cache_lock = threading.Lock()
        
    def _extract_parameter_names(self) -> List[str]:
        """Extract parameter names from circuit."""
        parameters = set()
        for operation in self.circuit.operations:
            if hasattr(operation, 'params') and operation.params:
                for param_name, param_value in operation.params.items():
                    if isinstance(param_value, ParameterValue) and param_value.is_symbolic:
                        # Extract parameter name from sympy symbol
                        if isinstance(param_value.value, sympy.Symbol):
                            parameters.add(str(param_value.value))
                        elif hasattr(param_value.value, 'free_symbols'):
                            # For more complex expressions, get all symbols
                            for symbol in param_value.value.free_symbols:
                                parameters.add(str(symbol))
                    elif isinstance(param_value, str):
                        parameters.add(param_value)
        return sorted(list(parameters))
    
    def _parameter_hash(self, parameters: Dict[str, float]) -> str:
        """Generate hash for parameter set."""
        param_str = json.dumps(parameters, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def bind_parameters(self, parameters: Dict[str, float], 
                       use_cache: bool = True) -> QuantumCircuit:
        """
        Bind parameters to circuit.
        
        Args:
            parameters: Dictionary of parameter values
            use_cache: Whether to use cached circuits
            
        Returns:
            Circuit with bound parameters
        """
        if use_cache:
            param_hash = self._parameter_hash(parameters)
            with self._cache_lock:
                if param_hash in self._cached_circuits:
                    return self._cached_circuits[param_hash]
        
        # Validate parameters
        missing_params = set(self.parameter_names) - set(parameters.keys())
        if missing_params:
            raise ParameterError(
                f"Missing parameters: {missing_params}",
                user_message=f"Please provide values for parameters: {missing_params}"
            )
        
        # Create substitution dictionary with sympy symbols
        substitutions = {}
        for param_name, param_value in parameters.items():
            symbol = sympy.Symbol(param_name, real=True)
            substitutions[symbol] = param_value
        
        # Create bound circuit
        bound_circuit = self.circuit.substitute_parameters(substitutions)
        
        # Cache if enabled
        if use_cache:
            with self._cache_lock:
                self._cached_circuits[param_hash] = bound_circuit
        
        return bound_circuit
    
    def clear_cache(self):
        """Clear parameter binding cache."""
        with self._cache_lock:
            self._cached_circuits.clear()


class OptimizationLoop:
    """High-level optimization loop for hybrid algorithms."""
    
    def __init__(self, executor: 'HybridExecutor', objective_function: Callable):
        """
        Initialize optimization loop.
        
        Args:
            executor: Hybrid executor instance
            objective_function: Function to minimize (takes parameters, returns float)
        """
        self.executor = executor
        self.objective_function = objective_function
        self.iteration = 0
        self.best_result = None
        self.history = []
        self._stop_requested = False
        
    def optimize(self, initial_parameters: Dict[str, float], 
                max_iterations: int = 100,
                tolerance: float = 1e-6,
                callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run optimization loop.
        
        Args:
            initial_parameters: Starting parameter values
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            callback: Optional callback function for progress monitoring
            
        Returns:
            Optimization result dictionary
        """
        self.iteration = 0
        self.best_result = None
        self.history = []
        self._stop_requested = False
        
        current_params = initial_parameters.copy()
        
        logger.info(f"Starting optimization with {len(current_params)} parameters")
        
        for iteration in range(max_iterations):
            if self._stop_requested:
                logger.info("Optimization stopped by user request")
                break
                
            self.iteration = iteration
            
            # Evaluate objective function
            try:
                result = self.objective_function(current_params)
                
                # Track best result
                if self.best_result is None or result < self.best_result['value']:
                    self.best_result = {
                        'parameters': current_params.copy(),
                        'value': result,
                        'iteration': iteration
                    }
                
                # Add to history
                self.history.append({
                    'iteration': iteration,
                    'parameters': current_params.copy(),
                    'value': result,
                    'timestamp': datetime.now()
                })
                
                logger.debug(f"Iteration {iteration}: value={result:.6f}")
                
                # Check convergence
                if len(self.history) > 1:
                    prev_value = self.history[-2]['value']
                    if abs(result - prev_value) < tolerance:
                        logger.info(f"Converged at iteration {iteration}")
                        break
                
                # Progress callback
                if callback:
                    callback(iteration, current_params, result)
                
                # Simple gradient descent update (placeholder)
                # In practice, this would be replaced with sophisticated optimizer
                current_params = self._update_parameters(current_params, result)
                
            except Exception as e:
                logger.error(f"Error in optimization iteration {iteration}: {e}")
                break
        
        return {
            'best_parameters': self.best_result['parameters'] if self.best_result else initial_parameters,
            'best_value': self.best_result['value'] if self.best_result else float('inf'),
            'iterations': self.iteration,
            'converged': self.iteration < max_iterations - 1,
            'history': self.history
        }
    
    def _update_parameters(self, params: Dict[str, float], current_value: float) -> Dict[str, float]:
        """Simple parameter update (placeholder for real optimizer)."""
        # This is a very basic implementation
        # In practice, you'd use scipy.optimize or other optimizers
        new_params = params.copy()
        for key in new_params:
            # Small random perturbation
            new_params[key] += np.random.normal(0, 0.1)
        return new_params
    
    def stop(self):
        """Request optimization to stop."""
        self._stop_requested = True


class HybridExecutor:
    """Main executor for hybrid quantum-classical algorithms."""
    
    def __init__(self, context: ExecutionContext):
        """
        Initialize hybrid executor.
        
        Args:
            context: Execution context with backend and options
        """
        self.context = context
        self.parameter_binder = None
        self.execution_count = 0
        self.total_time = 0.0
        self._result_cache: Dict[str, HybridResult] = {}
        self._cache_lock = threading.Lock()
        
        logger.info(f"Initialized HybridExecutor with backend: {context.backend.name}")
    
    def set_circuit(self, circuit: QuantumCircuit):
        """Set the quantum circuit to execute."""
        self.parameter_binder = ParameterBinder(circuit)
        logger.info(f"Circuit set with {len(self.parameter_binder.parameter_names)} parameters")
    
    def execute(self, parameters: Dict[str, float], 
                expectation_operator: Optional[Callable] = None) -> HybridResult:
        """
        Execute quantum circuit with given parameters.
        
        Args:
            parameters: Dictionary of parameter values
            expectation_operator: Optional function to compute expectation value
            
        Returns:
            Hybrid execution result
        """
        start_time = time.time()
        execution_id = f"hybrid_{int(time.time() * 1000)}"
        
        logger.info(f"Executing circuit with {len(parameters)} parameters")
        
        try:
            # Check if we have a circuit set
            if self.parameter_binder is None:
                raise ExecutionError(
                    "No circuit set for execution",
                    user_message="Please set a circuit before executing"
                )
            
            # Check cache first
            if self.context.enable_caching:
                cache_key = self._get_cache_key(parameters)
                with self._cache_lock:
                    if cache_key in self._result_cache:
                        cached_result = self._result_cache[cache_key]
                        cached_result.cache_hit = True
                        logger.debug(f"Cache hit for parameters: {parameters}")
                        return cached_result
            
            # Bind parameters to circuit
            bound_circuit = self.parameter_binder.bind_parameters(parameters)
            
            # Execute circuit with retry logic
            raw_result = None
            retry_count = 0
            last_error = None
            
            while retry_count <= self.context.retry_count:
                try:
                    # Try primary backend
                    raw_result = self.context.backend.submit_and_wait(
                        bound_circuit,
                        shots=self.context.shots,
                        timeout=self.context.timeout
                    )
                    break
                        
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    
                    logger.warning(f"Execution attempt {retry_count} failed: {e}")
                    
                    if retry_count <= self.context.retry_count:
                        # Try fallback backend if available
                        if self.context.fallback_backend and self.context.auto_switch_on_error:
                            try:
                                raw_result = self.context.fallback_backend.submit_and_wait(
                                    bound_circuit,
                                    shots=self.context.shots,
                                    timeout=self.context.timeout
                                )
                                logger.info("Fallback backend execution succeeded")
                                break
                            except Exception as fallback_error:
                                logger.warning(f"Fallback backend also failed: {fallback_error}")
                        
                        # Wait before retry
                        if retry_count < self.context.retry_count:
                            wait_time = self.context.backoff_factor ** retry_count
                            logger.info(f"Retrying in {wait_time:.2f} seconds...")
                            time.sleep(wait_time)
            
            # Check if we got a result
            if raw_result is None:
                raise ExecutionError(
                    f"Execution failed after {self.context.retry_count} retries",
                    user_message=f"Circuit execution failed: {last_error}"
                )
            
            # Compute expectation value
            if expectation_operator:
                expectation_value = expectation_operator(raw_result.counts)
            else:
                expectation_value = self._compute_default_expectation(raw_result.counts)
            
            # Create result
            execution_time = time.time() - start_time
            result = HybridResult(
                execution_id=execution_id,
                parameters=parameters,
                expectation_value=expectation_value,
                raw_result=raw_result,
                execution_time=execution_time,
                backend_used=self.context.backend.name,
                shots_used=self.context.shots,
                retry_count=retry_count
            )
            
            # Cache result
            if self.context.enable_caching:
                cache_key = self._get_cache_key(parameters)
                with self._cache_lock:
                    self._result_cache[cache_key] = result
            
            # Update statistics
            self.execution_count += 1
            self.total_time += execution_time
            
            logger.info(f"Execution completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            # Create error result
            execution_time = time.time() - start_time
            result = HybridResult(
                execution_id=execution_id,
                parameters=parameters,
                expectation_value=0.0,
                raw_result=None,
                execution_time=execution_time,
                backend_used=self.context.backend.name,
                shots_used=self.context.shots,
                error_message=str(e),
                retry_count=retry_count
            )
            
            # Re-raise the exception for the caller to handle
            raise ExecutionError(
                f"Hybrid execution failed: {e}",
                user_message=f"Circuit execution failed: {e}"
            ) from e
    
    def _get_cache_key(self, parameters: Dict[str, float]) -> str:
        """Generate cache key for parameters."""
        # Round parameters to avoid floating point precision issues
        rounded_params = {k: round(v, 10) for k, v in parameters.items()}
        param_str = json.dumps(rounded_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _compute_default_expectation(self, counts: Dict[str, int]) -> float:
        """Compute default expectation value from measurement counts."""
        if not counts:
            return 0.0
        
        total_counts = sum(counts.values())
        expectation = 0.0
        
        for bitstring, count in counts.items():
            # Simple parity expectation: +1 for even parity, -1 for odd parity
            parity = sum(int(bit) for bit in bitstring) % 2
            expectation += ((-1) ** parity) * count / total_counts
        
        return expectation
    
    def create_optimization_loop(self, objective_function: Callable) -> OptimizationLoop:
        """Create an optimization loop using this executor."""
        return OptimizationLoop(self, objective_function)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'execution_count': self.execution_count,
            'total_time': self.total_time,
            'average_time': self.total_time / max(1, self.execution_count),
            'cache_size': len(self._result_cache),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        # This is a simplified calculation
        # In practice, you'd track cache hits more carefully
        return 0.0 if self.execution_count == 0 else len(self._result_cache) / self.execution_count
    
    def clear_cache(self):
        """Clear result cache."""
        with self._cache_lock:
            self._result_cache.clear()
        if self.parameter_binder:
            self.parameter_binder.clear_cache()


# Global executor instance
_hybrid_executor = None


def get_hybrid_executor(context: Optional[ExecutionContext] = None) -> HybridExecutor:
    """Get or create hybrid executor instance."""
    global _hybrid_executor
    
    if _hybrid_executor is None or context is not None:
        if context is None:
            # Create default context with local simulator
            context = ExecutionContext(
                backend=LocalSimulatorBackend("local_simulator", "local"),
                shots=1000
            )
        _hybrid_executor = HybridExecutor(context)
    
    return _hybrid_executor


def execute_hybrid_algorithm(circuit: QuantumCircuit, 
                           parameters: Dict[str, float],
                           backend: Optional[QuantumHardwareBackend] = None,
                           shots: int = 1000,
                           expectation_operator: Optional[Callable] = None) -> HybridResult:
    """
    Convenience function for single hybrid execution.
    
    Args:
        circuit: Quantum circuit with parameters
        parameters: Parameter values
        backend: Backend to use (defaults to local simulator)
        shots: Number of shots
        expectation_operator: Function to compute expectation value
        
    Returns:
        Hybrid execution result
    """
    if backend is None:
        backend = LocalSimulatorBackend("local_simulator", "local")
        backend.initialize()
    
    context = ExecutionContext(backend=backend, shots=shots)
    executor = HybridExecutor(context)
    executor.set_circuit(circuit)
    
    return executor.execute(parameters, expectation_operator) 