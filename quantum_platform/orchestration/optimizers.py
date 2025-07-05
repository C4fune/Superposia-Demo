"""
Classical Optimizers for Hybrid Algorithms

This module provides classical optimization algorithms that can be integrated
with quantum circuits in hybrid quantum-classical algorithms like VQE and QAOA.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from ..errors import OptimizationError
from ..observability import get_logger

logger = get_logger(__name__)


class OptimizationStatus(Enum):
    """Status of optimization process."""
    RUNNING = "running"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class OptimizerResult:
    """Result from optimization."""
    success: bool
    optimal_parameters: Dict[str, float]
    optimal_value: float
    iterations: int
    function_evaluations: int
    
    # Convergence information
    status: OptimizationStatus
    message: str = ""
    
    # Optimization history
    parameter_history: List[Dict[str, float]] = field(default_factory=list)
    value_history: List[float] = field(default_factory=list)
    
    # Timing information
    total_time: float = 0.0
    average_time_per_iteration: float = 0.0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptimizerCallback:
    """Base class for optimizer callbacks."""
    
    def __call__(self, iteration: int, parameters: Dict[str, float], 
                 value: float, **kwargs) -> bool:
        """
        Callback function called during optimization.
        
        Args:
            iteration: Current iteration number
            parameters: Current parameter values
            value: Current objective function value
            **kwargs: Additional information
            
        Returns:
            True to continue optimization, False to stop
        """
        return True


class ClassicalOptimizer(ABC):
    """Abstract base class for classical optimizers."""
    
    def __init__(self, name: str):
        self.name = name
        self.callback: Optional[OptimizerCallback] = None
        self.max_iterations = 1000
        self.tolerance = 1e-6
        self.initial_step_size = 0.1
        
    @abstractmethod
    def minimize(self, objective_function: Callable, 
                initial_parameters: Dict[str, float],
                bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                **kwargs) -> OptimizerResult:
        """
        Minimize the objective function.
        
        Args:
            objective_function: Function to minimize
            initial_parameters: Starting parameter values
            bounds: Parameter bounds (optional)
            **kwargs: Additional optimizer-specific arguments
            
        Returns:
            Optimization result
        """
        pass
    
    def set_callback(self, callback: OptimizerCallback):
        """Set optimization callback."""
        self.callback = callback
    
    def set_options(self, max_iterations: int = None, tolerance: float = None,
                   initial_step_size: float = None):
        """Set optimizer options."""
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if tolerance is not None:
            self.tolerance = tolerance
        if initial_step_size is not None:
            self.initial_step_size = initial_step_size


class ScipyOptimizer(ClassicalOptimizer):
    """Wrapper for SciPy optimizers."""
    
    def __init__(self, method: str = "COBYLA"):
        """
        Initialize SciPy optimizer.
        
        Args:
            method: SciPy optimization method
        """
        super().__init__(f"scipy_{method}")
        self.method = method
        self._scipy_available = self._check_scipy()
        
    def _check_scipy(self) -> bool:
        """Check if SciPy is available."""
        try:
            import scipy.optimize
            return True
        except ImportError:
            logger.warning("SciPy not available, using fallback optimizer")
            return False
    
    def minimize(self, objective_function: Callable, 
                initial_parameters: Dict[str, float],
                bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                **kwargs) -> OptimizerResult:
        """Minimize using SciPy."""
        if not self._scipy_available:
            # Fall back to simple gradient descent
            return self._fallback_minimize(objective_function, initial_parameters, bounds, **kwargs)
        
        import scipy.optimize
        
        start_time = time.time()
        
        # Convert parameters to array format
        param_names = list(initial_parameters.keys())
        x0 = np.array([initial_parameters[name] for name in param_names])
        
        # Convert bounds
        scipy_bounds = None
        if bounds:
            scipy_bounds = [bounds.get(name, (None, None)) for name in param_names]
        
        # Optimization history
        history = {
            'parameters': [],
            'values': [],
            'iterations': 0,
            'function_evaluations': 0
        }
        
        def objective_wrapper(x):
            """Wrapper for objective function."""
            params = {name: float(x[i]) for i, name in enumerate(param_names)}
            value = objective_function(params)
            
            # Store history
            history['parameters'].append(params.copy())
            history['values'].append(value)
            history['function_evaluations'] += 1
            
            # Call callback if available
            if self.callback:
                continue_opt = self.callback(
                    history['iterations'], 
                    params, 
                    value,
                    function_evaluations=history['function_evaluations']
                )
                if not continue_opt:
                    raise StopIteration("Optimization stopped by callback")
            
            return value
        
        def callback_wrapper(xk):
            """Callback wrapper for SciPy."""
            history['iterations'] += 1
        
        try:
            # Set up optimization options
            options = {
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'disp': False
            }
            options.update(kwargs.get('options', {}))
            
            # Run optimization
            result = scipy.optimize.minimize(
                objective_wrapper,
                x0,
                method=self.method,
                bounds=scipy_bounds,
                callback=callback_wrapper,
                options=options
            )
            
            # Convert result back to parameter dictionary
            optimal_params = {name: float(result.x[i]) for i, name in enumerate(param_names)}
            
            # Determine status
            if result.success:
                status = OptimizationStatus.CONVERGED
            elif history['iterations'] >= self.max_iterations:
                status = OptimizationStatus.MAX_ITERATIONS
            else:
                status = OptimizationStatus.FAILED
            
            total_time = time.time() - start_time
            
            return OptimizerResult(
                success=result.success,
                optimal_parameters=optimal_params,
                optimal_value=result.fun,
                iterations=history['iterations'],
                function_evaluations=history['function_evaluations'],
                status=status,
                message=result.message,
                parameter_history=history['parameters'],
                value_history=history['values'],
                total_time=total_time,
                average_time_per_iteration=total_time / max(1, history['iterations']),
                metadata={'scipy_result': result}
            )
            
        except StopIteration:
            # Optimization stopped by callback
            optimal_params = history['parameters'][-1] if history['parameters'] else initial_parameters
            optimal_value = history['values'][-1] if history['values'] else float('inf')
            total_time = time.time() - start_time
            
            return OptimizerResult(
                success=False,
                optimal_parameters=optimal_params,
                optimal_value=optimal_value,
                iterations=history['iterations'],
                function_evaluations=history['function_evaluations'],
                status=OptimizationStatus.STOPPED,
                message="Optimization stopped by callback",
                parameter_history=history['parameters'],
                value_history=history['values'],
                total_time=total_time,
                average_time_per_iteration=total_time / max(1, history['iterations'])
            )
        
        except Exception as e:
            logger.error(f"SciPy optimization failed: {e}")
            raise OptimizationError(f"Optimization failed: {e}")
    
    def _fallback_minimize(self, objective_function: Callable, 
                          initial_parameters: Dict[str, float],
                          bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                          **kwargs) -> OptimizerResult:
        """Fallback optimization using simple gradient descent."""
        logger.info("Using fallback gradient descent optimizer")
        
        gradient_descent = GradientDescentOptimizer()
        gradient_descent.set_options(
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            initial_step_size=self.initial_step_size
        )
        gradient_descent.set_callback(self.callback)
        
        return gradient_descent.minimize(objective_function, initial_parameters, bounds, **kwargs)


class GradientDescentOptimizer(ClassicalOptimizer):
    """Simple gradient descent optimizer."""
    
    def __init__(self):
        super().__init__("gradient_descent")
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.adaptive_learning_rate = True
        
    def minimize(self, objective_function: Callable, 
                initial_parameters: Dict[str, float],
                bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                **kwargs) -> OptimizerResult:
        """Minimize using gradient descent."""
        start_time = time.time()
        
        # Initialize parameters
        current_params = initial_parameters.copy()
        param_names = list(current_params.keys())
        
        # Initialize momentum
        momentum = {name: 0.0 for name in param_names}
        
        # Optimization history
        parameter_history = [current_params.copy()]
        value_history = []
        
        # Initial evaluation
        current_value = objective_function(current_params)
        value_history.append(current_value)
        best_value = current_value
        best_params = current_params.copy()
        
        # Adaptive learning rate
        learning_rate = self.learning_rate
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            # Call callback if available
            if self.callback:
                continue_opt = self.callback(
                    iteration, 
                    current_params, 
                    current_value,
                    learning_rate=learning_rate
                )
                if not continue_opt:
                    break
            
            # Compute numerical gradients
            gradients = self._compute_gradients(objective_function, current_params)
            
            # Update parameters with momentum
            for name in param_names:
                momentum[name] = self.momentum * momentum[name] + learning_rate * gradients[name]
                current_params[name] -= momentum[name]
                
                # Apply bounds if specified
                if bounds and name in bounds:
                    lower, upper = bounds[name]
                    if lower is not None:
                        current_params[name] = max(current_params[name], lower)
                    if upper is not None:
                        current_params[name] = min(current_params[name], upper)
            
            # Evaluate new parameters
            try:
                new_value = objective_function(current_params)
            except Exception as e:
                logger.warning(f"Objective function evaluation failed: {e}")
                break
            
            # Store history
            parameter_history.append(current_params.copy())
            value_history.append(new_value)
            
            # Update best result
            if new_value < best_value:
                best_value = new_value
                best_params = current_params.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Adaptive learning rate
            if self.adaptive_learning_rate and no_improvement_count > 5:
                learning_rate *= 0.8
                no_improvement_count = 0
            
            # Check convergence
            if len(value_history) > 1:
                improvement = abs(value_history[-2] - value_history[-1])
                if improvement < self.tolerance:
                    logger.info(f"Converged after {iteration + 1} iterations")
                    break
            
            current_value = new_value
        
        # Determine final status
        if len(value_history) > 1 and abs(value_history[-2] - value_history[-1]) < self.tolerance:
            status = OptimizationStatus.CONVERGED
            success = True
        elif iteration >= self.max_iterations - 1:
            status = OptimizationStatus.MAX_ITERATIONS
            success = False
        else:
            status = OptimizationStatus.STOPPED
            success = False
        
        total_time = time.time() - start_time
        
        return OptimizerResult(
            success=success,
            optimal_parameters=best_params,
            optimal_value=best_value,
            iterations=iteration + 1,
            function_evaluations=len(value_history),
            status=status,
            message=f"Optimization completed with status: {status.value}",
            parameter_history=parameter_history,
            value_history=value_history,
            total_time=total_time,
            average_time_per_iteration=total_time / max(1, iteration + 1)
        )
    
    def _compute_gradients(self, objective_function: Callable, 
                          parameters: Dict[str, float], 
                          epsilon: float = 1e-8) -> Dict[str, float]:
        """Compute numerical gradients."""
        gradients = {}
        
        for param_name, param_value in parameters.items():
            # Forward difference
            params_plus = parameters.copy()
            params_plus[param_name] = param_value + epsilon
            
            params_minus = parameters.copy()
            params_minus[param_name] = param_value - epsilon
            
            try:
                f_plus = objective_function(params_plus)
                f_minus = objective_function(params_minus)
                gradient = (f_plus - f_minus) / (2 * epsilon)
            except Exception as e:
                logger.warning(f"Gradient computation failed for {param_name}: {e}")
                gradient = 0.0
            
            gradients[param_name] = gradient
        
        return gradients


class CustomOptimizer(ClassicalOptimizer):
    """Custom optimizer that allows user-defined optimization logic."""
    
    def __init__(self, optimizer_function: Callable):
        """
        Initialize custom optimizer.
        
        Args:
            optimizer_function: User-defined optimization function
        """
        super().__init__("custom")
        self.optimizer_function = optimizer_function
    
    def minimize(self, objective_function: Callable, 
                initial_parameters: Dict[str, float],
                bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                **kwargs) -> OptimizerResult:
        """Minimize using custom optimizer function."""
        try:
            return self.optimizer_function(
                objective_function, 
                initial_parameters, 
                bounds,
                callback=self.callback,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Custom optimizer failed: {e}")
            raise OptimizationError(f"Custom optimization failed: {e}")


# Optimizer registry
_optimizers = {}


def register_optimizer(name: str, optimizer_class: type):
    """Register a custom optimizer."""
    _optimizers[name] = optimizer_class


def get_optimizer(name: str, **kwargs) -> ClassicalOptimizer:
    """Get an optimizer by name."""
    if name == "scipy":
        return ScipyOptimizer(**kwargs)
    elif name == "gradient_descent":
        return GradientDescentOptimizer()
    elif name in _optimizers:
        return _optimizers[name](**kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def list_available_optimizers() -> List[str]:
    """List all available optimizers."""
    return ["scipy", "gradient_descent"] + list(_optimizers.keys())


# Convenience functions
def minimize_with_scipy(objective_function: Callable, 
                       initial_parameters: Dict[str, float],
                       method: str = "COBYLA",
                       bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                       **kwargs) -> OptimizerResult:
    """Convenience function for SciPy optimization."""
    optimizer = ScipyOptimizer(method=method)
    return optimizer.minimize(objective_function, initial_parameters, bounds, **kwargs)


def minimize_with_gradient_descent(objective_function: Callable, 
                                  initial_parameters: Dict[str, float],
                                  bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                                  **kwargs) -> OptimizerResult:
    """Convenience function for gradient descent optimization."""
    optimizer = GradientDescentOptimizer()
    return optimizer.minimize(objective_function, initial_parameters, bounds, **kwargs) 