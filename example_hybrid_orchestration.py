"""
Hybrid Quantum-Classical Orchestration Example

This example demonstrates the comprehensive hybrid quantum-classical orchestration
capabilities, including VQE algorithm implementation, multi-provider support,
and parameter optimization workflows.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any

# Core quantum platform imports
from quantum_platform.compiler.ir.circuit import QuantumCircuit
from quantum_platform.compiler.ir.qubit import Qubit
from quantum_platform.compiler.language.operations import H, RX, RY, RZ, CNOT, measure
from quantum_platform.compiler.language.dsl import QuantumProgram
from quantum_platform.hardware.backends import LocalSimulatorBackend
from quantum_platform.orchestration.hybrid_executor import (
    HybridExecutor, ExecutionContext, ExecutionMode, execute_hybrid_algorithm
)
from quantum_platform.orchestration.optimizers import (
    ScipyOptimizer, GradientDescentOptimizer, OptimizerCallback
)
from quantum_platform.providers.provider_manager import (
    get_provider_manager, switch_provider, get_active_backend,
    list_available_providers, list_available_devices
)
from quantum_platform.observability import get_logger

logger = get_logger(__name__)


class VQEExample:
    """Variational Quantum Eigensolver (VQE) example implementation."""
    
    def __init__(self, num_qubits: int = 2):
        """Initialize VQE example with specified number of qubits."""
        self.num_qubits = num_qubits
        self.circuit = None
        self.target_hamiltonian = self._create_target_hamiltonian()
        self.executor = None
        
    def _create_target_hamiltonian(self) -> np.ndarray:
        """Create a target Hamiltonian for VQE (H2 molecule example)."""
        # Simple 2-qubit Hamiltonian for H2 molecule
        # In practice, this would be computed from molecular data
        if self.num_qubits == 2:
            # H2 Hamiltonian coefficients
            return np.array([
                [-1.0523732, 0.0, 0.0, 0.0],
                [0.0, -0.4804532, -0.6727435, 0.0],
                [0.0, -0.6727435, -0.4804532, 0.0],
                [0.0, 0.0, 0.0, -1.0523732]
            ])
        else:
            # Random Hamiltonian for demonstration
            H = np.random.random((2**self.num_qubits, 2**self.num_qubits))
            return (H + H.T) / 2  # Make it Hermitian
    
    def create_ansatz_circuit(self) -> QuantumCircuit:
        """Create parameterized ansatz circuit for VQE."""
        with QuantumProgram() as qp:
            # Allocate qubits
            qubits = [qp.allocate_qubit(f"q{i}") for i in range(self.num_qubits)]
            
            # Create parameterized ansatz (Hardware Efficient Ansatz)
            for layer in range(2):  # 2 layers
                # Single-qubit rotations
                for i, qubit in enumerate(qubits):
                    RY(qubit, f"theta_{layer}_{i}_y")
                    RZ(qubit, f"phi_{layer}_{i}_z")
                
                # Entangling gates
                for i in range(self.num_qubits - 1):
                    CNOT(qubits[i], qubits[i + 1])
                
                # Ring connectivity for more than 2 qubits
                if self.num_qubits > 2:
                    CNOT(qubits[-1], qubits[0])
            
            # Measurements
            for qubit in qubits:
                measure(qubit)
        
        self.circuit = qp.circuit
        return self.circuit
    
    def expectation_value_operator(self, counts: Dict[str, int]) -> float:
        """Compute expectation value of Hamiltonian from measurement counts."""
        if not counts:
            return 0.0
        
        total_shots = sum(counts.values())
        expectation = 0.0
        
        # Convert measurement counts to expectation value
        for bitstring, count in counts.items():
            # Convert bitstring to integer
            state_index = int(bitstring, 2)
            
            # Compute expectation value contribution
            # This is a simplified calculation - in practice, you'd need
            # to measure different Pauli operators and combine them
            prob = count / total_shots
            
            # Simple energy calculation (placeholder)
            if state_index < len(self.target_hamiltonian):
                energy = self.target_hamiltonian[state_index, state_index]
                expectation += prob * energy
        
        return expectation
    
    def run_vqe(self, backend_name: str = "local", max_iterations: int = 50) -> Dict[str, Any]:
        """Run VQE algorithm."""
        logger.info(f"Starting VQE with {self.num_qubits} qubits on {backend_name}")
        
        # Switch to specified provider
        provider_manager = get_provider_manager()
        if backend_name != "local":
            provider_manager.set_active_provider(backend_name)
        
        # Get active backend
        backend = get_active_backend()
        if not backend:
            raise RuntimeError(f"Could not get backend for {backend_name}")
        
        # Create execution context
        context = ExecutionContext(
            backend=backend,
            shots=1000,
            mode=ExecutionMode.SYNCHRONOUS,
            enable_caching=True,
            enable_monitoring=True
        )
        
        # Create hybrid executor
        self.executor = HybridExecutor(context)
        self.executor.set_circuit(self.circuit)
        
        # Create initial parameters
        initial_params = {}
        for layer in range(2):
            for i in range(self.num_qubits):
                initial_params[f"theta_{layer}_{i}_y"] = np.random.uniform(0, 2*np.pi)
                initial_params[f"phi_{layer}_{i}_z"] = np.random.uniform(0, 2*np.pi)
        
        logger.info(f"Initial parameters: {initial_params}")
        
        # Define objective function
        def objective_function(params: Dict[str, float]) -> float:
            """Objective function for VQE optimization."""
            try:
                result = self.executor.execute(params, self.expectation_value_operator)
                return result.expectation_value
            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return float('inf')
        
        # Create progress callback
        class VQECallback(OptimizerCallback):
            def __init__(self):
                self.best_value = float('inf')
                self.iteration_count = 0
            
            def __call__(self, iteration: int, parameters: Dict[str, float], 
                        value: float, **kwargs) -> bool:
                self.iteration_count += 1
                if value < self.best_value:
                    self.best_value = value
                    logger.info(f"Iteration {iteration}: New best energy = {value:.6f}")
                return True  # Continue optimization
        
        callback = VQECallback()
        
        # Choose optimizer
        optimizer = ScipyOptimizer(method="COBYLA")
        optimizer.set_callback(callback)
        optimizer.set_options(max_iterations=max_iterations, tolerance=1e-6)
        
        # Run optimization
        start_time = time.time()
        result = optimizer.minimize(objective_function, initial_params)
        optimization_time = time.time() - start_time
        
        # Get executor statistics
        executor_stats = self.executor.get_statistics()
        
        logger.info(f"VQE completed in {optimization_time:.2f}s")
        logger.info(f"Optimal energy: {result.optimal_value:.6f}")
        logger.info(f"Convergence: {result.status.value}")
        
        return {
            'optimal_energy': result.optimal_value,
            'optimal_parameters': result.optimal_parameters,
            'convergence_status': result.status.value,
            'iterations': result.iterations,
            'function_evaluations': result.function_evaluations,
            'optimization_time': optimization_time,
            'executor_statistics': executor_stats,
            'backend_used': backend_name,
            'parameter_history': result.parameter_history,
            'energy_history': result.value_history
        }


class QAOAExample:
    """Quantum Approximate Optimization Algorithm (QAOA) example."""
    
    def __init__(self, problem_size: int = 4):
        """Initialize QAOA example."""
        self.problem_size = problem_size
        self.circuit = None
        self.cost_function = self._create_maxcut_problem()
        
    def _create_maxcut_problem(self) -> np.ndarray:
        """Create a Max-Cut problem instance."""
        # Random graph adjacency matrix
        np.random.seed(42)  # For reproducibility
        adj_matrix = np.random.choice([0, 1], size=(self.problem_size, self.problem_size), p=[0.7, 0.3])
        # Make symmetric and remove self-loops
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix
    
    def create_qaoa_circuit(self, layers: int = 2) -> QuantumCircuit:
        """Create QAOA circuit."""
        with QuantumProgram() as qp:
            # Allocate qubits
            qubits = [qp.allocate_qubit(f"q{i}") for i in range(self.problem_size)]
            
            # Initial superposition
            for qubit in qubits:
                H(qubit)
            
            # QAOA layers
            for layer in range(layers):
                # Cost unitary (problem-specific)
                for i in range(self.problem_size):
                    for j in range(i + 1, self.problem_size):
                        if self.cost_function[i, j] != 0:
                            # Add ZZ interaction
                            CNOT(qubits[i], qubits[j])
                            RZ(qubits[j], f"gamma_{layer}")
                            CNOT(qubits[i], qubits[j])
                
                # Mixer unitary
                for i, qubit in enumerate(qubits):
                    RX(qubit, f"beta_{layer}")
            
            # Measurements
            for qubit in qubits:
                measure(qubit)
        
        self.circuit = qp.circuit
        return self.circuit
    
    def cost_expectation_operator(self, counts: Dict[str, int]) -> float:
        """Compute cost function expectation value."""
        if not counts:
            return 0.0
        
        total_shots = sum(counts.values())
        expectation = 0.0
        
        for bitstring, count in counts.items():
            # Convert bitstring to list of binary values
            assignment = [int(bit) for bit in bitstring]
            
            # Compute cost for this assignment
            cost = 0.0
            for i in range(self.problem_size):
                for j in range(i + 1, self.problem_size):
                    if self.cost_function[i, j] != 0:
                        # Max-Cut cost: +1 for edges between different partitions
                        if assignment[i] != assignment[j]:
                            cost += self.cost_function[i, j]
            
            prob = count / total_shots
            expectation += prob * cost
        
        return expectation
    
    def run_qaoa(self, backend_name: str = "local", layers: int = 2, 
                max_iterations: int = 100) -> Dict[str, Any]:
        """Run QAOA algorithm."""
        logger.info(f"Starting QAOA with {self.problem_size} qubits, {layers} layers")
        
        # Create circuit
        self.create_qaoa_circuit(layers)
        
        # Switch provider
        provider_manager = get_provider_manager()
        if backend_name != "local":
            provider_manager.set_active_provider(backend_name)
        
        backend = get_active_backend()
        context = ExecutionContext(
            backend=backend,
            shots=2000,  # More shots for better statistics
            enable_caching=True
        )
        
        executor = HybridExecutor(context)
        executor.set_circuit(self.circuit)
        
        # Initial parameters
        initial_params = {}
        for layer in range(layers):
            initial_params[f"gamma_{layer}"] = np.random.uniform(0, np.pi)
            initial_params[f"beta_{layer}"] = np.random.uniform(0, np.pi/2)
        
        # Objective function (minimize negative cost for maximization)
        def objective_function(params: Dict[str, float]) -> float:
            result = executor.execute(params, self.cost_expectation_operator)
            return -result.expectation_value  # Minimize negative for maximization
        
        # Optimize
        optimizer = ScipyOptimizer(method="COBYLA")
        optimizer.set_options(max_iterations=max_iterations)
        
        start_time = time.time()
        result = optimizer.minimize(objective_function, initial_params)
        optimization_time = time.time() - start_time
        
        return {
            'optimal_cost': -result.optimal_value,  # Convert back to positive
            'optimal_parameters': result.optimal_parameters,
            'convergence_status': result.status.value,
            'iterations': result.iterations,
            'optimization_time': optimization_time,
            'layers': layers,
            'problem_size': self.problem_size
        }


class ProviderComparisonExample:
    """Example comparing different quantum providers."""
    
    def __init__(self):
        self.provider_manager = get_provider_manager()
        
    def compare_providers(self, circuit: QuantumCircuit, 
                         parameters: Dict[str, float],
                         providers: List[str] = None) -> Dict[str, Any]:
        """Compare execution across different providers."""
        if providers is None:
            providers = ["local"]  # Default to local for demo
        
        results = {}
        
        for provider_name in providers:
            try:
                logger.info(f"Testing provider: {provider_name}")
                
                # Switch to provider
                if provider_name != "local":
                    self.provider_manager.set_active_provider(provider_name)
                
                backend = get_active_backend()
                if not backend:
                    logger.warning(f"Could not get backend for {provider_name}")
                    continue
                
                # Execute circuit
                start_time = time.time()
                result = execute_hybrid_algorithm(
                    circuit=circuit,
                    parameters=parameters,
                    backend=backend,
                    shots=1000
                )
                execution_time = time.time() - start_time
                
                results[provider_name] = {
                    'expectation_value': result.expectation_value,
                    'execution_time': execution_time,
                    'backend_used': result.backend_used,
                    'shots_used': result.shots_used,
                    'cache_hit': result.cache_hit,
                    'success': True
                }
                
                logger.info(f"Provider {provider_name}: {result.expectation_value:.6f} "
                           f"in {execution_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                results[provider_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results


def main():
    """Main example function."""
    print("=" * 80)
    print("HYBRID QUANTUM-CLASSICAL ORCHESTRATION & MULTI-PROVIDER EXAMPLE")
    print("=" * 80)
    
    # 1. Provider Discovery and Management
    print("\n1. PROVIDER DISCOVERY AND MANAGEMENT")
    print("-" * 40)
    
    provider_manager = get_provider_manager()
    
    # Start device discovery
    provider_manager.start_device_discovery(interval=60)
    
    # List available providers
    providers = list_available_providers()
    print(f"Available providers: {list(providers.keys())}")
    
    for name, info in providers.items():
        print(f"  {name}: {info.description} ({info.status.value})")
        print(f"    Devices: {info.total_devices} total, {info.available_devices} available")
    
    # List available devices
    devices = list_available_devices()
    print(f"\nAvailable devices: {len(devices)}")
    for device in devices:
        print(f"  {device.name} ({device.provider}): {device.num_qubits} qubits, "
              f"{'online' if device.operational else 'offline'}")
    
    # 2. VQE Algorithm with Hybrid Orchestration
    print("\n2. VQE ALGORITHM WITH HYBRID ORCHESTRATION")
    print("-" * 40)
    
    vqe = VQEExample(num_qubits=2)
    vqe.create_ansatz_circuit()
    
    print(f"VQE circuit created with {vqe.circuit.num_qubits} qubits, "
          f"{vqe.circuit.num_operations} operations")
    
    # Run VQE on local simulator
    vqe_result = vqe.run_vqe(backend_name="local", max_iterations=30)
    
    print(f"VQE Results:")
    print(f"  Optimal energy: {vqe_result['optimal_energy']:.6f}")
    print(f"  Convergence: {vqe_result['convergence_status']}")
    print(f"  Iterations: {vqe_result['iterations']}")
    print(f"  Function evaluations: {vqe_result['function_evaluations']}")
    print(f"  Optimization time: {vqe_result['optimization_time']:.2f}s")
    print(f"  Executor statistics: {vqe_result['executor_statistics']}")
    
    # 3. QAOA Algorithm Example
    print("\n3. QAOA ALGORITHM EXAMPLE")
    print("-" * 40)
    
    qaoa = QAOAExample(problem_size=4)
    qaoa_result = qaoa.run_qaoa(backend_name="local", layers=2, max_iterations=50)
    
    print(f"QAOA Results:")
    print(f"  Optimal cost: {qaoa_result['optimal_cost']:.6f}")
    print(f"  Convergence: {qaoa_result['convergence_status']}")
    print(f"  Iterations: {qaoa_result['iterations']}")
    print(f"  Problem size: {qaoa_result['problem_size']} qubits")
    print(f"  QAOA layers: {qaoa_result['layers']}")
    
    # 4. Provider Comparison
    print("\n4. PROVIDER COMPARISON")
    print("-" * 40)
    
    # Create a simple test circuit
    with QuantumProgram() as test_qp:
        q0 = test_qp.allocate_qubit("q0")
        q1 = test_qp.allocate_qubit("q1")
        
        RY(q0, "theta")
        CNOT(q0, q1)
        RZ(q1, "phi")
        
        measure(q0)
        measure(q1)
    
    test_params = {"theta": np.pi/4, "phi": np.pi/6}
    
    comparison = ProviderComparisonExample()
    comparison_results = comparison.compare_providers(
        circuit=test_qp.circuit,
        parameters=test_params,
        providers=["local"]  # Add more providers as available
    )
    
    print("Provider comparison results:")
    for provider, result in comparison_results.items():
        if result['success']:
            print(f"  {provider}: {result['expectation_value']:.6f} "
                  f"({result['execution_time']:.3f}s)")
        else:
            print(f"  {provider}: FAILED - {result['error']}")
    
    # 5. Advanced Hybrid Workflow
    print("\n5. ADVANCED HYBRID WORKFLOW")
    print("-" * 40)
    
    # Demonstrate parameter sweeping and caching
    print("Running parameter sweep to demonstrate caching...")
    
    backend = get_active_backend()
    context = ExecutionContext(
        backend=backend,
        shots=500,
        enable_caching=True
    )
    
    executor = HybridExecutor(context)
    executor.set_circuit(test_qp.circuit)
    
    # Parameter sweep
    theta_values = np.linspace(0, 2*np.pi, 10)
    phi_values = np.linspace(0, np.pi, 5)
    
    sweep_results = []
    start_time = time.time()
    
    for theta in theta_values:
        for phi in phi_values:
            params = {"theta": theta, "phi": phi}
            result = executor.execute(params)
            sweep_results.append({
                'theta': theta,
                'phi': phi,
                'expectation': result.expectation_value,
                'cache_hit': result.cache_hit,
                'execution_time': result.execution_time
            })
    
    sweep_time = time.time() - start_time
    cache_hits = sum(1 for r in sweep_results if r['cache_hit'])
    
    print(f"Parameter sweep completed:")
    print(f"  Total executions: {len(sweep_results)}")
    print(f"  Cache hits: {cache_hits}")
    print(f"  Total time: {sweep_time:.2f}s")
    print(f"  Average time per execution: {sweep_time/len(sweep_results):.3f}s")
    
    # Final statistics
    final_stats = executor.get_statistics()
    print(f"  Final executor statistics: {final_stats}")
    
    # Stop device discovery
    provider_manager.stop_device_discovery()
    
    print("\n" + "=" * 80)
    print("HYBRID ORCHESTRATION EXAMPLE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main() 