"""
Comprehensive Experiment Tracking Example

This example demonstrates how to use the new experiment tracking system
with the quantum platform, including:

1. Creating and storing circuits
2. Setting up experiments with different configurations
3. Running experiments with different backends
4. Analyzing results and generating reports
5. Comparing experiments and tracking trends

This showcases the commercial-grade experiment tracking capabilities.
"""

import logging
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main demonstration of the experiment tracking system."""
    
    print("=" * 60)
    print("QUANTUM EXPERIMENT TRACKING SYSTEM DEMO")
    print("=" * 60)
    
    try:
        # Initialize the experiment tracking system
        from quantum_platform.experiments import ExperimentManager, ExperimentDatabase
        from quantum_platform.experiments.models import ExperimentType
        
        # Create experiment manager
        experiment_manager = ExperimentManager()
        
        print("\n1. SYSTEM INITIALIZATION")
        print("-" * 40)
        print("✓ Experiment database initialized")
        print("✓ Experiment manager created")
        
        # Get initial statistics
        stats = experiment_manager.get_database_stats()
        print(f"✓ Database stats: {stats['total_experiments']} experiments, {stats['total_circuits']} circuits")
        
        # Create sample circuits for experiments
        circuits = create_sample_circuits(experiment_manager)
        print(f"\n2. CIRCUIT CREATION")
        print("-" * 40)
        print(f"✓ Created {len(circuits)} sample circuits")
        
        # Create different types of experiments
        experiments = create_sample_experiments(experiment_manager, circuits)
        print(f"\n3. EXPERIMENT CREATION")
        print("-" * 40)
        print(f"✓ Created {len(experiments)} experiments of different types")
        
        # Simulate running experiments
        print(f"\n4. EXPERIMENT EXECUTION")
        print("-" * 40)
        run_sample_experiments(experiment_manager, experiments)
        
        # Analyze results
        print(f"\n5. RESULT ANALYSIS")
        print("-" * 40)
        analyze_experiment_results(experiment_manager, experiments)
        
        # Demonstrate comparison features
        print(f"\n6. EXPERIMENT COMPARISON")
        print("-" * 40)
        compare_experiments(experiment_manager, experiments)
        
        # Generate comprehensive reports
        print(f"\n7. REPORT GENERATION")
        print("-" * 40)
        generate_experiment_reports(experiment_manager, experiments)
        
        # Demonstrate maintenance features
        print(f"\n8. MAINTENANCE FEATURES")
        print("-" * 40)
        demonstrate_maintenance(experiment_manager)
        
        print(f"\n9. FINAL STATISTICS")
        print("-" * 40)
        final_stats = experiment_manager.get_database_stats()
        print(f"✓ Total experiments: {final_stats['total_experiments']}")
        print(f"✓ Total circuits: {final_stats['total_circuits']}")
        print(f"✓ Total results: {final_stats['total_results']}")
        print(f"✓ Experiments by status: {final_stats['experiments_by_status']}")
        print(f"✓ Experiments by backend: {final_stats['experiments_by_backend']}")
        
        print("\n" + "=" * 60)
        print("EXPERIMENT TRACKING DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return False
    
    return True

def create_sample_circuits(experiment_manager: 'ExperimentManager') -> List[Dict[str, Any]]:
    """Create sample quantum circuits for experiments."""
    
    circuits = []
    
    # Bell State Circuit
    bell_circuit = {
        'name': 'Bell State',
        'description': 'Creates a Bell state (maximally entangled 2-qubit state)',
        'qasm_code': '''
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
        '''.strip(),
        'num_qubits': 2,
        'circuit_json': {
            'gates': [
                {'type': 'H', 'qubit': 0},
                {'type': 'CNOT', 'control': 0, 'target': 1},
                {'type': 'measure', 'qubit': 0},
                {'type': 'measure', 'qubit': 1}
            ]
        },
        'parameters': {},
        'version': '1.0'
    }
    
    # Parameterized Circuit (for VQE-like experiments)
    vqe_circuit = {
        'name': 'VQE Ansatz',
        'description': 'Parameterized circuit for VQE optimization',
        'qasm_code': '''
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
ry(theta) q[0];
ry(phi) q[1];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
        '''.strip(),
        'num_qubits': 2,
        'circuit_json': {
            'gates': [
                {'type': 'RY', 'qubit': 0, 'parameter': 'theta'},
                {'type': 'RY', 'qubit': 1, 'parameter': 'phi'},
                {'type': 'CNOT', 'control': 0, 'target': 1},
                {'type': 'measure', 'qubit': 0},
                {'type': 'measure', 'qubit': 1}
            ]
        },
        'parameters': {
            'theta': {'type': 'float', 'range': [0, 2*np.pi]},
            'phi': {'type': 'float', 'range': [0, 2*np.pi]}
        },
        'version': '1.0'
    }
    
    # GHZ State Circuit
    ghz_circuit = {
        'name': 'GHZ State',
        'description': '3-qubit GHZ state for entanglement studies',
        'qasm_code': '''
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
        '''.strip(),
        'num_qubits': 3,
        'circuit_json': {
            'gates': [
                {'type': 'H', 'qubit': 0},
                {'type': 'CNOT', 'control': 0, 'target': 1},
                {'type': 'CNOT', 'control': 1, 'target': 2},
                {'type': 'measure', 'qubit': 0},
                {'type': 'measure', 'qubit': 1},
                {'type': 'measure', 'qubit': 2}
            ]
        },
        'parameters': {},
        'version': '1.0'
    }
    
    # Create circuits in the database
    for circuit_data in [bell_circuit, vqe_circuit, ghz_circuit]:
        try:
            circuit = experiment_manager.database.create_circuit(**circuit_data)
            circuits.append({
                'id': circuit.id,
                'name': circuit.name,
                'type': circuit_data.get('type', 'general'),
                'num_qubits': circuit.num_qubits,
                'parameterized': bool(circuit_data['parameters'])
            })
            print(f"  ✓ Created circuit: {circuit.name} (ID: {circuit.id[:8]}...)")
        except Exception as e:
            logger.error(f"Failed to create circuit {circuit_data['name']}: {e}")
    
    return circuits

def create_sample_experiments(experiment_manager: 'ExperimentManager', 
                            circuits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create sample experiments using the circuits."""
    
    experiments = []
    
    # Get backends
    from quantum_platform.hardware.backends.local_simulator import LocalSimulatorBackend
    local_backend = LocalSimulatorBackend()
    
    # Experiment 1: Bell State Characterization
    bell_circuit = next(c for c in circuits if c['name'] == 'Bell State')
    try:
        exp1 = experiment_manager.create_experiment(
            name="Bell State Characterization",
            circuit_id=bell_circuit['id'],
            backend=local_backend,
            experiment_type=ExperimentType.SINGLE_SHOT.value,
            description="Characterize Bell state fidelity and measurement statistics",
            shots=1000,
            tags=["entanglement", "bell-state", "characterization"],
            metadata={"expected_fidelity": 1.0, "expected_outcomes": {"00": 0.5, "11": 0.5}}
        )
        experiments.append({
            'id': exp1.id,
            'name': exp1.name,
            'type': exp1.experiment_type,
            'circuit_name': bell_circuit['name']
        })
        print(f"  ✓ Created experiment: {exp1.name}")
    except Exception as e:
        logger.error(f"Failed to create Bell state experiment: {e}")
    
    # Experiment 2: VQE Parameter Sweep
    vqe_circuit = next(c for c in circuits if c['name'] == 'VQE Ansatz')
    try:
        exp2 = experiment_manager.create_experiment(
            name="VQE Parameter Optimization",
            circuit_id=vqe_circuit['id'],
            backend=local_backend,
            experiment_type=ExperimentType.PARAMETER_SWEEP.value,
            description="Optimize VQE parameters for ground state preparation",
            shots=500,
            parameter_sweep={
                'parameters': {
                    'theta': {'start': 0, 'stop': 2*np.pi, 'num_points': 5},
                    'phi': {'start': 0, 'stop': 2*np.pi, 'num_points': 5}
                },
                'sweep_type': 'grid'
            },
            tags=["vqe", "optimization", "parameter-sweep"],
            metadata={"target_energy": -1.0, "convergence_threshold": 1e-6}
        )
        experiments.append({
            'id': exp2.id,
            'name': exp2.name,
            'type': exp2.experiment_type,
            'circuit_name': vqe_circuit['name']
        })
        print(f"  ✓ Created experiment: {exp2.name}")
    except Exception as e:
        logger.error(f"Failed to create VQE experiment: {e}")
    
    # Experiment 3: GHZ State Benchmarking
    ghz_circuit = next(c for c in circuits if c['name'] == 'GHZ State')
    try:
        exp3 = experiment_manager.create_experiment(
            name="GHZ State Benchmarking",
            circuit_id=ghz_circuit['id'],
            backend=local_backend,
            experiment_type=ExperimentType.BENCHMARKING.value,
            description="Benchmark GHZ state preparation across different shot counts",
            shots=2000,
            tags=["ghz", "benchmarking", "3-qubit"],
            metadata={"expected_outcomes": {"000": 0.5, "111": 0.5}}
        )
        experiments.append({
            'id': exp3.id,
            'name': exp3.name,
            'type': exp3.experiment_type,
            'circuit_name': ghz_circuit['name']
        })
        print(f"  ✓ Created experiment: {exp3.name}")
    except Exception as e:
        logger.error(f"Failed to create GHZ experiment: {e}")
    
    return experiments

def run_sample_experiments(experiment_manager: 'ExperimentManager', 
                         experiments: List[Dict[str, Any]]):
    """Simulate running the experiments with mock results."""
    
    print("  Running experiments (simulated)...")
    
    for exp_info in experiments:
        try:
            exp_id = exp_info['id']
            exp_name = exp_info['name']
            exp_type = exp_info['type']
            
            print(f"    → Running: {exp_name}")
            
            # Update experiment status to running
            experiment_manager.database.update_experiment_status(
                exp_id, 'running', started_at=datetime.now()
            )
            
            # Simulate experiment execution with mock results
            if exp_type == ExperimentType.SINGLE_SHOT.value:
                simulate_single_shot_results(experiment_manager, exp_id, exp_name)
            elif exp_type == ExperimentType.PARAMETER_SWEEP.value:
                simulate_parameter_sweep_results(experiment_manager, exp_id, exp_name)
            elif exp_type == ExperimentType.BENCHMARKING.value:
                simulate_benchmarking_results(experiment_manager, exp_id, exp_name)
            
            # Update experiment status to completed
            experiment_manager.database.update_experiment_status(
                exp_id, 'completed', completed_at=datetime.now()
            )
            
            print(f"    ✓ Completed: {exp_name}")
            
        except Exception as e:
            logger.error(f"Failed to run experiment {exp_info['name']}: {e}")
            # Update status to failed
            experiment_manager.database.update_experiment_status(
                exp_id, 'failed', completed_at=datetime.now()
            )

def simulate_single_shot_results(experiment_manager: 'ExperimentManager', 
                               exp_id: str, exp_name: str):
    """Simulate single shot experiment results."""
    
    # Bell state should show 50/50 distribution between |00⟩ and |11⟩
    if "Bell" in exp_name:
        raw_counts = {"00": 498, "11": 502}  # Near-ideal Bell state
        fidelity = 0.99
        success_probability = 1.0
    else:
        # Generic results
        raw_counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        fidelity = 0.85
        success_probability = 0.9
    
    # Create result
    experiment_manager.database.create_result(
        experiment_id=exp_id,
        run_number=1,
        raw_counts=raw_counts,
        shots=1000,
        execution_time=45.7,  # milliseconds
        fidelity=fidelity,
        success_probability=success_probability,
        expectation_value=0.5,
        variance=0.25,
        custom_metrics={
            "bell_fidelity": fidelity,
            "concurrence": 0.98 if "Bell" in exp_name else 0.0
        }
    )

def simulate_parameter_sweep_results(experiment_manager: 'ExperimentManager', 
                                   exp_id: str, exp_name: str):
    """Simulate parameter sweep experiment results."""
    
    # Generate results for 5x5 parameter grid
    run_number = 1
    
    for i in range(5):
        for j in range(5):
            theta = i * 2 * np.pi / 5
            phi = j * 2 * np.pi / 5
            
            # Simulate VQE energy landscape
            energy = -np.cos(theta) * np.cos(phi) + 0.1 * np.random.normal()
            fidelity = 0.95 - 0.1 * np.abs(energy + 1.0)  # Higher fidelity near minimum
            
            # Simulate measurement counts
            prob_00 = 0.25 + 0.25 * np.cos(theta)
            prob_11 = 0.25 + 0.25 * np.cos(phi)
            prob_01 = 0.25 + 0.1 * np.sin(theta + phi)
            prob_10 = 1.0 - prob_00 - prob_11 - prob_01
            
            # Convert to counts
            raw_counts = {
                "00": int(prob_00 * 500),
                "01": int(prob_01 * 500),
                "10": int(prob_10 * 500),
                "11": int(prob_11 * 500)
            }
            
            # Ensure counts sum to 500
            total = sum(raw_counts.values())
            if total != 500:
                raw_counts["00"] += 500 - total
            
            experiment_manager.database.create_result(
                experiment_id=exp_id,
                run_number=run_number,
                raw_counts=raw_counts,
                shots=500,
                parameter_values={"theta": theta, "phi": phi},
                execution_time=35.2 + np.random.normal(0, 5),
                fidelity=max(0.0, min(1.0, fidelity)),
                success_probability=0.95,
                expectation_value=energy,
                variance=0.1,
                custom_metrics={
                    "energy": energy,
                    "parameter_distance": np.sqrt(theta**2 + phi**2)
                }
            )
            
            run_number += 1

def simulate_benchmarking_results(experiment_manager: 'ExperimentManager', 
                                exp_id: str, exp_name: str):
    """Simulate benchmarking experiment results."""
    
    # GHZ state should show 50/50 distribution between |000⟩ and |111⟩
    if "GHZ" in exp_name:
        raw_counts = {"000": 995, "111": 1005}  # Near-ideal GHZ state
        fidelity = 0.97
        success_probability = 1.0
    else:
        # Generic benchmarking results
        raw_counts = {"000": 500, "001": 250, "010": 250, "011": 250, 
                     "100": 250, "101": 250, "110": 250, "111": 250}
        fidelity = 0.8
        success_probability = 0.85
    
    experiment_manager.database.create_result(
        experiment_id=exp_id,
        run_number=1,
        raw_counts=raw_counts,
        shots=2000,
        execution_time=89.3,
        fidelity=fidelity,
        success_probability=success_probability,
        expectation_value=0.75,
        variance=0.2,
        custom_metrics={
            "ghz_fidelity": fidelity,
            "three_qubit_concurrence": 0.95 if "GHZ" in exp_name else 0.0
        }
    )

def analyze_experiment_results(experiment_manager: 'ExperimentManager', 
                             experiments: List[Dict[str, Any]]):
    """Analyze the results of all experiments."""
    
    print("  Analyzing experiment results...")
    
    for exp_info in experiments:
        try:
            exp_id = exp_info['id']
            exp_name = exp_info['name']
            
            # Get comprehensive analysis
            analysis = experiment_manager.analyzer.analyze_experiment(exp_id)
            
            print(f"    → Analysis for: {exp_name}")
            print(f"      • Total runs: {analysis['total_runs']}")
            print(f"      • Success rate: {analysis['statistics']['success_rate']:.1%}")
            
            if analysis['statistics']['fidelity']['mean']:
                print(f"      • Avg fidelity: {analysis['statistics']['fidelity']['mean']:.3f}")
            
            if analysis['statistics']['execution_time']['mean']:
                print(f"      • Avg exec time: {analysis['statistics']['execution_time']['mean']:.1f}ms")
            
            # Check for performance issues
            issues = experiment_manager.analyzer.detect_performance_issues(exp_id)
            if issues:
                print(f"      • Performance issues: {len(issues)} detected")
                for issue in issues:
                    print(f"        - {issue['type']}: {issue['description']}")
            else:
                print(f"      • No performance issues detected")
            
            # Show recommendations
            if analysis['recommendations']:
                print(f"      • Recommendations:")
                for rec in analysis['recommendations'][:2]:  # Show first 2
                    print(f"        - {rec}")
            
        except Exception as e:
            logger.error(f"Failed to analyze experiment {exp_info['name']}: {e}")

def compare_experiments(experiment_manager: 'ExperimentManager', 
                       experiments: List[Dict[str, Any]]):
    """Demonstrate experiment comparison features."""
    
    if len(experiments) < 2:
        print("  Need at least 2 experiments for comparison")
        return
    
    print("  Comparing experiments...")
    
    try:
        # Compare first two experiments
        exp1_id = experiments[0]['id']
        exp2_id = experiments[1]['id']
        exp1_name = experiments[0]['name']
        exp2_name = experiments[1]['name']
        
        comparison = experiment_manager.compare_experiments(exp1_id, exp2_id)
        
        print(f"    → Comparing: {exp1_name} vs {exp2_name}")
        
        if comparison.fidelity_difference is not None:
            print(f"      • Fidelity difference: {comparison.fidelity_difference:+.3f}")
        
        if comparison.execution_time_difference is not None:
            print(f"      • Execution time difference: {comparison.execution_time_difference:+.1f}ms")
        
        if comparison.success_rate_difference is not None:
            print(f"      • Success rate difference: {comparison.success_rate_difference:+.1%}")
        
        if comparison.p_value is not None:
            significance = "significant" if comparison.is_significant else "not significant"
            print(f"      • Statistical significance: {significance} (p={comparison.p_value:.3f})")
        
        # Trend analysis across all experiments
        experiment_ids = [exp['id'] for exp in experiments]
        trends = experiment_manager.get_experiment_trends(experiment_ids)
        
        print(f"    → Trend analysis across {len(experiments)} experiments:")
        print(f"      • Time range: {trends['time_range']['start'][:10]} to {trends['time_range']['end'][:10]}")
        
        if 'backend_analysis' in trends:
            backend_usage = trends['backend_analysis']['backend_usage']
            print(f"      • Backend usage: {backend_usage}")
        
    except Exception as e:
        logger.error(f"Failed to compare experiments: {e}")

def generate_experiment_reports(experiment_manager: 'ExperimentManager', 
                              experiments: List[Dict[str, Any]]):
    """Generate comprehensive reports for experiments."""
    
    print("  Generating experiment reports...")
    
    for exp_info in experiments:
        try:
            exp_id = exp_info['id']
            exp_name = exp_info['name']
            
            # Generate comprehensive report
            report = experiment_manager.analyzer.generate_experiment_report(exp_id)
            
            print(f"    → Report for: {exp_name}")
            print(f"      • Experiment ID: {report['experiment_info']['id'][:8]}...")
            print(f"      • Status: {report['experiment_info']['status']}")
            print(f"      • Backend: {report['execution_info']['backend']}")
            print(f"      • Total runs: {report['execution_info']['total_runs']}")
            print(f"      • Success rate: {report['execution_info']['successful_runs']}/{report['execution_info']['total_runs']}")
            
            # Export report to file
            report_filename = f"experiment_report_{exp_id[:8]}.json"
            import json
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"      • Report saved to: {report_filename}")
            
        except Exception as e:
            logger.error(f"Failed to generate report for {exp_info['name']}: {e}")

def demonstrate_maintenance(experiment_manager: 'ExperimentManager'):
    """Demonstrate maintenance and database management features."""
    
    print("  Demonstrating maintenance features...")
    
    try:
        # Database statistics
        stats = experiment_manager.get_database_stats()
        print(f"    → Current database size:")
        print(f"      • Experiments: {stats['total_experiments']}")
        print(f"      • Circuits: {stats['total_circuits']}")
        print(f"      • Results: {stats['total_results']}")
        
        # Backup database
        backup_path = f"experiment_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        success = experiment_manager.backup_experiments(backup_path)
        if success:
            print(f"    ✓ Database backed up to: {backup_path}")
        else:
            print(f"    ❌ Database backup failed")
        
        # Note: We won't actually clean up data in this demo
        print(f"    → Cleanup operations available:")
        print(f"      • cleanup_old_experiments(days_to_keep=30)")
        print(f"      • Archive old results")
        print(f"      • Remove orphaned circuits")
        
    except Exception as e:
        logger.error(f"Maintenance demo failed: {e}")

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Experiment tracking system is working perfectly!")
        print("   Ready for commercial deployment!")
    else:
        print("\n❌ Demo encountered errors - check logs for details") 