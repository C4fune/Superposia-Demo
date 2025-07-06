"""
Comprehensive Test Suite for Experiment Tracking System

This test suite validates all major functionality of the quantum experiment
tracking system, including database operations, experiment management,
analysis features, and API endpoints.
"""

import unittest
import tempfile
import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any

# Test imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantum_platform.experiments import (
    ExperimentDatabase, ExperimentManager, ExperimentAnalyzer
)
from quantum_platform.experiments.models import (
    Experiment, Circuit, ExperimentResult, ExperimentType, ExperimentStatus
)


class TestExperimentDatabase(unittest.TestCase):
    """Test the experiment database functionality."""
    
    def setUp(self):
        """Set up test database."""
        self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_file.close()
        self.database = ExperimentDatabase(self.db_file.name)
    
    def tearDown(self):
        """Clean up test database."""
        self.database.close()
        if os.path.exists(self.db_file.name):
            os.unlink(self.db_file.name)
    
    def test_database_initialization(self):
        """Test database initialization."""
        self.assertIsNotNone(self.database.engine)
        self.assertIsNotNone(self.database.SessionLocal)
        
        # Test stats on empty database
        stats = self.database.get_database_stats()
        self.assertEqual(stats['total_circuits'], 0)
        self.assertEqual(stats['total_experiments'], 0)
        self.assertEqual(stats['total_results'], 0)
    
    def test_circuit_operations(self):
        """Test circuit CRUD operations."""
        # Create circuit
        circuit = self.database.create_circuit(
            name="Test Circuit",
            qasm_code="OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];",
            num_qubits=2,
            description="Test description"
        )
        
        self.assertIsNotNone(circuit.id)
        self.assertEqual(circuit.name, "Test Circuit")
        self.assertEqual(circuit.num_qubits, 2)
        self.assertIsNotNone(circuit.content_hash)
        
        # Get circuit
        retrieved = self.database.get_circuit(circuit.id)
        self.assertEqual(retrieved.id, circuit.id)
        self.assertEqual(retrieved.name, circuit.name)
        
        # List circuits
        circuits = self.database.get_circuits()
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0].id, circuit.id)
        
        # Update circuit
        success = self.database.update_circuit(circuit.id, {"description": "Updated description"})
        self.assertTrue(success)
        
        updated = self.database.get_circuit(circuit.id)
        self.assertEqual(updated.description, "Updated description")
        
        # Test duplicate circuit (same content hash)
        duplicate = self.database.create_circuit(
            name="Duplicate Circuit",
            qasm_code="OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];",
            num_qubits=2
        )
        self.assertEqual(duplicate.id, circuit.id)  # Should return existing circuit
    
    def test_experiment_operations(self):
        """Test experiment CRUD operations."""
        # First create a circuit
        circuit = self.database.create_circuit(
            name="Test Circuit",
            qasm_code="OPENQASM 2.0;\nqreg q[2];",
            num_qubits=2
        )
        
        # Create experiment
        experiment = self.database.create_experiment(
            name="Test Experiment",
            circuit_id=circuit.id,
            backend="local_simulator",
            experiment_type=ExperimentType.SINGLE_SHOT.value,
            shots=1000,
            tags=["test", "example"],
            metadata={"test_key": "test_value"}
        )
        
        self.assertIsNotNone(experiment.id)
        self.assertEqual(experiment.name, "Test Experiment")
        self.assertEqual(experiment.circuit_id, circuit.id)
        self.assertEqual(experiment.backend, "local_simulator")
        self.assertEqual(experiment.status, ExperimentStatus.CREATED.value)
        
        # Get experiment
        retrieved = self.database.get_experiment(experiment.id)
        self.assertEqual(retrieved.id, experiment.id)
        self.assertEqual(retrieved.name, experiment.name)
        
        # List experiments
        experiments = self.database.get_experiments()
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0].id, experiment.id)
        
        # Update experiment status
        success = self.database.update_experiment_status(
            experiment.id, ExperimentStatus.RUNNING.value
        )
        self.assertTrue(success)
        
        updated = self.database.get_experiment(experiment.id)
        self.assertEqual(updated.status, ExperimentStatus.RUNNING.value)
        self.assertIsNotNone(updated.started_at)
        
        # Test filtering
        filtered = self.database.get_experiments(status_filter=ExperimentStatus.RUNNING.value)
        self.assertEqual(len(filtered), 1)
        
        filtered = self.database.get_experiments(status_filter=ExperimentStatus.COMPLETED.value)
        self.assertEqual(len(filtered), 0)
    
    def test_result_operations(self):
        """Test experiment result operations."""
        # Create circuit and experiment
        circuit = self.database.create_circuit(
            name="Test Circuit",
            qasm_code="OPENQASM 2.0;\nqreg q[2];",
            num_qubits=2
        )
        
        experiment = self.database.create_experiment(
            name="Test Experiment",
            circuit_id=circuit.id,
            backend="local_simulator"
        )
        
        # Create result
        result = self.database.create_result(
            experiment_id=experiment.id,
            run_number=1,
            raw_counts={"00": 500, "11": 500},
            shots=1000,
            execution_time=45.7,
            fidelity=0.99,
            success_probability=1.0,
            expectation_value=0.5
        )
        
        self.assertIsNotNone(result.id)
        self.assertEqual(result.experiment_id, experiment.id)
        self.assertEqual(result.run_number, 1)
        self.assertEqual(result.shots, 1000)
        self.assertEqual(result.fidelity, 0.99)
        
        # Verify normalized counts were calculated
        self.assertIsNotNone(result.normalized_counts)
        self.assertEqual(result.normalized_counts["00"], 0.5)
        self.assertEqual(result.normalized_counts["11"], 0.5)
        
        # Get results
        results = self.database.get_results(experiment.id)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, result.id)
        
        # Verify experiment stats were updated
        updated_experiment = self.database.get_experiment(experiment.id)
        self.assertEqual(updated_experiment.total_runs, 1)
        self.assertEqual(updated_experiment.successful_runs, 1)
        self.assertEqual(updated_experiment.avg_execution_time, 45.7)
    
    def test_database_stats(self):
        """Test database statistics."""
        # Create test data
        circuit = self.database.create_circuit(
            name="Test Circuit",
            qasm_code="OPENQASM 2.0;\nqreg q[2];",
            num_qubits=2
        )
        
        exp1 = self.database.create_experiment(
            name="Experiment 1",
            circuit_id=circuit.id,
            backend="local_simulator",
            experiment_type=ExperimentType.SINGLE_SHOT.value
        )
        
        exp2 = self.database.create_experiment(
            name="Experiment 2",
            circuit_id=circuit.id,
            backend="noisy_simulator",
            experiment_type=ExperimentType.PARAMETER_SWEEP.value
        )
        
        # Update one experiment status
        self.database.update_experiment_status(exp1.id, ExperimentStatus.COMPLETED.value)
        
        # Add some results
        self.database.create_result(
            experiment_id=exp1.id,
            run_number=1,
            raw_counts={"00": 500, "11": 500},
            shots=1000
        )
        
        # Get stats
        stats = self.database.get_database_stats()
        
        self.assertEqual(stats['total_circuits'], 1)
        self.assertEqual(stats['total_experiments'], 2)
        self.assertEqual(stats['total_results'], 1)
        
        # Check breakdowns
        self.assertIn('experiments_by_status', stats)
        self.assertIn('experiments_by_backend', stats)
        self.assertIn('experiments_by_type', stats)
        
        self.assertEqual(stats['experiments_by_status'][ExperimentStatus.COMPLETED.value], 1)
        self.assertEqual(stats['experiments_by_status'][ExperimentStatus.CREATED.value], 1)
        
        self.assertEqual(stats['experiments_by_backend']['local_simulator'], 1)
        self.assertEqual(stats['experiments_by_backend']['noisy_simulator'], 1)


class TestExperimentManager(unittest.TestCase):
    """Test the experiment manager functionality."""
    
    def setUp(self):
        """Set up test manager."""
        self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_file.close()
        
        database = ExperimentDatabase(self.db_file.name)
        self.manager = ExperimentManager(database=database, enable_audit=False)
    
    def tearDown(self):
        """Clean up test manager."""
        self.manager.close()
        if os.path.exists(self.db_file.name):
            os.unlink(self.db_file.name)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        self.assertIsNotNone(self.manager.database)
        self.assertIsNotNone(self.manager.analyzer)
        self.assertIsNotNone(self.manager.executor)
    
    def test_experiment_creation_flow(self):
        """Test complete experiment creation flow."""
        # Create circuit through manager (this would normally use QuantumCircuit)
        circuit = self.manager.database.create_circuit(
            name="Bell State",
            qasm_code="OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];",
            num_qubits=2,
            description="Bell state preparation"
        )
        
        # Get mock backend
        from quantum_platform.hardware.backends.local_simulator import LocalSimulatorBackend
        backend = LocalSimulatorBackend()
        
        # Create experiment
        experiment = self.manager.create_experiment(
            name="Bell State Test",
            circuit_id=circuit.id,
            backend=backend,
            experiment_type=ExperimentType.SINGLE_SHOT.value,
            description="Test Bell state preparation",
            shots=1000,
            tags=["bell", "test"]
        )
        
        self.assertIsNotNone(experiment.id)
        self.assertEqual(experiment.name, "Bell State Test")
        self.assertEqual(experiment.backend, backend.name)
        
        # List experiments
        experiments = self.manager.list_experiments()
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0].experiment_id, experiment.id)
        
        # Get summary
        summary = self.manager.get_experiment_summary(experiment.id)
        self.assertIsNotNone(summary)
        self.assertEqual(summary.name, "Bell State Test")
    
    def test_active_experiments_tracking(self):
        """Test active experiments tracking."""
        # Initially no active experiments
        active = self.manager.get_active_experiments()
        self.assertEqual(len(active), 0)
        
        # Mock an active experiment
        test_exp_id = "test_experiment_123"
        self.manager._active_experiments[test_exp_id] = {
            "start_time": time.time(),
            "status": "running",
            "progress": 0.5
        }
        
        active = self.manager.get_active_experiments()
        self.assertEqual(len(active), 1)
        self.assertIn(test_exp_id, active)
        self.assertEqual(active[test_exp_id]["status"], "running")
        self.assertEqual(active[test_exp_id]["progress"], 0.5)
        
        # Test cancellation
        success = self.manager.cancel_experiment(test_exp_id)
        self.assertTrue(success)
        
        active = self.manager.get_active_experiments()
        self.assertEqual(active[test_exp_id]["status"], "cancelled")
    
    def test_export_functionality(self):
        """Test experiment data export."""
        # Create test data
        circuit = self.manager.database.create_circuit(
            name="Test Circuit",
            qasm_code="OPENQASM 2.0;\nqreg q[2];",
            num_qubits=2
        )
        
        from quantum_platform.hardware.backends.local_simulator import LocalSimulatorBackend
        backend = LocalSimulatorBackend()
        
        experiment = self.manager.create_experiment(
            name="Export Test",
            circuit_id=circuit.id,
            backend=backend
        )
        
        # Add a result
        self.manager.database.create_result(
            experiment_id=experiment.id,
            run_number=1,
            raw_counts={"00": 500, "01": 250, "10": 250, "11": 0},
            shots=1000,
            fidelity=0.95
        )
        
        # Test JSON export
        json_data = self.manager.export_experiment_data(experiment.id, "json")
        self.assertIsInstance(json_data, str)
        
        parsed_data = json.loads(json_data)
        self.assertIn("experiment", parsed_data)
        self.assertIn("circuit", parsed_data)
        self.assertIn("results", parsed_data)
        
        # Test CSV export
        csv_data = self.manager.export_experiment_data(experiment.id, "csv")
        self.assertIsInstance(csv_data, str)
        self.assertIn("run_number", csv_data)
        self.assertIn("execution_time", csv_data)


class TestExperimentAnalyzer(unittest.TestCase):
    """Test the experiment analyzer functionality."""
    
    def setUp(self):
        """Set up test analyzer."""
        self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_file.close()
        
        self.database = ExperimentDatabase(self.db_file.name)
        self.analyzer = ExperimentAnalyzer(self.database)
        
        # Create test data
        self.circuit = self.database.create_circuit(
            name="Test Circuit",
            qasm_code="OPENQASM 2.0;\nqreg q[2];",
            num_qubits=2
        )
        
        self.experiment = self.database.create_experiment(
            name="Analysis Test",
            circuit_id=self.circuit.id,
            backend="local_simulator"
        )
    
    def tearDown(self):
        """Clean up test analyzer."""
        self.database.close()
        if os.path.exists(self.db_file.name):
            os.unlink(self.db_file.name)
    
    def test_basic_analysis(self):
        """Test basic experiment analysis."""
        # Add some results
        results_data = [
            {"raw_counts": {"00": 480, "11": 520}, "fidelity": 0.99, "execution_time": 45.2},
            {"raw_counts": {"00": 505, "11": 495}, "fidelity": 0.97, "execution_time": 48.1},
            {"raw_counts": {"00": 490, "11": 510}, "fidelity": 0.98, "execution_time": 46.5}
        ]
        
        for i, result_data in enumerate(results_data):
            self.database.create_result(
                experiment_id=self.experiment.id,
                run_number=i + 1,
                raw_counts=result_data["raw_counts"],
                shots=1000,
                fidelity=result_data["fidelity"],
                execution_time=result_data["execution_time"]
            )
        
        # Analyze experiment
        analysis = self.analyzer.analyze_experiment(self.experiment.id)
        
        self.assertEqual(analysis["experiment_id"], self.experiment.id)
        self.assertEqual(analysis["total_runs"], 3)
        
        # Check statistics
        stats = analysis["statistics"]
        self.assertEqual(stats["total_runs"], 3)
        self.assertEqual(stats["successful_runs"], 3)
        self.assertEqual(stats["success_rate"], 1.0)
        
        # Check fidelity statistics
        fidelity_stats = stats["fidelity"]
        self.assertAlmostEqual(fidelity_stats["mean"], 0.98, places=2)
        self.assertGreater(fidelity_stats["std"], 0)
        self.assertEqual(fidelity_stats["min"], 0.97)
        self.assertEqual(fidelity_stats["max"], 0.99)
        
        # Check execution time statistics
        time_stats = stats["execution_time"]
        self.assertAlmostEqual(time_stats["mean"], 46.6, places=1)
        self.assertGreater(time_stats["std"], 0)
        
        # Check performance metrics
        performance = analysis["performance_metrics"]
        self.assertEqual(performance["reliability"], 1.0)
        self.assertGreater(performance["consistency"], 0.8)
        
        # Check recommendations
        self.assertIsInstance(analysis["recommendations"], list)
    
    def test_anomaly_detection(self):
        """Test anomaly detection in results."""
        # Add normal results
        for i in range(5):
            self.database.create_result(
                experiment_id=self.experiment.id,
                run_number=i + 1,
                raw_counts={"00": 500, "11": 500},
                shots=1000,
                fidelity=0.98 + 0.01 * i,  # Gradual increase
                execution_time=45.0 + i
            )
        
        # Add anomalous result
        self.database.create_result(
            experiment_id=self.experiment.id,
            run_number=6,
            raw_counts={"00": 500, "11": 500},
            shots=1000,
            fidelity=0.5,  # Much lower fidelity
            execution_time=200.0  # Much higher execution time
        )
        
        analysis = self.analyzer.analyze_experiment(self.experiment.id)
        anomalies = analysis["anomalies"]
        
        # Should detect both fidelity and execution time anomalies
        self.assertGreater(len(anomalies), 0)
        
        anomaly_types = [a["type"] for a in anomalies]
        self.assertIn("fidelity_anomaly", anomaly_types)
        self.assertIn("execution_time_anomaly", anomaly_types)
    
    def test_performance_issue_detection(self):
        """Test performance issue detection."""
        # Add results with high failure rate
        for i in range(10):
            status = "failed" if i < 5 else "completed"  # 50% failure rate
            fidelity = 0.5 if status == "failed" else 0.95
            
            self.database.create_result(
                experiment_id=self.experiment.id,
                run_number=i + 1,
                raw_counts={"00": 500, "11": 500},
                shots=1000,
                fidelity=fidelity,
                execution_time=45.0
            )
            
            # Update result status
            result = self.database.get_results(self.experiment.id)[-1]
            with self.database.get_session() as session:
                db_result = session.query(ExperimentResult).filter_by(id=result.id).first()
                db_result.status = status
                session.commit()
        
        issues = self.analyzer.detect_performance_issues(self.experiment.id)
        
        self.assertGreater(len(issues), 0)
        
        issue_types = [i["type"] for i in issues]
        self.assertIn("high_failure_rate", issue_types)
        self.assertIn("low_fidelity", issue_types)
    
    def test_experiment_comparison(self):
        """Test experiment comparison functionality."""
        # Create second experiment
        experiment2 = self.database.create_experiment(
            name="Comparison Test 2",
            circuit_id=self.circuit.id,
            backend="local_simulator"
        )
        
        # Add results to both experiments
        # Experiment 1: High fidelity, fast execution
        for i in range(3):
            self.database.create_result(
                experiment_id=self.experiment.id,
                run_number=i + 1,
                raw_counts={"00": 500, "11": 500},
                shots=1000,
                fidelity=0.98,
                execution_time=45.0
            )
        
        # Experiment 2: Lower fidelity, slower execution
        for i in range(3):
            self.database.create_result(
                experiment_id=experiment2.id,
                run_number=i + 1,
                raw_counts={"00": 500, "11": 500},
                shots=1000,
                fidelity=0.85,
                execution_time=65.0
            )
        
        # Compare experiments
        comparison = self.analyzer.compare_experiments(
            self.experiment.id, experiment2.id
        )
        
        self.assertEqual(comparison.experiment1_id, self.experiment.id)
        self.assertEqual(comparison.experiment2_id, experiment2.id)
        
        # Should show experiment 1 has higher fidelity
        self.assertGreater(comparison.fidelity_difference, 0)
        
        # Should show experiment 1 has faster execution
        self.assertLess(comparison.execution_time_difference, 0)
        
        # Check detailed analysis
        self.assertIn("experiment1_stats", comparison.detailed_analysis)
        self.assertIn("experiment2_stats", comparison.detailed_analysis)
    
    def test_report_generation(self):
        """Test comprehensive report generation."""
        # Add some results
        self.database.create_result(
            experiment_id=self.experiment.id,
            run_number=1,
            raw_counts={"00": 500, "11": 500},
            shots=1000,
            fidelity=0.98,
            execution_time=45.0
        )
        
        # Generate report
        report = self.analyzer.generate_experiment_report(self.experiment.id)
        
        # Check report structure
        self.assertIn("experiment_info", report)
        self.assertIn("circuit_info", report)
        self.assertIn("execution_info", report)
        self.assertIn("analysis", report)
        self.assertIn("performance_issues", report)
        self.assertIn("generated_at", report)
        
        # Check experiment info
        exp_info = report["experiment_info"]
        self.assertEqual(exp_info["id"], self.experiment.id)
        self.assertEqual(exp_info["name"], self.experiment.name)
        
        # Check circuit info
        circuit_info = report["circuit_info"]
        self.assertEqual(circuit_info["id"], self.circuit.id)
        self.assertEqual(circuit_info["name"], self.circuit.name)
        
        # Check execution info
        exec_info = report["execution_info"]
        self.assertEqual(exec_info["backend"], self.experiment.backend)
        self.assertEqual(exec_info["total_runs"], 1)
        self.assertEqual(exec_info["successful_runs"], 1)


class TestSystemIntegration(unittest.TestCase):
    """Test system integration and end-to-end workflows."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_file.close()
        
        database = ExperimentDatabase(self.db_file.name)
        self.manager = ExperimentManager(database=database, enable_audit=False)
    
    def tearDown(self):
        """Clean up integration test environment."""
        self.manager.close()
        if os.path.exists(self.db_file.name):
            os.unlink(self.db_file.name)
    
    def test_complete_experiment_workflow(self):
        """Test complete experiment workflow from creation to analysis."""
        # Step 1: Create circuit
        circuit = self.manager.database.create_circuit(
            name="Integration Test Circuit",
            qasm_code="OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];",
            num_qubits=2,
            description="Circuit for integration testing"
        )
        
        # Step 2: Create experiment
        from quantum_platform.hardware.backends.local_simulator import LocalSimulatorBackend
        backend = LocalSimulatorBackend()
        
        experiment = self.manager.create_experiment(
            name="Integration Test Experiment",
            circuit_id=circuit.id,
            backend=backend,
            experiment_type=ExperimentType.SINGLE_SHOT.value,
            description="End-to-end integration test",
            shots=1000,
            tags=["integration", "test"]
        )
        
        # Step 3: Simulate experiment execution by adding results
        self.manager.database.update_experiment_status(
            experiment.id, ExperimentStatus.RUNNING.value
        )
        
        # Add multiple results
        for i in range(5):
            self.manager.database.create_result(
                experiment_id=experiment.id,
                run_number=i + 1,
                raw_counts={"00": 480 + i*5, "11": 520 - i*5},
                shots=1000,
                fidelity=0.95 + i*0.01,
                execution_time=45.0 + i*2.0,
                success_probability=0.99
            )
        
        self.manager.database.update_experiment_status(
            experiment.id, ExperimentStatus.COMPLETED.value
        )
        
        # Step 4: Analyze results
        analysis = self.manager.analyzer.analyze_experiment(experiment.id)
        
        self.assertEqual(analysis["experiment_id"], experiment.id)
        self.assertEqual(analysis["total_runs"], 5)
        self.assertEqual(analysis["statistics"]["success_rate"], 1.0)
        self.assertAlmostEqual(analysis["statistics"]["fidelity"]["mean"], 0.97, places=2)
        
        # Step 5: Generate report
        report = self.manager.analyzer.generate_experiment_report(experiment.id)
        
        self.assertIsNotNone(report)
        self.assertEqual(report["experiment_info"]["name"], "Integration Test Experiment")
        self.assertEqual(report["execution_info"]["total_runs"], 5)
        
        # Step 6: Export data
        json_export = self.manager.export_experiment_data(experiment.id, "json")
        self.assertIsInstance(json_export, str)
        
        csv_export = self.manager.export_experiment_data(experiment.id, "csv")
        self.assertIsInstance(csv_export, str)
        
        # Step 7: Database statistics
        stats = self.manager.get_database_stats()
        self.assertEqual(stats["total_experiments"], 1)
        self.assertEqual(stats["total_circuits"], 1)
        self.assertEqual(stats["total_results"], 5)
    
    def test_maintenance_operations(self):
        """Test maintenance and database management operations."""
        # Create some test data
        circuit = self.manager.database.create_circuit(
            name="Maintenance Test Circuit",
            qasm_code="OPENQASM 2.0;\nqreg q[2];",
            num_qubits=2
        )
        
        from quantum_platform.hardware.backends.local_simulator import LocalSimulatorBackend
        backend = LocalSimulatorBackend()
        
        experiment = self.manager.create_experiment(
            name="Maintenance Test",
            circuit_id=circuit.id,
            backend=backend
        )
        
        # Test backup functionality
        backup_path = tempfile.mktemp(suffix='.db')
        success = self.manager.backup_experiments(backup_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(backup_path))
        
        # Verify backup integrity by opening it
        backup_db = ExperimentDatabase(backup_path)
        backup_stats = backup_db.get_database_stats()
        backup_db.close()
        
        original_stats = self.manager.get_database_stats()
        self.assertEqual(backup_stats["total_experiments"], original_stats["total_experiments"])
        self.assertEqual(backup_stats["total_circuits"], original_stats["total_circuits"])
        
        # Clean up backup file
        os.unlink(backup_path)


def run_tests():
    """Run all tests and report results."""
    
    print("=" * 60)
    print("QUANTUM EXPERIMENT TRACKING SYSTEM - TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestExperimentDatabase,
        TestExperimentManager,
        TestExperimentAnalyzer,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors
    
    print(f"Total Tests:    {total_tests}")
    print(f"Successful:     {successes}")
    print(f"Failures:       {failures}")
    print(f"Errors:         {errors}")
    print(f"Success Rate:   {successes/total_tests*100:.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if successes == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! Experiment tracking system is ready for deployment.")
        return True
    else:
        print(f"\n❌ Some tests failed. Please review and fix issues before deployment.")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 