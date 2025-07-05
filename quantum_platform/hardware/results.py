"""
Quantum Execution Results Management

This module provides comprehensive result management for multi-shot quantum
circuit execution, including result aggregation, analysis, and storage.
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np

from ..errors import ExecutionError
from .hal import HardwareResult, JobHandle, JobStatus


@dataclass
class ShotResult:
    """Result from a single shot execution."""
    shot_id: int
    outcome: str  # Bitstring result (e.g., "001101")
    measurement_data: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None  # microseconds
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Aggregated results from multi-shot execution."""
    
    # Basic execution info
    total_shots: int
    successful_shots: int
    failed_shots: int = 0
    
    # Result data
    counts: Dict[str, int] = field(default_factory=dict)
    probabilities: Dict[str, float] = field(default_factory=dict)
    raw_shots: Optional[List[ShotResult]] = None
    
    # Statistics
    most_frequent: Optional[str] = None
    least_frequent: Optional[str] = None
    entropy: Optional[float] = None
    unique_outcomes: int = 0
    
    # Execution metadata
    total_execution_time: Optional[float] = None  # milliseconds
    average_shot_time: Optional[float] = None  # microseconds
    circuit_id: Optional[str] = None
    backend_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Hardware-specific data
    hardware_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived statistics."""
        if self.counts and self.successful_shots > 0:
            self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate statistics from counts."""
        total = sum(self.counts.values())
        
        # Calculate probabilities
        self.probabilities = {
            outcome: count / total 
            for outcome, count in self.counts.items()
        }
        
        # Find most/least frequent
        if self.counts:
            sorted_counts = sorted(self.counts.items(), key=lambda x: x[1])
            self.least_frequent = sorted_counts[0][0]
            self.most_frequent = sorted_counts[-1][0]
            self.unique_outcomes = len(self.counts)
        
        # Calculate entropy
        if self.probabilities:
            self.entropy = -sum(
                p * np.log2(p) for p in self.probabilities.values() if p > 0
            )
    
    def get_top_outcomes(self, n: int = 5) -> List[Tuple[str, int, float]]:
        """Get top N most frequent outcomes."""
        sorted_items = sorted(
            self.counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [
            (outcome, count, self.probabilities.get(outcome, 0.0))
            for outcome, count in sorted_items[:n]
        ]
    
    def filter_by_probability(self, min_prob: float = 0.01) -> Dict[str, float]:
        """Filter outcomes by minimum probability."""
        return {
            outcome: prob 
            for outcome, prob in self.probabilities.items()
            if prob >= min_prob
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_shots': self.total_shots,
            'successful_shots': self.successful_shots,
            'failed_shots': self.failed_shots,
            'counts': self.counts,
            'probabilities': self.probabilities,
            'most_frequent': self.most_frequent,
            'least_frequent': self.least_frequent,
            'entropy': self.entropy,
            'unique_outcomes': self.unique_outcomes,
            'total_execution_time': self.total_execution_time,
            'average_shot_time': self.average_shot_time,
            'circuit_id': self.circuit_id,
            'backend_name': self.backend_name,
            'timestamp': self.timestamp.isoformat(),
            'hardware_metadata': self.hardware_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AggregatedResult':
        """Create from dictionary."""
        # Convert timestamp back
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)


class ResultAggregator:
    """Aggregates results from multiple shots or partial executions."""
    
    def __init__(self):
        self.partial_results: List[HardwareResult] = []
        self.shot_results: List[ShotResult] = []
    
    def aggregate_hardware_results(self, 
                                 results: List[HardwareResult],
                                 circuit_id: Optional[str] = None) -> AggregatedResult:
        """Aggregate multiple hardware results into a single result."""
        
        if not results:
            raise ExecutionError("No results to aggregate")
        
        try:
            # Combine counts from all results
            combined_counts = Counter()
            total_shots = 0
            successful_shots = 0
            failed_shots = 0
            total_time = 0.0
            
            backend_name = results[0].job_handle.backend_name
            hardware_metadata = {}
            
            for result in results:
                if result.status == JobStatus.COMPLETED:
                    successful_shots += result.shots
                    if result.counts:
                        combined_counts.update(result.counts)
                    if result.execution_time:
                        total_time += result.execution_time
                else:
                    failed_shots += result.shots
                
                total_shots += result.shots
                
                # Merge metadata
                if result.metadata:
                    hardware_metadata.update(result.metadata)
            
            # Calculate average shot time
            avg_shot_time = (total_time * 1000) / successful_shots if successful_shots > 0 else None
            
            return AggregatedResult(
                total_shots=total_shots,
                successful_shots=successful_shots,
                failed_shots=failed_shots,
                counts=dict(combined_counts),
                total_execution_time=total_time,
                average_shot_time=avg_shot_time,
                circuit_id=circuit_id,
                backend_name=backend_name,
                hardware_metadata=hardware_metadata
            )
        except Exception as e:
            raise ExecutionError(f"Failed to aggregate hardware results: {e}")
    
    def aggregate_shot_results(self, 
                             shot_results: List[ShotResult],
                             circuit_id: Optional[str] = None,
                             backend_name: Optional[str] = None) -> AggregatedResult:
        """Aggregate individual shot results."""
        
        if not shot_results:
            raise ExecutionError("No shot results to aggregate")
        
        try:
            # Count outcomes
            counts = Counter(shot.outcome for shot in shot_results)
            
            # Calculate timing statistics
            shot_times = [shot.execution_time for shot in shot_results 
                         if shot.execution_time is not None]
            
            total_time = sum(shot_times) if shot_times else None
            avg_shot_time = np.mean(shot_times) if shot_times else None
            
            return AggregatedResult(
                total_shots=len(shot_results),
                successful_shots=len([s for s in shot_results if s.outcome]),
                failed_shots=len([s for s in shot_results if not s.outcome]),
                counts=dict(counts),
                total_execution_time=total_time / 1000 if total_time else None,  # Convert to ms
                average_shot_time=avg_shot_time,
                circuit_id=circuit_id,
                backend_name=backend_name,
                raw_shots=shot_results
            )
        except Exception as e:
            raise ExecutionError(f"Failed to aggregate shot results: {e}")
    
    def merge_aggregated_results(self, 
                               results: List[AggregatedResult]) -> AggregatedResult:
        """Merge multiple aggregated results."""
        
        if not results:
            raise ExecutionError("No aggregated results to merge")
        
        try:
            # Combine counts
            combined_counts = Counter()
            total_shots = 0
            successful_shots = 0
            failed_shots = 0
            total_time = 0.0
            
            for result in results:
                combined_counts.update(result.counts)
                total_shots += result.total_shots
                successful_shots += result.successful_shots
                failed_shots += result.failed_shots
                
                if result.total_execution_time:
                    total_time += result.total_execution_time
            
            # Use metadata from first result
            base_result = results[0]
            
            return AggregatedResult(
                total_shots=total_shots,
                successful_shots=successful_shots,
                failed_shots=failed_shots,
                counts=dict(combined_counts),
                total_execution_time=total_time,
                circuit_id=base_result.circuit_id,
                backend_name=base_result.backend_name,
                hardware_metadata=base_result.hardware_metadata
            )
        except Exception as e:
            raise ExecutionError(f"Failed to merge aggregated results: {e}")


class ResultAnalyzer:
    """Provides analysis tools for quantum execution results."""
    
    @staticmethod
    def compare_results(result1: AggregatedResult, 
                       result2: AggregatedResult) -> Dict[str, Any]:
        """Compare two aggregated results."""
        
        # Calculate statistical distance
        all_outcomes = set(result1.probabilities.keys()) | set(result2.probabilities.keys())
        
        total_variation = 0.5 * sum(
            abs(result1.probabilities.get(outcome, 0) - 
                result2.probabilities.get(outcome, 0))
            for outcome in all_outcomes
        )
        
        # Calculate overlap
        overlap = sum(
            min(result1.probabilities.get(outcome, 0),
                result2.probabilities.get(outcome, 0))
            for outcome in all_outcomes
        )
        
        # Common outcomes
        common_outcomes = set(result1.counts.keys()) & set(result2.counts.keys())
        
        return {
            'total_variation_distance': total_variation,
            'overlap': overlap,
            'common_outcomes': len(common_outcomes),
            'unique_to_result1': len(set(result1.counts.keys()) - set(result2.counts.keys())),
            'unique_to_result2': len(set(result2.counts.keys()) - set(result1.counts.keys())),
            'entropy_difference': abs(result1.entropy - result2.entropy) if result1.entropy and result2.entropy else None
        }
    
    @staticmethod
    def calculate_expectation_value(result: AggregatedResult,
                                  observable: Dict[str, float]) -> float:
        """Calculate expectation value of an observable."""
        
        expectation = 0.0
        
        for outcome, probability in result.probabilities.items():
            if outcome in observable:
                expectation += probability * observable[outcome]
        
        return expectation
    
    @staticmethod
    def estimate_sampling_error(result: AggregatedResult,
                              outcome: str) -> float:
        """Estimate sampling error for a specific outcome."""
        
        p = result.probabilities.get(outcome, 0.0)
        n = result.successful_shots
        
        if n > 0 and 0 < p < 1:
            # Standard error for binomial distribution
            return np.sqrt(p * (1 - p) / n)
        
        return 0.0
    
    @staticmethod
    def detect_bias(result: AggregatedResult,
                   expected_uniform: bool = False) -> Dict[str, Any]:
        """Detect potential bias in results."""
        
        n_qubits = len(list(result.counts.keys())[0]) if result.counts else 0
        expected_outcomes = 2 ** n_qubits
        
        bias_metrics = {
            'n_qubits': n_qubits,
            'expected_outcomes': expected_outcomes,
            'observed_outcomes': result.unique_outcomes,
            'coverage': result.unique_outcomes / expected_outcomes if expected_outcomes > 0 else 0
        }
        
        if expected_uniform and result.probabilities:
            # Test for uniformity
            expected_prob = 1.0 / expected_outcomes
            chi_squared = sum(
                result.successful_shots * (prob - expected_prob) ** 2 / expected_prob
                for prob in result.probabilities.values()
            )
            bias_metrics['chi_squared'] = chi_squared
            bias_metrics['expected_uniform'] = expected_uniform
        
        return bias_metrics


class ResultStorage:
    """Handles storage and retrieval of execution results."""
    
    def __init__(self, storage_dir: str = "results"):
        self.storage_dir = storage_dir
        import os
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_result(self, result: AggregatedResult, 
                   result_id: Optional[str] = None) -> str:
        """Save an aggregated result to storage."""
        
        try:
            if not result_id:
                result_id = f"result_{uuid.uuid4().hex[:8]}_{int(time.time())}"
            
            filename = f"{self.storage_dir}/{result_id}.json"
            
            # Save result data
            result_data = {
                'result_id': result_id,
                'saved_at': datetime.now().isoformat(),
                'result': result.to_dict()
            }
            
            with open(filename, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            return result_id
        except Exception as e:
            raise ExecutionError(f"Failed to save result: {e}")
    
    def load_result(self, result_id: str) -> AggregatedResult:
        """Load an aggregated result from storage."""
        
        try:
            filename = f"{self.storage_dir}/{result_id}.json"
            
            with open(filename, 'r') as f:
                result_data = json.load(f)
            
            return AggregatedResult.from_dict(result_data['result'])
        except Exception as e:
            raise ExecutionError(f"Failed to load result {result_id}: {e}")
    
    def list_results(self) -> List[Dict[str, Any]]:
        """List all stored results."""
        
        import os
        import glob
        
        results = []
        pattern = f"{self.storage_dir}/*.json"
        
        for filename in glob.glob(pattern):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                result_info = {
                    'result_id': data['result_id'],
                    'saved_at': data['saved_at'],
                    'circuit_id': data['result'].get('circuit_id'),
                    'backend_name': data['result'].get('backend_name'),
                    'total_shots': data['result'].get('total_shots'),
                    'unique_outcomes': data['result'].get('unique_outcomes')
                }
                
                results.append(result_info)
                
            except Exception:
                continue  # Skip corrupted files
        
        return sorted(results, key=lambda x: x['saved_at'], reverse=True)
    
    def delete_result(self, result_id: str) -> bool:
        """Delete a stored result."""
        
        import os
        filename = f"{self.storage_dir}/{result_id}.json"
        
        try:
            os.remove(filename)
            return True
        except OSError:
            return False


class MultiShotExecutor:
    """Handles multi-shot execution with result aggregation."""
    
    def __init__(self, aggregator: Optional[ResultAggregator] = None,
                 storage: Optional[ResultStorage] = None):
        self.aggregator = aggregator or ResultAggregator()
        self.storage = storage or ResultStorage()
        self.analyzer = ResultAnalyzer()
    
    def execute_with_aggregation(self,
                               backend,
                               circuit,
                               total_shots: int,
                               max_shots_per_job: Optional[int] = None,
                               store_result: bool = True) -> AggregatedResult:
        """Execute circuit with automatic result aggregation."""
        
        try:
            if max_shots_per_job and total_shots > max_shots_per_job:
                # Split into multiple jobs
                return self._execute_split_jobs(
                    backend, circuit, total_shots, max_shots_per_job, store_result
                )
            else:
                # Single job execution
                return self._execute_single_job(
                    backend, circuit, total_shots, store_result
                )
        except Exception as e:
            raise ExecutionError(f"Failed to execute with aggregation: {e}")
    
    def _execute_single_job(self, backend, circuit, shots: int, 
                          store_result: bool) -> AggregatedResult:
        """Execute as a single job."""
        
        # Submit job
        job_handle = backend.submit_circuit(circuit, shots=shots)
        
        # Wait for completion
        result = backend.submit_and_wait(circuit, shots=shots, timeout=300)
        
        # Create aggregated result
        aggregated = self.aggregator.aggregate_hardware_results(
            [result], circuit_id=getattr(circuit, 'name', None)
        )
        
        # Store if requested
        if store_result:
            result_id = self.storage.save_result(aggregated)
            aggregated.hardware_metadata['result_id'] = result_id
        
        return aggregated
    
    def _execute_split_jobs(self, backend, circuit, total_shots: int,
                          max_shots_per_job: int, store_result: bool) -> AggregatedResult:
        """Execute as multiple jobs and aggregate."""
        
        results = []
        remaining_shots = total_shots
        
        while remaining_shots > 0:
            current_shots = min(remaining_shots, max_shots_per_job)
            
            # Execute job
            result = backend.submit_and_wait(
                circuit, shots=current_shots, timeout=300
            )
            results.append(result)
            
            remaining_shots -= current_shots
        
        # Aggregate all results
        aggregated = self.aggregator.aggregate_hardware_results(
            results, circuit_id=getattr(circuit, 'name', None)
        )
        
        # Store if requested
        if store_result:
            result_id = self.storage.save_result(aggregated)
            aggregated.hardware_metadata['result_id'] = result_id
        
        return aggregated
    
    def get_execution_summary(self, result: AggregatedResult) -> str:
        """Generate a human-readable execution summary."""
        
        summary_lines = [
            f"Quantum Execution Summary",
            f"=" * 40,
            f"Total shots: {result.total_shots:,}",
            f"Successful: {result.successful_shots:,} ({result.successful_shots/result.total_shots*100:.1f}%)",
            f"Failed: {result.failed_shots:,}",
            f"Unique outcomes: {result.unique_outcomes}",
        ]
        
        if result.entropy is not None:
            summary_lines.append(f"Entropy: {result.entropy:.3f} bits")
        
        if result.total_execution_time:
            summary_lines.append(f"Execution time: {result.total_execution_time:.1f} ms")
        
        if result.average_shot_time:
            summary_lines.append(f"Average per shot: {result.average_shot_time:.1f} μs")
        
        # Top outcomes
        top_outcomes = result.get_top_outcomes(5)
        if top_outcomes:
            summary_lines.append(f"\nTop outcomes:")
            for outcome, count, prob in top_outcomes:
                summary_lines.append(f"  |{outcome}⟩: {count:,} ({prob*100:.1f}%)")
        
        return "\n".join(summary_lines)


# Global instances
_result_aggregator = ResultAggregator()
_result_analyzer = ResultAnalyzer()
_result_storage = ResultStorage()
_multi_shot_executor = MultiShotExecutor()


def get_result_aggregator() -> ResultAggregator:
    """Get the global result aggregator."""
    return _result_aggregator


def get_result_analyzer() -> ResultAnalyzer:
    """Get the global result analyzer."""
    return _result_analyzer


def get_result_storage() -> ResultStorage:
    """Get the global result storage."""
    return _result_storage


def get_multi_shot_executor() -> MultiShotExecutor:
    """Get the global multi-shot executor."""
    return _multi_shot_executor 