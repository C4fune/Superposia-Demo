"""
Experiment Analyzer

This module provides comprehensive analytics and comparison capabilities
for quantum experiments, including statistical analysis, trend detection,
and performance comparisons.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

from .database import ExperimentDatabase
from .models import Experiment, ExperimentResult, ComparisonResult


class ExperimentAnalyzer:
    """
    Comprehensive experiment analysis and comparison engine.
    
    This class provides advanced analytics capabilities for quantum experiments,
    including statistical analysis, trend detection, and performance comparisons.
    """
    
    def __init__(self, database: ExperimentDatabase):
        """
        Initialize the experiment analyzer.
        
        Args:
            database: Experiment database instance
        """
        self.database = database
        self.logger = logging.getLogger(__name__)
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a single experiment.
        
        Args:
            experiment_id: ID of the experiment to analyze
            
        Returns:
            Analysis results including statistics, trends, and insights
        """
        experiment = self.database.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        results = self.database.get_results(experiment_id)
        if not results:
            return {"experiment_id": experiment_id, "analysis": "No results available"}
        
        analysis = {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "experiment_type": experiment.experiment_type,
            "total_runs": len(results),
            "statistics": self._calculate_statistics(results),
            "performance_metrics": self._calculate_performance_metrics(results),
            "result_distribution": self._analyze_result_distribution(results),
            "quality_metrics": self._calculate_quality_metrics(results),
            "anomalies": self._detect_anomalies(results),
            "recommendations": self._generate_recommendations(experiment, results)
        }
        
        # Add parameter analysis if applicable
        if experiment.experiment_type == "parameter_sweep":
            analysis["parameter_analysis"] = self._analyze_parameter_sweep(results)
        
        return analysis
    
    def compare_experiments(self, experiment_id1: str, experiment_id2: str) -> ComparisonResult:
        """
        Compare two experiments statistically.
        
        Args:
            experiment_id1: First experiment ID
            experiment_id2: Second experiment ID
            
        Returns:
            Comparison result with statistical analysis
        """
        exp1 = self.database.get_experiment(experiment_id1)
        exp2 = self.database.get_experiment(experiment_id2)
        
        if not exp1 or not exp2:
            raise ValueError("One or both experiments not found")
        
        results1 = self.database.get_results(experiment_id1)
        results2 = self.database.get_results(experiment_id2)
        
        if not results1 or not results2:
            raise ValueError("One or both experiments have no results")
        
        # Calculate basic statistics
        stats1 = self._calculate_statistics(results1)
        stats2 = self._calculate_statistics(results2)
        
        # Perform statistical tests
        comparison = ComparisonResult(
            experiment1_id=experiment_id1,
            experiment2_id=experiment_id2,
            comparison_type="statistical"
        )
        
        # Compare fidelity
        if stats1["fidelity"]["mean"] and stats2["fidelity"]["mean"]:
            comparison.fidelity_difference = stats1["fidelity"]["mean"] - stats2["fidelity"]["mean"]
        
        # Compare execution times
        if stats1["execution_time"]["mean"] and stats2["execution_time"]["mean"]:
            comparison.execution_time_difference = (
                stats1["execution_time"]["mean"] - stats2["execution_time"]["mean"]
            )
        
        # Compare success rates
        comparison.success_rate_difference = stats1["success_rate"] - stats2["success_rate"]
        
        # Perform significance testing
        if self._has_sufficient_data(results1, results2):
            comparison.p_value = self._perform_t_test(results1, results2)
            comparison.is_significant = comparison.p_value < 0.05 if comparison.p_value else False
        
        # Detailed analysis
        comparison.detailed_analysis = {
            "experiment1_stats": stats1,
            "experiment2_stats": stats2,
            "effect_size": self._calculate_effect_size(results1, results2),
            "confidence_intervals": self._calculate_confidence_intervals(results1, results2),
            "distribution_comparison": self._compare_distributions(results1, results2)
        }
        
        return comparison
    
    def analyze_trends(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze trends across multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to analyze
            
        Returns:
            Trend analysis results
        """
        experiments = []
        all_results = []
        
        for exp_id in experiment_ids:
            exp = self.database.get_experiment(exp_id)
            if exp:
                experiments.append(exp)
                results = self.database.get_results(exp_id)
                all_results.extend(results)
        
        if not experiments:
            return {"trend_analysis": "No valid experiments found"}
        
        # Sort experiments by creation date
        experiments.sort(key=lambda x: x.created_at)
        
        trend_analysis = {
            "experiment_count": len(experiments),
            "time_range": {
                "start": experiments[0].created_at.isoformat(),
                "end": experiments[-1].created_at.isoformat()
            },
            "performance_trends": self._analyze_performance_trends(experiments),
            "quality_trends": self._analyze_quality_trends(experiments),
            "backend_analysis": self._analyze_backend_trends(experiments),
            "temporal_patterns": self._analyze_temporal_patterns(experiments),
            "recommendations": self._generate_trend_recommendations(experiments)
        }
        
        return trend_analysis
    
    def detect_performance_issues(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        Detect performance issues in an experiment.
        
        Args:
            experiment_id: ID of the experiment to analyze
            
        Returns:
            List of detected issues with severity and recommendations
        """
        experiment = self.database.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        results = self.database.get_results(experiment_id)
        if not results:
            return []
        
        issues = []
        
        # Check for high failure rate
        failed_results = [r for r in results if r.status == "failed"]
        failure_rate = len(failed_results) / len(results)
        
        if failure_rate > 0.1:  # More than 10% failure rate
            issues.append({
                "type": "high_failure_rate",
                "severity": "high" if failure_rate > 0.3 else "medium",
                "value": failure_rate,
                "description": f"High failure rate: {failure_rate:.1%}",
                "recommendation": "Check circuit validity and backend connectivity"
            })
        
        # Check for poor fidelity
        fidelities = [r.fidelity for r in results if r.fidelity is not None]
        if fidelities:
            avg_fidelity = np.mean(fidelities)
            if avg_fidelity < 0.8:  # Less than 80% fidelity
                issues.append({
                    "type": "low_fidelity",
                    "severity": "high" if avg_fidelity < 0.6 else "medium",
                    "value": avg_fidelity,
                    "description": f"Low average fidelity: {avg_fidelity:.3f}",
                    "recommendation": "Consider noise mitigation or circuit optimization"
                })
        
        # Check for slow execution
        execution_times = [r.execution_time for r in results if r.execution_time is not None]
        if execution_times:
            avg_time = np.mean(execution_times)
            if avg_time > 30000:  # More than 30 seconds
                issues.append({
                    "type": "slow_execution",
                    "severity": "medium",
                    "value": avg_time,
                    "description": f"Slow average execution: {avg_time:.1f}ms",
                    "recommendation": "Consider backend optimization or circuit reduction"
                })
        
        # Check for high variance
        if fidelities and len(fidelities) > 1:
            fidelity_std = np.std(fidelities)
            if fidelity_std > 0.2:  # High variance in fidelity
                issues.append({
                    "type": "high_variance",
                    "severity": "medium",
                    "value": fidelity_std,
                    "description": f"High fidelity variance: {fidelity_std:.3f}",
                    "recommendation": "Investigate noise sources or calibration issues"
                })
        
        return issues
    
    def generate_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive report for an experiment.
        
        Args:
            experiment_id: ID of the experiment to report on
            
        Returns:
            Comprehensive experiment report
        """
        experiment = self.database.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        circuit = self.database.get_circuit(experiment.circuit_id)
        results = self.database.get_results(experiment_id)
        
        report = {
            "experiment_info": {
                "id": experiment.id,
                "name": experiment.name,
                "description": experiment.description,
                "type": experiment.experiment_type,
                "status": experiment.status,
                "created_at": experiment.created_at.isoformat(),
                "completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None
            },
            "circuit_info": {
                "id": circuit.id if circuit else None,
                "name": circuit.name if circuit else None,
                "num_qubits": circuit.num_qubits if circuit else None,
                "depth": circuit.depth if circuit else None
            },
            "execution_info": {
                "backend": experiment.backend,
                "provider": experiment.provider,
                "device_name": experiment.device_name,
                "shots": experiment.shots,
                "total_runs": len(results),
                "successful_runs": len([r for r in results if r.status == "completed"]),
                "failed_runs": len([r for r in results if r.status == "failed"])
            },
            "analysis": self.analyze_experiment(experiment_id),
            "performance_issues": self.detect_performance_issues(experiment_id),
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def _calculate_statistics(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Calculate basic statistics for experiment results."""
        stats = {
            "total_runs": len(results),
            "successful_runs": len([r for r in results if r.status == "completed"]),
            "failed_runs": len([r for r in results if r.status == "failed"]),
            "success_rate": 0.0,
            "fidelity": {"mean": None, "std": None, "min": None, "max": None},
            "execution_time": {"mean": None, "std": None, "min": None, "max": None},
            "expectation_value": {"mean": None, "std": None, "min": None, "max": None}
        }
        
        if len(results) > 0:
            stats["success_rate"] = stats["successful_runs"] / len(results)
        
        # Calculate fidelity statistics
        fidelities = [r.fidelity for r in results if r.fidelity is not None]
        if fidelities:
            stats["fidelity"] = {
                "mean": np.mean(fidelities),
                "std": np.std(fidelities),
                "min": np.min(fidelities),
                "max": np.max(fidelities)
            }
        
        # Calculate execution time statistics
        exec_times = [r.execution_time for r in results if r.execution_time is not None]
        if exec_times:
            stats["execution_time"] = {
                "mean": np.mean(exec_times),
                "std": np.std(exec_times),
                "min": np.min(exec_times),
                "max": np.max(exec_times)
            }
        
        # Calculate expectation value statistics
        exp_values = [r.expectation_value for r in results if r.expectation_value is not None]
        if exp_values:
            stats["expectation_value"] = {
                "mean": np.mean(exp_values),
                "std": np.std(exp_values),
                "min": np.min(exp_values),
                "max": np.max(exp_values)
            }
        
        return stats
    
    def _calculate_performance_metrics(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Calculate performance metrics for experiment results."""
        metrics = {
            "throughput": 0.0,  # Results per second
            "reliability": 0.0,  # Success rate
            "efficiency": 0.0,  # Results per unit time
            "consistency": 0.0  # 1 - coefficient of variation
        }
        
        if not results:
            return metrics
        
        # Calculate throughput
        completed_results = [r for r in results if r.status == "completed"]
        if completed_results:
            total_time = sum(r.execution_time for r in completed_results if r.execution_time)
            if total_time > 0:
                metrics["throughput"] = len(completed_results) / (total_time / 1000)  # Convert ms to s
        
        # Calculate reliability
        metrics["reliability"] = len(completed_results) / len(results)
        
        # Calculate efficiency
        if completed_results:
            avg_time = np.mean([r.execution_time for r in completed_results if r.execution_time])
            if avg_time > 0:
                metrics["efficiency"] = 1 / avg_time  # Inverse of average time
        
        # Calculate consistency
        fidelities = [r.fidelity for r in completed_results if r.fidelity is not None]
        if len(fidelities) > 1:
            mean_fidelity = np.mean(fidelities)
            std_fidelity = np.std(fidelities)
            if mean_fidelity > 0:
                cv = std_fidelity / mean_fidelity
                metrics["consistency"] = max(0, 1 - cv)
        
        return metrics
    
    def _analyze_result_distribution(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze the distribution of experiment results."""
        distribution = {
            "count_distribution": {},
            "outcome_frequency": {},
            "entropy": 0.0,
            "uniformity": 0.0
        }
        
        if not results:
            return distribution
        
        # Analyze count distributions
        all_counts = {}
        for result in results:
            if result.raw_counts:
                for state, count in result.raw_counts.items():
                    all_counts[state] = all_counts.get(state, 0) + count
        
        if all_counts:
            total_counts = sum(all_counts.values())
            distribution["count_distribution"] = {
                state: count / total_counts for state, count in all_counts.items()
            }
            
            # Calculate entropy
            probabilities = list(distribution["count_distribution"].values())
            distribution["entropy"] = -sum(p * np.log2(p) for p in probabilities if p > 0)
            
            # Calculate uniformity (how close to uniform distribution)
            expected_prob = 1.0 / len(all_counts)
            distribution["uniformity"] = 1 - np.sum(np.abs(np.array(probabilities) - expected_prob)) / 2
        
        return distribution
    
    def _calculate_quality_metrics(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Calculate quality metrics for experiment results."""
        quality = {
            "average_fidelity": 0.0,
            "fidelity_consistency": 0.0,
            "measurement_quality": 0.0,
            "error_rate": 0.0
        }
        
        if not results:
            return quality
        
        # Calculate average fidelity
        fidelities = [r.fidelity for r in results if r.fidelity is not None]
        if fidelities:
            quality["average_fidelity"] = np.mean(fidelities)
            
            # Calculate fidelity consistency
            if len(fidelities) > 1:
                quality["fidelity_consistency"] = 1 - (np.std(fidelities) / np.mean(fidelities))
        
        # Calculate error rate
        total_results = len(results)
        failed_results = len([r for r in results if r.status == "failed"])
        quality["error_rate"] = failed_results / total_results if total_results > 0 else 0
        
        # Calculate measurement quality (based on shot consistency)
        shot_variances = []
        for result in results:
            if result.raw_counts and len(result.raw_counts) > 1:
                counts = list(result.raw_counts.values())
                expected_count = sum(counts) / len(counts)
                variance = np.var(counts) / expected_count if expected_count > 0 else 0
                shot_variances.append(variance)
        
        if shot_variances:
            quality["measurement_quality"] = 1 - np.mean(shot_variances)
        
        return quality
    
    def _detect_anomalies(self, results: List[ExperimentResult]) -> List[Dict[str, Any]]:
        """Detect anomalies in experiment results."""
        anomalies = []
        
        if len(results) < 3:  # Need at least 3 results for anomaly detection
            return anomalies
        
        # Check for fidelity anomalies
        fidelities = [r.fidelity for r in results if r.fidelity is not None]
        if len(fidelities) >= 3:
            mean_fidelity = np.mean(fidelities)
            std_fidelity = np.std(fidelities)
            threshold = 2 * std_fidelity  # 2-sigma threshold
            
            for i, result in enumerate(results):
                if result.fidelity is not None:
                    if abs(result.fidelity - mean_fidelity) > threshold:
                        anomalies.append({
                            "type": "fidelity_anomaly",
                            "run_number": result.run_number,
                            "value": result.fidelity,
                            "expected": mean_fidelity,
                            "deviation": abs(result.fidelity - mean_fidelity)
                        })
        
        # Check for execution time anomalies
        exec_times = [r.execution_time for r in results if r.execution_time is not None]
        if len(exec_times) >= 3:
            mean_time = np.mean(exec_times)
            std_time = np.std(exec_times)
            threshold = 3 * std_time  # 3-sigma threshold for execution time
            
            for result in results:
                if result.execution_time is not None:
                    if abs(result.execution_time - mean_time) > threshold:
                        anomalies.append({
                            "type": "execution_time_anomaly",
                            "run_number": result.run_number,
                            "value": result.execution_time,
                            "expected": mean_time,
                            "deviation": abs(result.execution_time - mean_time)
                        })
        
        return anomalies
    
    def _generate_recommendations(self, experiment: Experiment, 
                                results: List[ExperimentResult]) -> List[str]:
        """Generate recommendations for experiment optimization."""
        recommendations = []
        
        if not results:
            return ["No results available for analysis"]
        
        # Analyze success rate
        success_rate = len([r for r in results if r.status == "completed"]) / len(results)
        if success_rate < 0.9:
            recommendations.append(
                f"Success rate is {success_rate:.1%}. Consider checking circuit validity and backend connectivity."
            )
        
        # Analyze fidelity
        fidelities = [r.fidelity for r in results if r.fidelity is not None]
        if fidelities:
            avg_fidelity = np.mean(fidelities)
            if avg_fidelity < 0.8:
                recommendations.append(
                    f"Average fidelity is {avg_fidelity:.3f}. Consider noise mitigation techniques."
                )
        
        # Analyze execution time
        exec_times = [r.execution_time for r in results if r.execution_time is not None]
        if exec_times:
            avg_time = np.mean(exec_times)
            if avg_time > 30000:  # 30 seconds
                recommendations.append(
                    f"Average execution time is {avg_time:.1f}ms. Consider circuit optimization."
                )
        
        # Analyze variance
        if len(fidelities) > 1:
            fidelity_std = np.std(fidelities)
            if fidelity_std > 0.1:
                recommendations.append(
                    f"High fidelity variance ({fidelity_std:.3f}). Consider calibration improvements."
                )
        
        return recommendations
    
    def _analyze_parameter_sweep(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze parameter sweep results."""
        analysis = {
            "parameter_correlations": {},
            "optimal_parameters": None,
            "parameter_sensitivity": {},
            "optimization_surface": []
        }
        
        # Extract parameter values and corresponding metrics
        param_results = []
        for result in results:
            if result.parameter_values and result.fidelity is not None:
                param_results.append({
                    "parameters": result.parameter_values,
                    "fidelity": result.fidelity,
                    "expectation_value": result.expectation_value
                })
        
        if not param_results:
            return analysis
        
        # Find optimal parameters
        best_result = max(param_results, key=lambda x: x["fidelity"])
        analysis["optimal_parameters"] = best_result["parameters"]
        
        # Calculate parameter correlations
        if len(param_results) > 2:
            param_names = list(param_results[0]["parameters"].keys())
            for param_name in param_names:
                param_values = [r["parameters"][param_name] for r in param_results]
                fidelities = [r["fidelity"] for r in param_results]
                
                if len(set(param_values)) > 1:  # Only if parameter varies
                    correlation = np.corrcoef(param_values, fidelities)[0, 1]
                    analysis["parameter_correlations"][param_name] = correlation
        
        return analysis
    
    def _has_sufficient_data(self, results1: List[ExperimentResult], 
                           results2: List[ExperimentResult]) -> bool:
        """Check if there's sufficient data for statistical testing."""
        return len(results1) >= 3 and len(results2) >= 3
    
    def _perform_t_test(self, results1: List[ExperimentResult], 
                       results2: List[ExperimentResult]) -> Optional[float]:
        """Perform t-test between two result sets."""
        try:
            from scipy import stats
            
            fidelities1 = [r.fidelity for r in results1 if r.fidelity is not None]
            fidelities2 = [r.fidelity for r in results2 if r.fidelity is not None]
            
            if len(fidelities1) >= 3 and len(fidelities2) >= 3:
                t_stat, p_value = stats.ttest_ind(fidelities1, fidelities2)
                return p_value
            
        except ImportError:
            # Fallback if scipy not available
            pass
        
        return None
    
    def _calculate_effect_size(self, results1: List[ExperimentResult], 
                             results2: List[ExperimentResult]) -> Optional[float]:
        """Calculate Cohen's d effect size."""
        fidelities1 = [r.fidelity for r in results1 if r.fidelity is not None]
        fidelities2 = [r.fidelity for r in results2 if r.fidelity is not None]
        
        if len(fidelities1) >= 2 and len(fidelities2) >= 2:
            mean1, mean2 = np.mean(fidelities1), np.mean(fidelities2)
            std1, std2 = np.std(fidelities1), np.std(fidelities2)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((len(fidelities1) - 1) * std1**2 + 
                                (len(fidelities2) - 1) * std2**2) / 
                               (len(fidelities1) + len(fidelities2) - 2))
            
            if pooled_std > 0:
                return (mean1 - mean2) / pooled_std
        
        return None
    
    def _calculate_confidence_intervals(self, results1: List[ExperimentResult], 
                                      results2: List[ExperimentResult]) -> Dict[str, Any]:
        """Calculate confidence intervals for the results."""
        intervals = {}
        
        for name, results in [("experiment1", results1), ("experiment2", results2)]:
            fidelities = [r.fidelity for r in results if r.fidelity is not None]
            if len(fidelities) >= 2:
                mean = np.mean(fidelities)
                std = np.std(fidelities)
                n = len(fidelities)
                
                # 95% confidence interval
                margin = 1.96 * std / np.sqrt(n)
                intervals[name] = {
                    "mean": mean,
                    "lower": mean - margin,
                    "upper": mean + margin
                }
        
        return intervals
    
    def _compare_distributions(self, results1: List[ExperimentResult], 
                             results2: List[ExperimentResult]) -> Dict[str, Any]:
        """Compare the distributions of two result sets."""
        comparison = {
            "distribution_difference": None,
            "variance_ratio": None,
            "range_comparison": None
        }
        
        fidelities1 = [r.fidelity for r in results1 if r.fidelity is not None]
        fidelities2 = [r.fidelity for r in results2 if r.fidelity is not None]
        
        if len(fidelities1) >= 2 and len(fidelities2) >= 2:
            var1, var2 = np.var(fidelities1), np.var(fidelities2)
            comparison["variance_ratio"] = var1 / var2 if var2 > 0 else None
            
            range1 = np.max(fidelities1) - np.min(fidelities1)
            range2 = np.max(fidelities2) - np.min(fidelities2)
            comparison["range_comparison"] = {
                "range1": range1,
                "range2": range2,
                "ratio": range1 / range2 if range2 > 0 else None
            }
        
        return comparison
    
    def _analyze_performance_trends(self, experiments: List[Experiment]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        trends = {
            "fidelity_trend": "stable",
            "execution_time_trend": "stable",
            "success_rate_trend": "stable",
            "trend_data": []
        }
        
        for experiment in experiments:
            results = self.database.get_results(experiment.id)
            if results:
                stats = self._calculate_statistics(results)
                trends["trend_data"].append({
                    "date": experiment.created_at.isoformat(),
                    "fidelity": stats["fidelity"]["mean"],
                    "execution_time": stats["execution_time"]["mean"],
                    "success_rate": stats["success_rate"]
                })
        
        # Simple trend analysis
        if len(trends["trend_data"]) >= 3:
            fidelities = [d["fidelity"] for d in trends["trend_data"] if d["fidelity"]]
            if len(fidelities) >= 3:
                # Simple linear trend
                x = np.arange(len(fidelities))
                slope = np.polyfit(x, fidelities, 1)[0]
                if slope > 0.01:
                    trends["fidelity_trend"] = "improving"
                elif slope < -0.01:
                    trends["fidelity_trend"] = "declining"
        
        return trends
    
    def _analyze_quality_trends(self, experiments: List[Experiment]) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        return {"quality_trend": "stable"}  # Placeholder
    
    def _analyze_backend_trends(self, experiments: List[Experiment]) -> Dict[str, Any]:
        """Analyze backend usage trends."""
        backend_usage = defaultdict(int)
        for experiment in experiments:
            backend_usage[experiment.backend] += 1
        
        return {
            "backend_usage": dict(backend_usage),
            "most_used_backend": max(backend_usage, key=backend_usage.get) if backend_usage else None
        }
    
    def _analyze_temporal_patterns(self, experiments: List[Experiment]) -> Dict[str, Any]:
        """Analyze temporal patterns in experiment execution."""
        patterns = {
            "experiments_per_day": defaultdict(int),
            "peak_hours": defaultdict(int),
            "execution_frequency": "irregular"
        }
        
        for experiment in experiments:
            date_str = experiment.created_at.strftime("%Y-%m-%d")
            patterns["experiments_per_day"][date_str] += 1
            
            hour = experiment.created_at.hour
            patterns["peak_hours"][hour] += 1
        
        return {
            "experiments_per_day": dict(patterns["experiments_per_day"]),
            "peak_hours": dict(patterns["peak_hours"]),
            "most_active_hour": max(patterns["peak_hours"], key=patterns["peak_hours"].get) if patterns["peak_hours"] else None
        }
    
    def _generate_trend_recommendations(self, experiments: List[Experiment]) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        if len(experiments) < 3:
            recommendations.append("Run more experiments to identify meaningful trends.")
        
        # Analyze backend distribution
        backends = [exp.backend for exp in experiments]
        backend_counts = defaultdict(int)
        for backend in backends:
            backend_counts[backend] += 1
        
        if len(backend_counts) == 1:
            recommendations.append("Consider testing multiple backends for comparison.")
        
        return recommendations 