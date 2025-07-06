"""
Quantum Experiments API for Next.js Integration

This module provides API endpoints that integrate the experiment tracking system
with the existing Next.js quantum platform frontend.
"""

import json
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from typing import Dict, List, Optional, Any

# Import experiment tracking system
try:
    from quantum_platform.experiments.manager import ExperimentManager
    from quantum_platform.experiments.models import ExperimentType, ExperimentStatus
    from quantum_platform.hardware.backends.local_simulator import LocalSimulatorBackend
    
    # Initialize experiment manager
    experiment_manager = ExperimentManager()
    EXPERIMENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Experiment tracking system not available: {e}")
    experiment_manager = None
    EXPERIMENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_experiment_request(endpoint_name: str):
    """Decorator to handle experiment API requests with proper error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                if not EXPERIMENTS_AVAILABLE:
                    return {
                        'success': False,
                        'error': {
                            'message': 'Experiment tracking system not available',
                            'code': 'EXPERIMENTS_UNAVAILABLE'
                        }
                    }, 503
                
                logger.info(f"Processing experiment request: {endpoint_name}")
                result = func(*args, **kwargs)
                logger.info(f"Successfully processed: {endpoint_name}")
                return result
                
            except Exception as e:
                logger.error(f"Error in {endpoint_name}: {str(e)}")
                logger.error(traceback.format_exc())
                return {
                    'success': False,
                    'error': {
                        'message': str(e),
                        'code': 'INTERNAL_ERROR',
                        'details': traceback.format_exc() if logging.getLogger().level <= logging.DEBUG else None
                    }
                }, 500
        return wrapper
    return decorator

@handle_experiment_request("health_check")
def health_check():
    """Check if experiment tracking system is healthy."""
    if not EXPERIMENTS_AVAILABLE:
        return {
            'success': False,
            'error': {'message': 'Experiment tracking not available'}
        }, 503
    
    try:
        # Test database connection
        stats = experiment_manager.get_database_stats()
        return {
            'success': True,
            'data': {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'experiments_available': True,
                'database_stats': stats
            }
        }, 200
    except Exception as e:
        return {
            'success': False,
            'error': {'message': f'Health check failed: {str(e)}'}
        }, 500

@handle_experiment_request("list_experiments")
def list_experiments():
    """List all experiments with optional filtering."""
    try:
        # Get query parameters
        request_data = request.get_json() if request.method == 'POST' else {}
        
        limit = request_data.get('limit', 50)
        offset = request_data.get('offset', 0)
        status_filter = request_data.get('status')
        backend_filter = request_data.get('backend')
        experiment_type_filter = request_data.get('experiment_type')
        
        # Build filters
        filters = {}
        if status_filter:
            filters['status_filter'] = status_filter
        if backend_filter:
            filters['backend_filter'] = backend_filter
        if experiment_type_filter:
            filters['experiment_type_filter'] = experiment_type_filter
        
        # Get experiments
        experiments = experiment_manager.list_experiments(
            limit=limit,
            offset=offset,
            **filters
        )
        
        return {
            'success': True,
            'data': {
                'experiments': [exp.to_dict() for exp in experiments],
                'pagination': {
                    'limit': limit,
                    'offset': offset,
                    'total': len(experiments)
                }
            }
        }, 200
        
    except Exception as e:
        raise Exception(f"Failed to list experiments: {str(e)}")

@handle_experiment_request("get_experiment")
def get_experiment(experiment_id: str):
    """Get detailed experiment information."""
    try:
        experiment = experiment_manager.database.get_experiment(experiment_id)
        if not experiment:
            return {
                'success': False,
                'error': {'message': 'Experiment not found'}
            }, 404
        
        # Get additional details
        results = experiment_manager.get_experiment_results(experiment_id)
        summary = experiment_manager.get_experiment_summary(experiment_id)
        
        return {
            'success': True,
            'data': {
                'experiment': experiment.to_dict(),
                'summary': summary.to_dict() if summary else None,
                'results_count': len(results),
                'recent_results': [r.to_dict() for r in results[:5]]  # Last 5 results
            }
        }, 200
        
    except Exception as e:
        raise Exception(f"Failed to get experiment: {str(e)}")

@handle_experiment_request("create_experiment")
def create_experiment():
    """Create a new experiment."""
    try:
        data = request.get_json()
        if not data:
            return {
                'success': False,
                'error': {'message': 'No data provided'}
            }, 400
        
        # Validate required fields
        required_fields = ['name', 'circuit_data']
        for field in required_fields:
            if field not in data:
                return {
                    'success': False,
                    'error': {'message': f'Missing required field: {field}'}
                }, 400
        
        # Create circuit from provided data
        circuit_data = data['circuit_data']
        circuit = experiment_manager.database.create_circuit(
            name=circuit_data.get('name', f"Circuit for {data['name']}"),
            qasm_code=circuit_data.get('qasm_code', ''),
            num_qubits=circuit_data.get('num_qubits', 1),
            description=circuit_data.get('description'),
            circuit_json=circuit_data.get('circuit_json', {}),
            parameters=circuit_data.get('parameters', {})
        )
        
        # Get backend (default to local simulator)
        backend_name = data.get('backend', 'local_simulator')
        backend = LocalSimulatorBackend()  # Default backend
        
        # Create experiment
        experiment = experiment_manager.create_experiment(
            name=data['name'],
            circuit_id=circuit.id,
            backend=backend,
            experiment_type=data.get('experiment_type', ExperimentType.SINGLE_SHOT.value),
            description=data.get('description'),
            shots=data.get('shots', 1000),
            parameter_sweep=data.get('parameter_sweep'),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )
        
        return {
            'success': True,
            'data': {
                'experiment': experiment.to_dict(),
                'circuit': circuit.to_dict()
            }
        }, 201
        
    except Exception as e:
        raise Exception(f"Failed to create experiment: {str(e)}")

@handle_experiment_request("run_experiment")
def run_experiment():
    """Run an experiment with the current circuit."""
    try:
        data = request.get_json()
        if not data:
            return {
                'success': False,
                'error': {'message': 'No data provided'}
            }, 400
        
        experiment_id = data.get('experiment_id')
        if not experiment_id:
            return {
                'success': False,
                'error': {'message': 'Missing experiment_id'}
            }, 400
        
        # Check if experiment exists
        experiment = experiment_manager.database.get_experiment(experiment_id)
        if not experiment:
            return {
                'success': False,
                'error': {'message': 'Experiment not found'}
            }, 404
        
        # Run experiment asynchronously
        async_execution = data.get('async', True)
        result = experiment_manager.run_experiment(
            experiment_id=experiment_id,
            async_execution=async_execution
        )
        
        if async_execution:
            return {
                'success': True,
                'data': {
                    'experiment_id': experiment_id,
                    'status': 'running',
                    'message': 'Experiment started successfully'
                }
            }, 200
        else:
            return {
                'success': True,
                'data': {
                    'experiment': result.to_dict(),
                    'status': 'completed'
                }
            }, 200
        
    except Exception as e:
        raise Exception(f"Failed to run experiment: {str(e)}")

@handle_experiment_request("get_experiment_results")
def get_experiment_results(experiment_id: str):
    """Get results for an experiment."""
    try:
        # Get results with pagination
        request_data = request.get_json() if request.method == 'POST' else {}
        limit = request_data.get('limit', 100)
        offset = request_data.get('offset', 0)
        
        results = experiment_manager.database.get_results(
            experiment_id, limit=limit, offset=offset
        )
        
        if not results:
            # Check if experiment exists
            experiment = experiment_manager.database.get_experiment(experiment_id)
            if not experiment:
                return {
                    'success': False,
                    'error': {'message': 'Experiment not found'}
                }, 404
        
        return {
            'success': True,
            'data': {
                'experiment_id': experiment_id,
                'results': [result.to_dict() for result in results],
                'pagination': {
                    'limit': limit,
                    'offset': offset,
                    'total': len(results)
                }
            }
        }, 200
        
    except Exception as e:
        raise Exception(f"Failed to get experiment results: {str(e)}")

@handle_experiment_request("analyze_experiment")
def analyze_experiment(experiment_id: str):
    """Get comprehensive analysis of an experiment."""
    try:
        # Check if experiment exists
        experiment = experiment_manager.database.get_experiment(experiment_id)
        if not experiment:
            return {
                'success': False,
                'error': {'message': 'Experiment not found'}
            }, 404
        
        # Get analysis
        analysis = experiment_manager.analyzer.analyze_experiment(experiment_id)
        
        # Get performance issues
        issues = experiment_manager.analyzer.detect_performance_issues(experiment_id)
        
        return {
            'success': True,
            'data': {
                'experiment_id': experiment_id,
                'analysis': analysis,
                'performance_issues': issues
            }
        }, 200
        
    except Exception as e:
        raise Exception(f"Failed to analyze experiment: {str(e)}")

@handle_experiment_request("export_experiment")
def export_experiment(experiment_id: str):
    """Export experiment data in various formats."""
    try:
        # Get format from query parameters
        format_type = request.args.get('format', 'json')
        
        if format_type not in ['json', 'csv']:
            return {
                'success': False,
                'error': {'message': 'Unsupported format. Use json or csv.'}
            }, 400
        
        # Export data
        export_data = experiment_manager.export_experiment_data(
            experiment_id, format_type
        )
        
        if format_type == 'csv':
            from flask import Response
            return Response(
                export_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=experiment_{experiment_id}.csv'}
            )
        else:
            return {
                'success': True,
                'data': {
                    'experiment_id': experiment_id,
                    'format': format_type,
                    'export_data': json.loads(export_data)
                }
            }, 200
        
    except Exception as e:
        raise Exception(f"Failed to export experiment: {str(e)}")

@handle_experiment_request("delete_experiment")
def delete_experiment(experiment_id: str):
    """Delete an experiment and all its data."""
    try:
        success = experiment_manager.delete_experiment(experiment_id)
        
        if success:
            return {
                'success': True,
                'data': {
                    'experiment_id': experiment_id,
                    'message': 'Experiment deleted successfully'
                }
            }, 200
        else:
            return {
                'success': False,
                'error': {'message': 'Experiment not found or could not be deleted'}
            }, 404
        
    except Exception as e:
        raise Exception(f"Failed to delete experiment: {str(e)}")

@handle_experiment_request("get_database_stats")
def get_database_stats():
    """Get database statistics and overview."""
    try:
        stats = experiment_manager.get_database_stats()
        
        # Get active experiments
        active_experiments = experiment_manager.get_active_experiments()
        
        return {
            'success': True,
            'data': {
                'database_stats': stats,
                'active_experiments': len(active_experiments),
                'active_experiment_details': active_experiments
            }
        }, 200
        
    except Exception as e:
        raise Exception(f"Failed to get database stats: {str(e)}")

@handle_experiment_request("compare_experiments")
def compare_experiments():
    """Compare two experiments statistically."""
    try:
        data = request.get_json()
        if not data or 'experiment_ids' not in data:
            return {
                'success': False,
                'error': {'message': 'Missing experiment_ids'}
            }, 400
        
        experiment_ids = data['experiment_ids']
        if len(experiment_ids) != 2:
            return {
                'success': False,
                'error': {'message': 'Exactly 2 experiment IDs required'}
            }, 400
        
        # Perform comparison
        comparison = experiment_manager.compare_experiments(
            experiment_ids[0], experiment_ids[1]
        )
        
        return {
            'success': True,
            'data': {
                'comparison': comparison.to_dict()
            }
        }, 200
        
    except Exception as e:
        raise Exception(f"Failed to compare experiments: {str(e)}")

# Integration function for Next.js API routes
def handle_experiments_api(method: str, path: str, **kwargs):
    """
    Main handler for experiments API that integrates with Next.js API routes.
    
    Args:
        method: HTTP method (GET, POST, DELETE, etc.)
        path: API path after /api/quantum/experiments/
        **kwargs: Additional parameters like experiment_id
    
    Returns:
        Tuple of (response_data, status_code)
    """
    
    # Map API paths to functions
    api_routes = {
        'health': {
            'GET': health_check
        },
        'list': {
            'GET': list_experiments,
            'POST': list_experiments
        },
        'stats': {
            'GET': get_database_stats
        },
        'compare': {
            'POST': compare_experiments
        },
        'create': {
            'POST': create_experiment
        },
        'run': {
            'POST': run_experiment
        }
    }
    
    # Handle experiment-specific routes (with experiment_id)
    experiment_routes = {
        'details': {
            'GET': get_experiment
        },
        'results': {
            'GET': get_experiment_results,
            'POST': get_experiment_results
        },
        'analysis': {
            'GET': analyze_experiment
        },
        'export': {
            'GET': export_experiment
        },
        'delete': {
            'DELETE': delete_experiment,
            'POST': delete_experiment  # Allow POST for easier frontend integration
        }
    }
    
    try:
        # Check if this is an experiment-specific route
        if 'experiment_id' in kwargs:
            experiment_id = kwargs['experiment_id']
            
            if path in experiment_routes and method in experiment_routes[path]:
                handler = experiment_routes[path][method]
                return handler(experiment_id)
            else:
                return {
                    'success': False,
                    'error': {'message': f'Method {method} not allowed for {path}'}
                }, 405
        
        # Handle general routes
        elif path in api_routes and method in api_routes[path]:
            handler = api_routes[path][method]
            return handler()
        
        else:
            return {
                'success': False,
                'error': {'message': f'Endpoint not found: {method} {path}'}
            }, 404
    
    except Exception as e:
        logger.error(f"API handler error: {str(e)}")
        return {
            'success': False,
            'error': {
                'message': str(e),
                'code': 'HANDLER_ERROR'
            }
        }, 500

# Example usage for Next.js API integration
if __name__ == "__main__":
    # Test the API handlers
    print("Testing Experiments API Integration...")
    
    # Test health check
    response, status = handle_experiments_api('GET', 'health')
    print(f"Health Check: {status} - {response}")
    
    # Test stats
    response, status = handle_experiments_api('GET', 'stats')
    print(f"Database Stats: {status} - {response['success']}")
    
    print("API integration test completed!") 