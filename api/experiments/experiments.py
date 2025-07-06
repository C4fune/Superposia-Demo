"""
Experiments API Endpoints for Next.js Application

This module provides the API endpoints used by the Next.js frontend
to interact with the quantum experiment tracking system.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify

# Import the experiment system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from quantum_platform.experiments.manager import ExperimentManager
from quantum_platform.experiments.models import ExperimentType, ExperimentStatus

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize experiment manager
experiment_manager = ExperimentManager()

# Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

def create_error_response(message: str, status_code: int = 500) -> tuple:
    """Create a standardized error response."""
    return jsonify({
        'success': False,
        'error': {'message': message}
    }), status_code

def create_success_response(data: Any, status_code: int = 200) -> tuple:
    """Create a standardized success response."""
    return jsonify({
        'success': True,
        'data': data
    }), status_code

@app.route('/api/experiments/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return create_success_response({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/experiments/stats', methods=['GET'])
def get_database_stats():
    """Get database statistics."""
    try:
        stats = experiment_manager.get_database_stats()
        return create_success_response(stats)
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return create_error_response(str(e))

@app.route('/api/experiments/list', methods=['GET'])
def list_experiments():
    """List experiments with optional filtering."""
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Optional filters
        filters = {}
        if request.args.get('status'):
            filters['status_filter'] = request.args.get('status')
        if request.args.get('backend'):
            filters['backend_filter'] = request.args.get('backend')
        if request.args.get('created_by'):
            filters['created_by_filter'] = request.args.get('created_by')
        if request.args.get('experiment_type'):
            filters['experiment_type_filter'] = request.args.get('experiment_type')
        if request.args.get('tags'):
            filters['tags_filter'] = request.args.get('tags').split(',')
        
        experiments = experiment_manager.list_experiments(
            limit=limit, 
            offset=offset, 
            **filters
        )
        
        return create_success_response({
            'experiments': [exp.to_dict() for exp in experiments],
            'pagination': {
                'limit': limit,
                'offset': offset,
                'total': len(experiments)
            }
        })
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        return create_error_response(str(e))

@app.route('/api/experiments/create', methods=['POST'])
def create_experiment():
    """Create a new experiment."""
    try:
        data = request.get_json()
        if not data:
            return create_error_response('No data provided', 400)
        
        required_fields = ['name', 'circuit_id', 'backend']
        for field in required_fields:
            if field not in data:
                return create_error_response(f'Missing required field: {field}', 400)
        
        # Get backend instance (simplified for now)
        from quantum_platform.hardware.backends.local_simulator import LocalSimulatorBackend
        backend = LocalSimulatorBackend()
        
        experiment = experiment_manager.create_experiment(
            name=data['name'],
            circuit_id=data['circuit_id'],
            backend=backend,
            experiment_type=data.get('experiment_type', ExperimentType.SINGLE_SHOT.value),
            description=data.get('description'),
            shots=data.get('shots', 1000),
            parameter_sweep=data.get('parameter_sweep'),
            tags=data.get('tags'),
            metadata=data.get('metadata')
        )
        
        return create_success_response(experiment.to_dict(), 201)
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        return create_error_response(str(e))

@app.route('/api/experiments/<experiment_id>', methods=['GET'])
def get_experiment(experiment_id):
    """Get experiment by ID."""
    try:
        experiment = experiment_manager.database.get_experiment(experiment_id)
        if not experiment:
            return create_error_response('Experiment not found', 404)
        
        return create_success_response(experiment.to_dict())
    except Exception as e:
        logger.error(f"Failed to get experiment {experiment_id}: {e}")
        return create_error_response(str(e))

@app.route('/api/experiments/<experiment_id>/run', methods=['POST'])
def run_experiment(experiment_id):
    """Run an experiment."""
    try:
        data = request.get_json() or {}
        async_execution = data.get('async', False)
        
        result = experiment_manager.run_experiment(
            experiment_id=experiment_id,
            async_execution=async_execution
        )
        
        if async_execution:
            return create_success_response({
                'experiment_id': experiment_id,
                'status': 'running',
                'async': True
            })
        else:
            return create_success_response(result.to_dict())
    except Exception as e:
        logger.error(f"Failed to run experiment {experiment_id}: {e}")
        return create_error_response(str(e))

@app.route('/api/experiments/<experiment_id>/results', methods=['GET'])
def get_experiment_results(experiment_id):
    """Get results for an experiment."""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        results = experiment_manager.database.get_results(
            experiment_id, limit=limit, offset=offset
        )
        
        return create_success_response({
            'results': [result.to_dict() for result in results],
            'pagination': {
                'limit': limit,
                'offset': offset,
                'total': len(results)
            }
        })
    except Exception as e:
        logger.error(f"Failed to get results for experiment {experiment_id}: {e}")
        return create_error_response(str(e))

@app.route('/api/experiments/<experiment_id>/summary', methods=['GET'])
def get_experiment_summary(experiment_id):
    """Get experiment summary."""
    try:
        summary = experiment_manager.get_experiment_summary(experiment_id)
        if not summary:
            return create_error_response('Experiment not found', 404)
        
        return create_success_response(summary.to_dict())
    except Exception as e:
        logger.error(f"Failed to get summary for experiment {experiment_id}: {e}")
        return create_error_response(str(e))

@app.route('/api/experiments/<experiment_id>/analysis', methods=['GET'])
def get_experiment_analysis(experiment_id):
    """Get experiment analysis."""
    try:
        analysis = experiment_manager.analyzer.analyze_experiment(experiment_id)
        return create_success_response(analysis)
    except Exception as e:
        logger.error(f"Failed to analyze experiment {experiment_id}: {e}")
        return create_error_response(str(e))

@app.route('/api/experiments/<experiment_id>/delete', methods=['POST'])
def delete_experiment(experiment_id):
    """Delete an experiment."""
    try:
        success = experiment_manager.delete_experiment(experiment_id)
        if success:
            return create_success_response({
                'experiment_id': experiment_id, 
                'status': 'deleted'
            })
        else:
            return create_error_response('Experiment not found', 404)
    except Exception as e:
        logger.error(f"Failed to delete experiment {experiment_id}: {e}")
        return create_error_response(str(e))

@app.route('/api/experiments/compare', methods=['POST'])
def compare_experiments():
    """Compare two experiments."""
    try:
        data = request.get_json()
        if not data or 'experiment_ids' not in data:
            return create_error_response('Missing experiment_ids', 400)
        
        experiment_ids = data['experiment_ids']
        if len(experiment_ids) != 2:
            return create_error_response('Exactly 2 experiment IDs required', 400)
        
        comparison = experiment_manager.compare_experiments(
            experiment_ids[0], experiment_ids[1]
        )
        
        return create_success_response(comparison.to_dict())
    except Exception as e:
        logger.error(f"Failed to compare experiments: {e}")
        return create_error_response(str(e))

@app.route('/api/experiments/active', methods=['GET'])
def get_active_experiments():
    """Get currently active experiments."""
    try:
        active = experiment_manager.get_active_experiments()
        return create_success_response(active)
    except Exception as e:
        logger.error(f"Failed to get active experiments: {e}")
        return create_error_response(str(e))

# Circuit endpoints
@app.route('/api/experiments/circuits/list', methods=['GET'])
def list_circuits():
    """List circuits with optional filtering."""
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        name_filter = request.args.get('name')
        
        circuits = experiment_manager.database.get_circuits(
            limit=limit, 
            offset=offset, 
            name_filter=name_filter
        )
        
        return create_success_response({
            'circuits': [circuit.to_dict() for circuit in circuits],
            'pagination': {
                'limit': limit,
                'offset': offset,
                'total': len(circuits)
            }
        })
    except Exception as e:
        logger.error(f"Failed to list circuits: {e}")
        return create_error_response(str(e))

@app.route('/api/experiments/circuits/create', methods=['POST'])
def create_circuit():
    """Create a new circuit."""
    try:
        data = request.get_json()
        if not data:
            return create_error_response('No data provided', 400)
        
        required_fields = ['name', 'qasm_code', 'num_qubits']
        for field in required_fields:
            if field not in data:
                return create_error_response(f'Missing required field: {field}', 400)
        
        circuit = experiment_manager.database.create_circuit(
            name=data['name'],
            qasm_code=data['qasm_code'],
            num_qubits=data['num_qubits'],
            description=data.get('description'),
            circuit_json=data.get('circuit_json'),
            parameters=data.get('parameters'),
            version=data.get('version', '1.0')
        )
        
        return create_success_response(circuit.to_dict(), 201)
    except Exception as e:
        logger.error(f"Failed to create circuit: {e}")
        return create_error_response(str(e))

@app.route('/api/experiments/circuits/<circuit_id>', methods=['GET'])
def get_circuit(circuit_id):
    """Get circuit by ID."""
    try:
        circuit = experiment_manager.database.get_circuit(circuit_id)
        if not circuit:
            return create_error_response('Circuit not found', 404)
        
        return create_success_response(circuit.to_dict())
    except Exception as e:
        logger.error(f"Failed to get circuit {circuit_id}: {e}")
        return create_error_response(str(e))

# Export endpoint
@app.route('/api/experiments/<experiment_id>/export', methods=['GET'])
def export_experiment(experiment_id):
    """Export experiment data."""
    try:
        format_type = request.args.get('format', 'json')
        
        if format_type not in ['json', 'csv']:
            return create_error_response('Unsupported format. Use json or csv.', 400)
        
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
            return create_success_response(json.loads(export_data))
    except Exception as e:
        logger.error(f"Failed to export experiment {experiment_id}: {e}")
        return create_error_response(str(e))

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return create_error_response('Endpoint not found', 404)

@app.errorhandler(405)
def method_not_allowed(error):
    return create_error_response('Method not allowed', 405)

@app.errorhandler(500)
def internal_error(error):
    return create_error_response('Internal server error', 500)

if __name__ == '__main__':
    app.run(debug=True) 