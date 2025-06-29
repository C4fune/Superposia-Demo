"""
Quantum Circuit API Endpoint

This module provides REST API endpoints for quantum circuit operations,
integrating with the comprehensive error handling system.
"""

import json
import traceback
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import sys
import os

# Add the quantum platform to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from quantum_platform import QuantumProgram
from quantum_platform.compiler.language.operations import H, X, Y, Z, CNOT, RX, RY, RZ, measure
from quantum_platform.simulation import StateVectorSimulator
from quantum_platform.errors import (
    get_error_reporter, get_alert_manager,
    QubitError, CompilationError, SimulationError,
    format_error_message
)


class QuantumCircuitAPI(BaseHTTPRequestHandler):
    """API handler for quantum circuit operations."""
    
    def __init__(self, request, client_address, server):
        self.error_reporter = get_error_reporter()
        self.alert_manager = get_alert_manager()
        super().__init__(request, client_address, server)
    
    def do_GET(self):
        """Handle GET requests."""
        try:
            parsed_url = urlparse(self.path)
            
            if parsed_url.path == '/api/quantum/circuit/info':
                self._handle_circuit_info()
            elif parsed_url.path == '/api/quantum/circuit/gates':
                self._handle_gates_list()
            else:
                self._send_error(404, "Endpoint not found")
                
        except Exception as e:
            self._handle_api_error(e)
    
    def do_POST(self):
        """Handle POST requests."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError as e:
                self._send_error(400, f"Invalid JSON: {e}")
                return
            
            parsed_url = urlparse(self.path)
            
            if parsed_url.path == '/api/quantum/circuit/create':
                self._handle_circuit_creation(data)
            elif parsed_url.path == '/api/quantum/circuit/simulate':
                self._handle_circuit_simulation(data)
            elif parsed_url.path == '/api/quantum/circuit/validate':
                self._handle_circuit_validation(data)
            else:
                self._send_error(404, "Endpoint not found")
                
        except Exception as e:
            self._handle_api_error(e)
    
    def _handle_circuit_info(self):
        """Get general circuit information."""
        info = {
            "platform_version": "1.0.0",
            "max_qubits": 30,
            "supported_gates": [
                "H", "X", "Y", "Z", "CNOT", "RX", "RY", "RZ",
                "S", "T", "SWAP", "TOFFOLI"
            ],
            "error_handling": True,
            "simulation_backends": ["statevector"],
            "max_circuit_depth": 1000
        }
        self._send_json_response(info)
    
    def _handle_gates_list(self):
        """Get list of available quantum gates."""
        gates = {
            "single_qubit": {
                "H": {"name": "Hadamard", "params": []},
                "X": {"name": "Pauli-X", "params": []},
                "Y": {"name": "Pauli-Y", "params": []},
                "Z": {"name": "Pauli-Z", "params": []},
                "RX": {"name": "X-Rotation", "params": ["theta"]},
                "RY": {"name": "Y-Rotation", "params": ["theta"]},
                "RZ": {"name": "Z-Rotation", "params": ["theta"]},
                "S": {"name": "S Gate", "params": []},
                "T": {"name": "T Gate", "params": []}
            },
            "two_qubit": {
                "CNOT": {"name": "Controlled-NOT", "params": []},
                "CZ": {"name": "Controlled-Z", "params": []},
                "SWAP": {"name": "SWAP", "params": []}
            },
            "three_qubit": {
                "TOFFOLI": {"name": "Toffoli", "params": []},
                "FREDKIN": {"name": "Fredkin", "params": []}
            }
        }
        self._send_json_response(gates)
    
    def _handle_circuit_creation(self, data):
        """Create a quantum circuit from specification."""
        try:
            circuit_spec = data.get('circuit', {})
            num_qubits = circuit_spec.get('num_qubits', 2)
            operations = circuit_spec.get('operations', [])
            
            # Validate input
            if num_qubits <= 0 or num_qubits > 30:
                raise QubitError(
                    f"Invalid number of qubits: {num_qubits}",
                    user_message="Number of qubits must be between 1 and 30"
                )
            
            # Create quantum program
            with QuantumProgram(name=circuit_spec.get('name', 'api_circuit')) as qp:
                qubits = qp.allocate(num_qubits)
                
                # Apply operations
                for op in operations:
                    self._apply_operation(qubits, op)
                
                # Add measurements if requested
                if circuit_spec.get('measure_all', True):
                    measure(qubits)
            
            # Convert circuit to serializable format
            circuit_data = {
                "circuit_id": qp.circuit.name,
                "num_qubits": qp.circuit.num_qubits,
                "num_operations": qp.circuit.num_operations,
                "depth": qp.circuit.depth,
                "operations": self._serialize_operations(qp.circuit.operations)
            }
            
            self._send_json_response({
                "success": True,
                "circuit": circuit_data,
                "message": "Circuit created successfully"
            })
            
        except Exception as e:
            self._handle_circuit_error(e, "circuit_creation")
    
    def _handle_circuit_simulation(self, data):
        """Simulate a quantum circuit."""
        try:
            circuit_spec = data.get('circuit', {})
            sim_config = data.get('simulation', {})
            
            shots = sim_config.get('shots', 1000)
            backend = sim_config.get('backend', 'statevector')
            
            # Validate simulation parameters
            if shots <= 0 or shots > 1000000:
                raise SimulationError(
                    f"Invalid shots count: {shots}",
                    user_message="Shots must be between 1 and 1,000,000"
                )
            
            # Create circuit
            num_qubits = circuit_spec.get('num_qubits', 2)
            operations = circuit_spec.get('operations', [])
            
            with QuantumProgram(name='simulation_circuit') as qp:
                qubits = qp.allocate(num_qubits)
                
                for op in operations:
                    self._apply_operation(qubits, op)
                
                measure(qubits)
            
            # Run simulation
            simulator = StateVectorSimulator()
            result = simulator.run(qp.circuit, shots=shots)
            
            # Format results
            simulation_results = {
                "success": True,
                "results": {
                    "counts": result.counts if hasattr(result, 'counts') else {},
                    "shots": shots,
                    "execution_time": getattr(result, 'execution_time', 0),
                    "backend": backend
                },
                "circuit_info": {
                    "num_qubits": qp.circuit.num_qubits,
                    "depth": qp.circuit.depth,
                    "num_operations": qp.circuit.num_operations
                }
            }
            
            self._send_json_response(simulation_results)
            
        except Exception as e:
            self._handle_circuit_error(e, "simulation")
    
    def _handle_circuit_validation(self, data):
        """Validate a quantum circuit specification."""
        try:
            circuit_spec = data.get('circuit', {})
            
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "suggestions": []
            }
            
            # Basic validation
            num_qubits = circuit_spec.get('num_qubits', 0)
            operations = circuit_spec.get('operations', [])
            
            if num_qubits <= 0:
                validation_results["valid"] = False
                validation_results["errors"].append("Number of qubits must be positive")
            
            if num_qubits > 30:
                validation_results["valid"] = False
                validation_results["errors"].append("Maximum 30 qubits supported")
            
            # Validate operations
            for i, op in enumerate(operations):
                try:
                    self._validate_operation(op, num_qubits)
                except Exception as e:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Operation {i}: {str(e)}")
            
            # Add suggestions
            if len(operations) > 100:
                validation_results["warnings"].append("Large circuit may be slow to simulate")
                validation_results["suggestions"].append("Consider circuit optimization")
            
            if num_qubits > 20:
                validation_results["warnings"].append("High qubit count requires significant memory")
                validation_results["suggestions"].append("Use fewer qubits for faster simulation")
            
            self._send_json_response(validation_results)
            
        except Exception as e:
            self._handle_circuit_error(e, "validation")
    
    def _apply_operation(self, qubits, operation):
        """Apply a quantum operation to qubits."""
        gate_type = operation.get('gate')
        targets = operation.get('targets', [])
        params = operation.get('params', [])
        
        if not targets:
            raise CompilationError(f"No target qubits specified for {gate_type}")
        
        # Validate qubit indices
        for target in targets:
            if target < 0 or target >= len(qubits):
                raise QubitError(
                    f"Qubit index {target} out of range",
                    user_message=f"Qubit {target} is not allocated"
                )
        
        # Apply gate based on type
        if gate_type == 'H':
            H(qubits[targets[0]])
        elif gate_type == 'X':
            X(qubits[targets[0]])
        elif gate_type == 'Y':
            Y(qubits[targets[0]])
        elif gate_type == 'Z':
            Z(qubits[targets[0]])
        elif gate_type == 'CNOT':
            if len(targets) != 2:
                raise CompilationError("CNOT requires exactly 2 target qubits")
            CNOT(qubits[targets[0]], qubits[targets[1]])
        elif gate_type == 'RX':
            if len(params) != 1:
                raise CompilationError("RX requires exactly 1 parameter (theta)")
            RX(qubits[targets[0]], params[0])
        elif gate_type == 'RY':
            if len(params) != 1:
                raise CompilationError("RY requires exactly 1 parameter (theta)")
            RY(qubits[targets[0]], params[0])
        elif gate_type == 'RZ':
            if len(params) != 1:
                raise CompilationError("RZ requires exactly 1 parameter (theta)")
            RZ(qubits[targets[0]], params[0])
        else:
            raise CompilationError(f"Unsupported gate: {gate_type}")
    
    def _validate_operation(self, operation, num_qubits):
        """Validate a single operation."""
        gate_type = operation.get('gate')
        targets = operation.get('targets', [])
        params = operation.get('params', [])
        
        if not gate_type:
            raise CompilationError("Gate type not specified")
        
        if not targets:
            raise CompilationError("No target qubits specified")
        
        # Check qubit indices
        for target in targets:
            if target < 0 or target >= num_qubits:
                raise QubitError(f"Qubit index {target} out of range")
        
        # Gate-specific validation
        single_qubit_gates = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'S', 'T']
        two_qubit_gates = ['CNOT', 'CZ', 'SWAP']
        
        if gate_type in single_qubit_gates and len(targets) != 1:
            raise CompilationError(f"{gate_type} requires exactly 1 target qubit")
        
        if gate_type in two_qubit_gates and len(targets) != 2:
            raise CompilationError(f"{gate_type} requires exactly 2 target qubits")
        
        # Parameter validation
        if gate_type in ['RX', 'RY', 'RZ'] and len(params) != 1:
            raise CompilationError(f"{gate_type} requires exactly 1 parameter")
    
    def _serialize_operations(self, operations):
        """Convert circuit operations to serializable format."""
        serialized = []
        for op in operations:
            serialized.append({
                "type": op.__class__.__name__,
                "targets": [q.id for q in op.targets] if hasattr(op, 'targets') else [],
                "params": list(op.parameters.values()) if hasattr(op, 'parameters') else []
            })
        return serialized
    
    def _handle_circuit_error(self, exception, operation):
        """Handle circuit-specific errors with our error system."""
        # Report error
        report = self.error_reporter.collect_error(
            exception,
            user_action=f"API {operation}"
        )
        
        # Format error for API response
        formatted = format_error_message(exception)
        
        error_response = {
            "success": False,
            "error": {
                "type": exception.__class__.__name__,
                "message": formatted.message,
                "code": formatted.error_code,
                "category": formatted.category.value,
                "suggestions": formatted.suggestions[:3],  # Limit suggestions
                "report_id": report.report_id
            }
        }
        
        # Determine HTTP status code
        if isinstance(exception, (QubitError, CompilationError)):
            status_code = 400  # Bad Request
        elif isinstance(exception, SimulationError):
            status_code = 422  # Unprocessable Entity
        else:
            status_code = 500  # Internal Server Error
        
        self._send_json_response(error_response, status_code)
    
    def _handle_api_error(self, exception):
        """Handle general API errors."""
        error_response = {
            "success": False,
            "error": {
                "type": "APIError",
                "message": "An unexpected error occurred",
                "code": "API500",
                "details": str(exception) if hasattr(exception, '__str__') else "Unknown error"
            }
        }
        self._send_json_response(error_response, 500)
    
    def _send_json_response(self, data, status_code=200):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response_json = json.dumps(data, indent=2)
        self.wfile.write(response_json.encode('utf-8'))
    
    def _send_error(self, status_code, message):
        """Send error response."""
        error_data = {
            "success": False,
            "error": {
                "message": message,
                "status_code": status_code
            }
        }
        self._send_json_response(error_data, status_code)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


# Vercel handler function
def handler(request, context):
    """Main handler function for Vercel."""
    return QuantumCircuitAPI(request, None, None) 