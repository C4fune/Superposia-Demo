"""
Quantum State Visualization Components

This module provides high-level visualization components for quantum states,
including Bloch sphere rendering, probability histograms, and comprehensive
state analysis displays.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
from io import BytesIO

from quantum_platform.visualization.state_utils import (
    compute_bloch_coordinates, analyze_state_structure, get_state_probabilities,
    BlochCoordinates, StateStructure, calculate_entanglement_measures
)


class VisualizationMode(Enum):
    """Visualization display modes."""
    BLOCH_SPHERE = "bloch_sphere"
    PROBABILITY_HISTOGRAM = "probability_histogram"
    STATE_VECTOR_TABLE = "state_vector_table"
    ENTANGLEMENT_ANALYSIS = "entanglement_analysis"
    COHERENCE_ANALYSIS = "coherence_analysis"
    AMPLITUDE_HEATMAP = "amplitude_heatmap"


@dataclass
class VisualizationConfig:
    """Configuration for state visualizations."""
    # General settings
    max_qubits_for_full_display: int = 10
    max_basis_states_displayed: int = 20
    precision_digits: int = 4
    
    # Bloch sphere settings
    show_bloch_axes: bool = True
    show_bloch_labels: bool = True
    bloch_sphere_size: Tuple[int, int] = (400, 400)
    
    # Histogram settings
    histogram_bar_color: str = "#4CAF50"
    histogram_height: int = 300
    show_probability_labels: bool = True
    
    # Heatmap settings
    heatmap_colormap: str = "viridis"
    show_amplitude_phase: bool = True
    
    # Performance settings
    auto_reduce_large_states: bool = True
    sampling_threshold: int = 1000000  # For states larger than this


@dataclass
class BlochSphere:
    """
    Bloch sphere visualization for a single qubit.
    
    Provides 2D and 3D representations of qubit states on the Bloch sphere.
    """
    coordinates: BlochCoordinates
    qubit_index: int
    label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'type': 'bloch_sphere',
            'qubit_index': self.qubit_index,
            'label': self.label,
            'coordinates': {
                'x': self.coordinates.x,
                'y': self.coordinates.y,
                'z': self.coordinates.z,
                'radius': self.coordinates.radius,
                'theta': self.coordinates.theta,
                'phi': self.coordinates.phi
            }
        }
    
    def get_classical_state_description(self) -> str:
        """Get human-readable description of the quantum state."""
        x, y, z = self.coordinates.x, self.coordinates.y, self.coordinates.z
        
        # Check for special states
        if abs(z - 1) < 1e-6:
            return "|0⟩ (Ground state)"
        elif abs(z + 1) < 1e-6:
            return "|1⟩ (Excited state)"
        elif abs(x - 1) < 1e-6:
            return "|+⟩ (Superposition +)"
        elif abs(x + 1) < 1e-6:
            return "|-⟩ (Superposition -)"
        elif abs(y - 1) < 1e-6:
            return "|+i⟩ (Y-basis +)"
        elif abs(y + 1) < 1e-6:
            return "|-i⟩ (Y-basis -)"
        else:
            return f"Mixed state (r={self.coordinates.radius:.3f})"
    
    def get_svg_representation(self, size: Tuple[int, int] = (200, 200)) -> str:
        """Generate SVG representation of the Bloch sphere (2D projection)."""
        width, height = size
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 2 - 20
        
        # Project 3D coordinates to 2D (simple orthographic projection)
        x_2d = center_x + self.coordinates.x * radius
        y_2d = center_y - self.coordinates.z * radius  # Flip Y for SVG coordinates
        
        svg = f'''
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <!-- Background circle -->
            <circle cx="{center_x}" cy="{center_y}" r="{radius}" 
                    fill="none" stroke="#cccccc" stroke-width="2"/>
            
            <!-- Axes -->
            <line x1="{center_x - radius}" y1="{center_y}" 
                  x2="{center_x + radius}" y2="{center_y}" 
                  stroke="#888888" stroke-width="1" opacity="0.5"/>
            <line x1="{center_x}" y1="{center_y - radius}" 
                  x2="{center_x}" y2="{center_y + radius}" 
                  stroke="#888888" stroke-width="1" opacity="0.5"/>
            
            <!-- State point -->
            <circle cx="{x_2d:.1f}" cy="{y_2d:.1f}" r="8" 
                    fill="#ff4444" stroke="#aa0000" stroke-width="2"/>
            
            <!-- State vector -->
            <line x1="{center_x}" y1="{center_y}" x2="{x_2d:.1f}" y2="{y_2d:.1f}" 
                  stroke="#ff4444" stroke-width="3" marker-end="url(#arrowhead)"/>
            
            <!-- Arrow marker -->
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                        refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#ff4444"/>
                </marker>
            </defs>
            
            <!-- Labels -->
            <text x="{center_x + radius + 10}" y="{center_y + 5}" font-size="12" fill="#666">X</text>
            <text x="{center_x - 5}" y="{center_y - radius - 5}" font-size="12" fill="#666">Z</text>
        </svg>
        '''
        
        return svg.strip()


@dataclass
class ProbabilityHistogram:
    """
    Probability histogram visualization for measurement outcomes.
    """
    probabilities: Dict[str, float]
    num_qubits: int
    basis: str = "computational"
    config: Optional[VisualizationConfig] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = VisualizationConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'type': 'probability_histogram',
            'num_qubits': self.num_qubits,
            'basis': self.basis,
            'probabilities': self.probabilities,
            'sorted_probabilities': self.get_sorted_probabilities()
        }
    
    def get_sorted_probabilities(self, max_entries: Optional[int] = None) -> List[Tuple[str, float]]:
        """Get probabilities sorted by magnitude."""
        sorted_probs = sorted(self.probabilities.items(), key=lambda x: x[1], reverse=True)
        
        if max_entries is None:
            max_entries = self.config.max_basis_states_displayed
        
        return sorted_probs[:max_entries]
    
    def get_entropy(self) -> float:
        """Calculate Shannon entropy of the probability distribution."""
        entropy = 0.0
        for prob in self.probabilities.values():
            if prob > 1e-12:
                entropy -= prob * np.log2(prob)
        return entropy
    
    def get_dominant_states(self, threshold: float = 0.01) -> List[Tuple[str, float]]:
        """Get states with probability above threshold."""
        return [(state, prob) for state, prob in self.probabilities.items() 
                if prob >= threshold]
    
    def generate_html_chart(self) -> str:
        """Generate HTML/JavaScript chart for the histogram."""
        sorted_probs = self.get_sorted_probabilities()
        
        states = [item[0] for item in sorted_probs]
        probs = [item[1] for item in sorted_probs]
        
        # Generate Chart.js compatible data
        chart_data = {
            'labels': states,
            'datasets': [{
                'label': 'Probability',
                'data': probs,
                'backgroundColor': self.config.histogram_bar_color,
                'borderColor': '#388E3C',
                'borderWidth': 1
            }]
        }
        
        html = f'''
        <div style="width: 100%; height: {self.config.histogram_height}px;">
            <canvas id="probabilityChart"></canvas>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            const ctx = document.getElementById('probabilityChart').getContext('2d');
            const chart = new Chart(ctx, {{
                type: 'bar',
                data: {json.dumps(chart_data)},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 1.0,
                            title: {{
                                display: true,
                                text: 'Probability'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Basis State'
                            }}
                        }}
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Quantum State Probabilities ({self.basis} basis)'
                        }}
                    }}
                }}
            }});
        </script>
        '''
        
        return html


@dataclass
class StateVectorAnalysis:
    """
    Comprehensive analysis and visualization of quantum state vectors.
    """
    state_vector: np.ndarray
    structure: StateStructure
    config: Optional[VisualizationConfig] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = VisualizationConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'type': 'state_vector_analysis',
            'num_qubits': self.structure.num_qubits,
            'state_dimension': self.structure.state_dimension,
            'is_pure': self.structure.is_pure,
            'is_separable': self.structure.is_separable,
            'max_amplitude': self.structure.max_amplitude,
            'dominant_basis_states': self.structure.dominant_basis_states,
            'coherence_measures': self.structure.coherence_measures,
            'entanglement_measures': (
                {
                    'concurrence': self.structure.entanglement_structure.concurrence,
                    'negativity': self.structure.entanglement_structure.negativity,
                    'von_neumann_entropy': self.structure.entanglement_structure.von_neumann_entropy,
                    'linear_entropy': self.structure.entanglement_structure.linear_entropy,
                    'schmidt_rank': self.structure.entanglement_structure.schmidt_rank,
                    'schmidt_coefficients': self.structure.entanglement_structure.schmidt_coefficients
                } if self.structure.entanglement_structure else None
            ),
            'amplitude_table': self.get_amplitude_table()
        }
    
    def get_amplitude_table(self, max_entries: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get formatted table of state amplitudes."""
        if max_entries is None:
            max_entries = self.config.max_basis_states_displayed
        
        num_qubits = self.structure.num_qubits
        amplitudes = []
        
        # Get significant amplitudes
        for i, amplitude in enumerate(self.state_vector):
            if abs(amplitude) > 1e-10:  # Only show non-negligible amplitudes
                basis_state = format(i, f'0{num_qubits}b')
                probability = abs(amplitude) ** 2
                
                amplitudes.append({
                    'basis_state': basis_state,
                    'amplitude_real': float(amplitude.real),
                    'amplitude_imag': float(amplitude.imag),
                    'amplitude_magnitude': float(abs(amplitude)),
                    'amplitude_phase': float(np.angle(amplitude)),
                    'probability': float(probability)
                })
        
        # Sort by probability
        amplitudes.sort(key=lambda x: x['probability'], reverse=True)
        
        return amplitudes[:max_entries]
    
    def generate_amplitude_heatmap_data(self) -> Dict[str, Any]:
        """Generate data for amplitude heatmap visualization."""
        num_qubits = self.structure.num_qubits
        
        if num_qubits > 8:  # Too large for meaningful heatmap
            return {'error': 'State too large for heatmap visualization'}
        
        state_dim = 2 ** num_qubits
        
        # Create 2D grid for visualization
        if num_qubits <= 4:
            # Show full matrix
            grid_size = int(np.sqrt(state_dim))
            magnitude_grid = np.abs(self.state_vector).reshape(grid_size, -1)
            phase_grid = np.angle(self.state_vector).reshape(grid_size, -1)
        else:
            # Use sampling for larger states
            grid_size = 16  # Fixed grid size
            indices = np.linspace(0, state_dim - 1, grid_size * grid_size, dtype=int)
            magnitudes = np.abs(self.state_vector[indices])
            phases = np.angle(self.state_vector[indices])
            
            magnitude_grid = magnitudes.reshape(grid_size, grid_size)
            phase_grid = phases.reshape(grid_size, grid_size)
        
        return {
            'magnitude_grid': magnitude_grid.tolist(),
            'phase_grid': phase_grid.tolist(),
            'grid_size': magnitude_grid.shape,
            'max_magnitude': float(np.max(magnitude_grid)),
            'colormap': self.config.heatmap_colormap
        }


class StateVisualizer:
    """
    Main coordinator for quantum state visualizations.
    
    Provides a unified interface for creating and managing various types
    of quantum state visualizations.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self._visualization_cache: Dict[str, Any] = {}
    
    def visualize_state(self, state_vector: np.ndarray, 
                       modes: Optional[List[VisualizationMode]] = None,
                       qubit_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create comprehensive visualization of a quantum state.
        
        Args:
            state_vector: Quantum state vector to visualize
            modes: List of visualization modes to generate
            qubit_labels: Optional labels for qubits
            
        Returns:
            Dictionary containing all requested visualizations
        """
        if modes is None:
            modes = [VisualizationMode.BLOCH_SPHERE, VisualizationMode.PROBABILITY_HISTOGRAM]
        
        num_qubits = int(np.log2(len(state_vector)))
        
        # Check size limits
        if (num_qubits > self.config.max_qubits_for_full_display and 
            self.config.auto_reduce_large_states):
            return self._create_reduced_visualization(state_vector, modes)
        
        visualizations = {}
        
        # Generate state structure analysis
        structure = analyze_state_structure(state_vector, self.config.max_basis_states_displayed)
        
        for mode in modes:
            if mode == VisualizationMode.BLOCH_SPHERE:
                visualizations['bloch_spheres'] = self._create_bloch_spheres(
                    state_vector, num_qubits, qubit_labels
                )
            
            elif mode == VisualizationMode.PROBABILITY_HISTOGRAM:
                probabilities = get_state_probabilities(state_vector)
                visualizations['probability_histogram'] = ProbabilityHistogram(
                    probabilities=probabilities,
                    num_qubits=num_qubits,
                    config=self.config
                )
            
            elif mode == VisualizationMode.STATE_VECTOR_TABLE:
                visualizations['state_analysis'] = StateVectorAnalysis(
                    state_vector=state_vector,
                    structure=structure,
                    config=self.config
                )
            
            elif mode == VisualizationMode.ENTANGLEMENT_ANALYSIS:
                if num_qubits > 1:
                    entanglement = calculate_entanglement_measures(state_vector)
                    visualizations['entanglement_analysis'] = entanglement
            
            elif mode == VisualizationMode.AMPLITUDE_HEATMAP:
                analysis = StateVectorAnalysis(
                    state_vector=state_vector,
                    structure=structure,
                    config=self.config
                )
                visualizations['amplitude_heatmap'] = analysis.generate_amplitude_heatmap_data()
        
        # Add general state information
        visualizations['state_info'] = {
            'num_qubits': num_qubits,
            'state_dimension': len(state_vector),
            'structure': structure,
            'timestamp': np.datetime64('now').astype(str)
        }
        
        return visualizations
    
    def _create_bloch_spheres(self, state_vector: np.ndarray, 
                            num_qubits: int, 
                            qubit_labels: Optional[List[str]] = None) -> List[BlochSphere]:
        """Create Bloch sphere representations for all qubits."""
        bloch_spheres = []
        
        for i in range(num_qubits):
            try:
                coords = compute_bloch_coordinates(state_vector, i)
                label = qubit_labels[i] if qubit_labels and i < len(qubit_labels) else f"q{i}"
                
                bloch_sphere = BlochSphere(
                    coordinates=coords,
                    qubit_index=i,
                    label=label
                )
                bloch_spheres.append(bloch_sphere)
                
            except Exception as e:
                # Create placeholder for problematic qubits
                bloch_spheres.append(BlochSphere(
                    coordinates=BlochCoordinates(0, 0, 0, 0),
                    qubit_index=i,
                    label=f"q{i} (error: {str(e)[:50]})"
                ))
        
        return bloch_spheres
    
    def _create_reduced_visualization(self, state_vector: np.ndarray, 
                                    modes: List[VisualizationMode]) -> Dict[str, Any]:
        """Create reduced visualizations for large quantum states."""
        num_qubits = int(np.log2(len(state_vector)))
        
        # Sample dominant basis states
        probabilities = np.abs(state_vector) ** 2
        dominant_indices = np.argsort(probabilities)[-20:]  # Top 20 states
        
        sampled_probs = {}
        for idx in dominant_indices:
            if probabilities[idx] > 1e-10:
                basis_state = format(idx, f'0{num_qubits}b')
                sampled_probs[basis_state] = float(probabilities[idx])
        
        return {
            'reduced_visualization': True,
            'num_qubits': num_qubits,
            'state_dimension': len(state_vector),
            'sampled_probabilities': sampled_probs,
            'total_probability_shown': sum(sampled_probs.values()),
            'note': f'Showing dominant states only (system too large for full visualization)',
            'probability_histogram': ProbabilityHistogram(
                probabilities=sampled_probs,
                num_qubits=num_qubits,
                basis="computational (sampled)",
                config=self.config
            )
        }
    
    def create_comparison_visualization(self, states: List[Tuple[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Create side-by-side comparison of multiple quantum states.
        
        Args:
            states: List of (label, state_vector) tuples
            
        Returns:
            Comparison visualization data
        """
        comparisons = {}
        
        for label, state_vector in states:
            comparisons[label] = self.visualize_state(
                state_vector,
                modes=[VisualizationMode.BLOCH_SPHERE, VisualizationMode.PROBABILITY_HISTOGRAM]
            )
        
        # Add fidelity comparisons if multiple states
        if len(states) > 1:
            from quantum_platform.visualization.state_utils import compute_fidelity
            
            fidelities = {}
            for i, (label1, state1) in enumerate(states):
                for j, (label2, state2) in enumerate(states[i+1:], i+1):
                    fidelity = compute_fidelity(state1, state2)
                    fidelities[f"{label1}_vs_{label2}"] = fidelity
            
            comparisons['fidelity_matrix'] = fidelities
        
        return {
            'type': 'state_comparison',
            'states': comparisons,
            'num_states': len(states)
        }
    
    def export_visualization(self, visualization_data: Dict[str, Any], 
                           format: str = 'json') -> Union[str, bytes]:
        """
        Export visualization data in specified format.
        
        Args:
            visualization_data: Visualization data to export
            format: Export format ('json', 'html', 'png')
            
        Returns:
            Exported data
        """
        if format == 'json':
            return json.dumps(visualization_data, indent=2, default=str)
        
        elif format == 'html':
            return self._generate_html_report(visualization_data)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_html_report(self, visualization_data: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report of visualizations."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>Quantum State Visualization Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".visualization { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }",
            ".bloch-sphere { display: inline-block; margin: 10px; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "</style>",
            "</head><body>",
            f"<h1>Quantum State Visualization Report</h1>",
            f"<p>Generated: {visualization_data.get('state_info', {}).get('timestamp', 'Unknown')}</p>"
        ]
        
        # Add each visualization section
        if 'bloch_spheres' in visualization_data:
            html_parts.append("<div class='visualization'>")
            html_parts.append("<h2>Bloch Sphere Representations</h2>")
            
            for sphere in visualization_data['bloch_spheres']:
                html_parts.append(f"<div class='bloch-sphere'>")
                html_parts.append(f"<h3>{sphere.label}</h3>")
                html_parts.append(sphere.get_svg_representation())
                html_parts.append(f"<p>{sphere.get_classical_state_description()}</p>")
                html_parts.append("</div>")
            
            html_parts.append("</div>")
        
        if 'probability_histogram' in visualization_data:
            histogram = visualization_data['probability_histogram']
            html_parts.append("<div class='visualization'>")
            html_parts.append("<h2>Probability Distribution</h2>")
            html_parts.append(histogram.generate_html_chart())
            html_parts.append("</div>")
        
        html_parts.extend(["</body></html>"])
        
        return "\n".join(html_parts) 