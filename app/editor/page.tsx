'use client'

import { useState } from 'react'
import { useQuantum } from '@/contexts/QuantumContext'
import { useError } from '@/contexts/ErrorContext'
import { 
  Play, 
  Square, 
  RotateCcw, 
  Save, 
  Download, 
  Upload,
  Settings,
  Info,
  Zap,
  Atom
} from 'lucide-react'

const quantumGates = [
  { id: 'H', name: 'Hadamard', symbol: 'H', description: 'Creates superposition' },
  { id: 'X', name: 'Pauli-X', symbol: 'X', description: 'Bit flip gate' },
  { id: 'Y', name: 'Pauli-Y', symbol: 'Y', description: 'Bit and phase flip' },
  { id: 'Z', name: 'Pauli-Z', symbol: 'Z', description: 'Phase flip gate' },
  { id: 'CNOT', name: 'CNOT', symbol: '⊕', description: 'Controlled NOT gate' },
  { id: 'RX', name: 'RX', symbol: 'RX', description: 'X-axis rotation' },
  { id: 'RY', name: 'RY', symbol: 'RY', description: 'Y-axis rotation' },
  { id: 'RZ', name: 'RZ', symbol: 'RZ', description: 'Z-axis rotation' }
]

export default function EditorPage() {
  const { state, createCircuit, addGate, simulateCircuit, validateCircuit } = useQuantum()
  const { addAlert } = useError()
  const [circuitName, setCircuitName] = useState('My Circuit')
  const [numQubits, setNumQubits] = useState(2)
  const [selectedGate, setSelectedGate] = useState<string | null>(null)
  const [isCreating, setIsCreating] = useState(!state.currentCircuit)

  const handleCreateCircuit = () => {
    if (numQubits < 1 || numQubits > 30) {
      addAlert({
        type: 'error',
        title: 'Invalid Qubit Count',
        message: 'Number of qubits must be between 1 and 30',
        persistent: false
      })
      return
    }

    createCircuit(circuitName, numQubits)
    setIsCreating(false)
  }

  const handleAddGate = (qubitIndex: number) => {
    if (!selectedGate) {
      addAlert({
        type: 'warning',
        title: 'No Gate Selected',
        message: 'Please select a gate from the toolbar first',
        persistent: false
      })
      return
    }

    if (selectedGate === 'CNOT') {
      // For CNOT, we need two qubits
      if (qubitIndex === state.currentCircuit!.qubits - 1) {
        addAlert({
          type: 'warning',
          title: 'CNOT Target Required',
          message: 'CNOT gate requires a target qubit below the control',
          persistent: false
        })
        return
      }
      addGate(selectedGate, [qubitIndex, qubitIndex + 1])
    } else if (['RX', 'RY', 'RZ'].includes(selectedGate)) {
      // Rotation gates need angle parameter
      const angle = parseFloat(prompt('Enter rotation angle (in radians):') || '0')
      if (isNaN(angle)) {
        addAlert({
          type: 'error',
          title: 'Invalid Angle',
          message: 'Please enter a valid number for the rotation angle',
          persistent: false
        })
        return
      }
      addGate(selectedGate, [qubitIndex], [angle])
    } else {
      // Single qubit gates
      addGate(selectedGate, [qubitIndex])
    }
  }

  const handleSimulate = async () => {
    if (!state.currentCircuit || state.currentCircuit.operations.length === 0) {
      addAlert({
        type: 'warning',
        title: 'Empty Circuit',
        message: 'Add some gates to your circuit before simulating',
        persistent: false
      })
      return
    }

    const shots = parseInt(prompt('Enter number of shots (1-10000):') || '1000')
    if (isNaN(shots) || shots < 1 || shots > 10000) {
      addAlert({
        type: 'error',
        title: 'Invalid Shots',
        message: 'Number of shots must be between 1 and 10,000',
        persistent: false
      })
      return
    }

    await simulateCircuit(shots)
  }

  if (isCreating) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-8">
          <div className="text-center mb-6">
            <Atom className="h-12 w-12 text-quantum-600 mx-auto mb-4" />
            <h1 className="text-2xl font-bold text-gray-900">Create Quantum Circuit</h1>
            <p className="text-gray-600 mt-2">Start building your quantum circuit</p>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Circuit Name
              </label>
              <input
                type="text"
                value={circuitName}
                onChange={(e) => setCircuitName(e.target.value)}
                className="quantum-input"
                placeholder="Enter circuit name"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Number of Qubits
              </label>
              <input
                type="number"
                min="1"
                max="30"
                value={numQubits}
                onChange={(e) => setNumQubits(parseInt(e.target.value) || 1)}
                className="quantum-input"
              />
              <p className="text-sm text-gray-500 mt-1">
                Maximum 30 qubits supported
              </p>
            </div>

            <button
              onClick={handleCreateCircuit}
              className="quantum-button w-full flex items-center justify-center space-x-2"
            >
              <Zap className="h-4 w-4" />
              <span>Create Circuit</span>
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <Atom className="h-8 w-8 text-quantum-600" />
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  {state.currentCircuit?.name || 'Quantum Circuit Editor'}
                </h1>
                <p className="text-sm text-gray-500">
                  {state.currentCircuit?.qubits} qubits • {state.currentCircuit?.operations.length} operations
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={validateCircuit}
                className="quantum-button-secondary flex items-center space-x-2"
              >
                <Info className="h-4 w-4" />
                <span>Validate</span>
              </button>
              <button
                onClick={handleSimulate}
                disabled={state.isSimulating}
                className="quantum-button flex items-center space-x-2"
              >
                {state.isSimulating ? (
                  <div className="quantum-spinner w-4 h-4" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                <span>{state.isSimulating ? 'Simulating...' : 'Simulate'}</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-4 lg:p-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Gate Palette */}
          <div className="lg:col-span-1">
            <div className="quantum-card p-4 sticky top-4">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Quantum Gates</h2>
              <div className="space-y-2">
                {quantumGates.map((gate) => (
                  <button
                    key={gate.id}
                    onClick={() => setSelectedGate(gate.id)}
                    className={`w-full p-3 rounded-lg border text-left transition-colors ${
                      selectedGate === gate.id
                        ? 'border-quantum-500 bg-quantum-50 text-quantum-700'
                        : 'border-gray-200 hover:border-gray-300 bg-white'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium">{gate.name}</span>
                      <span className="text-lg font-mono">{gate.symbol}</span>
                    </div>
                    <p className="text-sm text-gray-600">{gate.description}</p>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Circuit Canvas */}
          <div className="lg:col-span-3">
            <div className="quantum-card p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-gray-900">Circuit Diagram</h2>
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <span>Selected:</span>
                  <span className="font-medium text-quantum-600">
                    {selectedGate || 'None'}
                  </span>
                </div>
              </div>

              {/* Circuit Visualization */}
              <div className="bg-gray-50 rounded-lg p-6 min-h-96">
                {state.currentCircuit && (
                  <div className="space-y-4">
                    {Array.from({ length: state.currentCircuit.qubits }, (_, i) => (
                      <div key={i} className="flex items-center space-x-4">
                        <div className="w-16 text-right">
                          <span className="text-sm font-medium text-gray-600">
                            |q{i}⟩
                          </span>
                        </div>
                        <div className="flex-1 relative">
                          <div className="h-px bg-gray-400 w-full absolute top-1/2 transform -translate-y-1/2" />
                          <div className="flex items-center space-x-2 relative z-10">
                            {/* Render operations for this qubit */}
                            {state.currentCircuit.operations
                              .filter(op => op.targets.includes(i))
                              .map((op, opIndex) => (
                                <div
                                  key={op.id}
                                  className="bg-white border-2 border-quantum-500 rounded px-3 py-1 text-sm font-medium"
                                >
                                  {op.gate}
                                  {op.params.length > 0 && (
                                    <span className="text-xs text-gray-500">
                                      ({op.params[0].toFixed(2)})
                                    </span>
                                  )}
                                </div>
                              ))}
                            {/* Add gate button */}
                            <button
                              onClick={() => handleAddGate(i)}
                              className="w-8 h-8 border-2 border-dashed border-gray-300 rounded hover:border-quantum-500 hover:bg-quantum-50 transition-colors flex items-center justify-center"
                            >
                              <span className="text-gray-400 hover:text-quantum-600">+</span>
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {!state.currentCircuit && (
                  <div className="text-center text-gray-500 py-12">
                    <Square className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No circuit loaded. Create a new circuit to get started.</p>
                  </div>
                )}
              </div>

              {/* Circuit Statistics */}
              {state.currentCircuit && (
                <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="bg-blue-50 rounded-lg p-4">
                    <div className="text-2xl font-bold text-blue-600">
                      {state.currentCircuit.qubits}
                    </div>
                    <div className="text-sm text-blue-600">Qubits</div>
                  </div>
                  <div className="bg-green-50 rounded-lg p-4">
                    <div className="text-2xl font-bold text-green-600">
                      {state.currentCircuit.operations.length}
                    </div>
                    <div className="text-sm text-green-600">Operations</div>
                  </div>
                  <div className="bg-purple-50 rounded-lg p-4">
                    <div className="text-2xl font-bold text-purple-600">
                      {Math.max(...state.currentCircuit.operations.map(op => op.position.x), 0) + 1}
                    </div>
                    <div className="text-sm text-purple-600">Circuit Depth</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Simulation Results */}
        {state.simulationResults.length > 0 && (
          <div className="mt-6">
            <div className="quantum-card p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Latest Simulation Results</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-md font-medium text-gray-700 mb-2">Measurement Counts</h3>
                  <div className="bg-gray-50 rounded-lg p-4 max-h-64 overflow-y-auto">
                    {Object.entries(state.simulationResults[state.simulationResults.length - 1].counts)
                      .sort(([, a], [, b]) => b - a)
                      .map(([state, count]) => (
                        <div key={state} className="flex justify-between items-center py-1">
                          <span className="font-mono text-sm">|{state}⟩</span>
                          <span className="text-sm text-gray-600">
                            {count} ({((count / state.simulationResults[state.simulationResults.length - 1].shots) * 100).toFixed(1)}%)
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
                <div>
                  <h3 className="text-md font-medium text-gray-700 mb-2">Simulation Info</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Shots:</span>
                      <span>{state.simulationResults[state.simulationResults.length - 1].shots}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Execution Time:</span>
                      <span>{state.simulationResults[state.simulationResults.length - 1].executionTime.toFixed(2)}ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Timestamp:</span>
                      <span>{state.simulationResults[state.simulationResults.length - 1].timestamp.toLocaleTimeString()}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
} 