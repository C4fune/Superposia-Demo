'use client'

import React, { createContext, useContext, useReducer, ReactNode } from 'react'
import { useError } from './ErrorContext'

interface QuantumCircuit {
  id: string
  name: string
  qubits: number
  operations: QuantumOperation[]
  measurements: boolean[]
  created: Date
  modified: Date
}

interface QuantumOperation {
  id: string
  gate: string
  targets: number[]
  params: number[]
  position: { x: number; y: number }
}

interface SimulationResult {
  id: string
  circuitId: string
  counts: Record<string, number>
  shots: number
  executionTime: number
  timestamp: Date
}

interface QuantumState {
  currentCircuit: QuantumCircuit | null
  circuits: QuantumCircuit[]
  simulationResults: SimulationResult[]
  isSimulating: boolean
  selectedGate: string | null
  maxQubits: number
}

type QuantumAction =
  | { type: 'CREATE_CIRCUIT'; circuit: Omit<QuantumCircuit, 'id' | 'created' | 'modified'> }
  | { type: 'LOAD_CIRCUIT'; circuit: QuantumCircuit }
  | { type: 'UPDATE_CIRCUIT'; updates: Partial<QuantumCircuit> }
  | { type: 'ADD_OPERATION'; operation: Omit<QuantumOperation, 'id'> }
  | { type: 'REMOVE_OPERATION'; operationId: string }
  | { type: 'UPDATE_OPERATION'; operationId: string; updates: Partial<QuantumOperation> }
  | { type: 'SET_MEASUREMENTS'; measurements: boolean[] }
  | { type: 'START_SIMULATION' }
  | { type: 'FINISH_SIMULATION'; result: SimulationResult }
  | { type: 'SELECT_GATE'; gate: string | null }
  | { type: 'SET_MAX_QUBITS'; maxQubits: number }

const initialState: QuantumState = {
  currentCircuit: null,
  circuits: [],
  simulationResults: [],
  isSimulating: false,
  selectedGate: null,
  maxQubits: 30
}

function quantumReducer(state: QuantumState, action: QuantumAction): QuantumState {
  switch (action.type) {
    case 'CREATE_CIRCUIT': {
      const newCircuit: QuantumCircuit = {
        ...action.circuit,
        id: `circuit_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        created: new Date(),
        modified: new Date()
      }
      
      return {
        ...state,
        currentCircuit: newCircuit,
        circuits: [...state.circuits, newCircuit]
      }
    }

    case 'LOAD_CIRCUIT':
      return {
        ...state,
        currentCircuit: action.circuit
      }

    case 'UPDATE_CIRCUIT': {
      if (!state.currentCircuit) return state

      const updatedCircuit = {
        ...state.currentCircuit,
        ...action.updates,
        modified: new Date()
      }

      return {
        ...state,
        currentCircuit: updatedCircuit,
        circuits: state.circuits.map(circuit =>
          circuit.id === updatedCircuit.id ? updatedCircuit : circuit
        )
      }
    }

    case 'ADD_OPERATION': {
      if (!state.currentCircuit) return state

      const newOperation: QuantumOperation = {
        ...action.operation,
        id: `op_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      }

      const updatedCircuit = {
        ...state.currentCircuit,
        operations: [...state.currentCircuit.operations, newOperation],
        modified: new Date()
      }

      return {
        ...state,
        currentCircuit: updatedCircuit,
        circuits: state.circuits.map(circuit =>
          circuit.id === updatedCircuit.id ? updatedCircuit : circuit
        )
      }
    }

    case 'REMOVE_OPERATION': {
      if (!state.currentCircuit) return state

      const updatedCircuit = {
        ...state.currentCircuit,
        operations: state.currentCircuit.operations.filter(op => op.id !== action.operationId),
        modified: new Date()
      }

      return {
        ...state,
        currentCircuit: updatedCircuit,
        circuits: state.circuits.map(circuit =>
          circuit.id === updatedCircuit.id ? updatedCircuit : circuit
        )
      }
    }

    case 'UPDATE_OPERATION': {
      if (!state.currentCircuit) return state

      const updatedCircuit = {
        ...state.currentCircuit,
        operations: state.currentCircuit.operations.map(op =>
          op.id === action.operationId ? { ...op, ...action.updates } : op
        ),
        modified: new Date()
      }

      return {
        ...state,
        currentCircuit: updatedCircuit,
        circuits: state.circuits.map(circuit =>
          circuit.id === updatedCircuit.id ? updatedCircuit : circuit
        )
      }
    }

    case 'SET_MEASUREMENTS': {
      if (!state.currentCircuit) return state

      const updatedCircuit = {
        ...state.currentCircuit,
        measurements: action.measurements,
        modified: new Date()
      }

      return {
        ...state,
        currentCircuit: updatedCircuit,
        circuits: state.circuits.map(circuit =>
          circuit.id === updatedCircuit.id ? updatedCircuit : circuit
        )
      }
    }

    case 'START_SIMULATION':
      return {
        ...state,
        isSimulating: true
      }

    case 'FINISH_SIMULATION':
      return {
        ...state,
        isSimulating: false,
        simulationResults: [...state.simulationResults, action.result]
      }

    case 'SELECT_GATE':
      return {
        ...state,
        selectedGate: action.gate
      }

    case 'SET_MAX_QUBITS':
      return {
        ...state,
        maxQubits: action.maxQubits
      }

    default:
      return state
  }
}

interface QuantumContextType {
  state: QuantumState
  createCircuit: (name: string, qubits: number) => void
  loadCircuit: (circuit: QuantumCircuit) => void
  updateCircuit: (updates: Partial<QuantumCircuit>) => void
  addGate: (gate: string, targets: number[], params?: number[]) => void
  removeOperation: (operationId: string) => void
  updateOperation: (operationId: string, updates: Partial<QuantumOperation>) => void
  setMeasurements: (measurements: boolean[]) => void
  simulateCircuit: (shots?: number) => Promise<SimulationResult | null>
  selectGate: (gate: string | null) => void
  validateCircuit: () => Promise<boolean>
  exportCircuit: (format: 'qasm' | 'json') => string
}

const QuantumContext = createContext<QuantumContextType | null>(null)

export function QuantumProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(quantumReducer, initialState)
  const { handleApiError, addAlert } = useError()

  const createCircuit = (name: string, qubits: number) => {
    if (qubits <= 0 || qubits > state.maxQubits) {
      handleApiError(
        new Error(`Invalid qubit count: ${qubits}. Must be between 1 and ${state.maxQubits}`),
        'Circuit Creation'
      )
      return
    }

    dispatch({
      type: 'CREATE_CIRCUIT',
      circuit: {
        name,
        qubits,
        operations: [],
        measurements: new Array(qubits).fill(false)
      }
    })

    addAlert({
      type: 'success',
      title: 'Circuit Created',
      message: `Created ${name} with ${qubits} qubits`,
      persistent: false
    })
  }

  const loadCircuit = (circuit: QuantumCircuit) => {
    dispatch({ type: 'LOAD_CIRCUIT', circuit })
  }

  const updateCircuit = (updates: Partial<QuantumCircuit>) => {
    dispatch({ type: 'UPDATE_CIRCUIT', updates })
  }

  const addGate = (gate: string, targets: number[], params: number[] = []) => {
    if (!state.currentCircuit) {
      handleApiError(new Error('No circuit selected'), 'Add Gate')
      return
    }

    // Validate targets
    for (const target of targets) {
      if (target < 0 || target >= state.currentCircuit.qubits) {
        handleApiError(
          new Error(`Qubit ${target} is out of range. Circuit has ${state.currentCircuit.qubits} qubits.`),
          'Add Gate'
        )
        return
      }
    }

    // Calculate position for new operation
    const existingOps = state.currentCircuit.operations.filter(op => 
      op.targets.some(t => targets.includes(t))
    )
    const maxX = existingOps.length > 0 ? Math.max(...existingOps.map(op => op.position.x)) : 0

    dispatch({
      type: 'ADD_OPERATION',
      operation: {
        gate,
        targets,
        params,
        position: { x: maxX + 1, y: Math.min(...targets) }
      }
    })
  }

  const removeOperation = (operationId: string) => {
    dispatch({ type: 'REMOVE_OPERATION', operationId })
  }

  const updateOperation = (operationId: string, updates: Partial<QuantumOperation>) => {
    dispatch({ type: 'UPDATE_OPERATION', operationId, updates })
  }

  const setMeasurements = (measurements: boolean[]) => {
    dispatch({ type: 'SET_MEASUREMENTS', measurements })
  }

  const simulateCircuit = async (shots: number = 1000): Promise<SimulationResult | null> => {
    if (!state.currentCircuit) {
      handleApiError(new Error('No circuit to simulate'), 'Simulation')
      return null
    }

    if (state.currentCircuit.operations.length === 0) {
      handleApiError(new Error('Circuit is empty'), 'Simulation')
      return null
    }

    dispatch({ type: 'START_SIMULATION' })

    try {
      const response = await fetch('/api/quantum/circuit/simulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          circuit: {
            num_qubits: state.currentCircuit.qubits,
            operations: state.currentCircuit.operations.map(op => ({
              gate: op.gate,
              targets: op.targets,
              params: op.params
            }))
          },
          simulation: {
            shots,
            backend: 'statevector'
          }
        })
      })

      const data = await response.json()

      if (!data.success) {
        throw new Error(data.error?.message || 'Simulation failed')
      }

      const result: SimulationResult = {
        id: `sim_${Date.now()}`,
        circuitId: state.currentCircuit.id,
        counts: data.results.counts,
        shots: data.results.shots,
        executionTime: data.results.execution_time,
        timestamp: new Date()
      }

      dispatch({ type: 'FINISH_SIMULATION', result })

      addAlert({
        type: 'success',
        title: 'Simulation Complete',
        message: `Simulated ${shots} shots in ${result.executionTime.toFixed(2)}ms`,
        persistent: false
      })

      return result

    } catch (error) {
      dispatch({ type: 'FINISH_SIMULATION', result: null as any })
      handleApiError(error, 'Simulation')
      return null
    }
  }

  const selectGate = (gate: string | null) => {
    dispatch({ type: 'SELECT_GATE', gate })
  }

  const validateCircuit = async (): Promise<boolean> => {
    if (!state.currentCircuit) {
      handleApiError(new Error('No circuit to validate'), 'Validation')
      return false
    }

    try {
      const response = await fetch('/api/quantum/circuit/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          circuit: {
            num_qubits: state.currentCircuit.qubits,
            operations: state.currentCircuit.operations.map(op => ({
              gate: op.gate,
              targets: op.targets,
              params: op.params
            }))
          }
        })
      })

      const data = await response.json()

      if (data.valid) {
        addAlert({
          type: 'success',
          title: 'Circuit Valid',
          message: 'Circuit validation passed',
          persistent: false
        })

        if (data.warnings.length > 0) {
          addAlert({
            type: 'warning',
            title: 'Validation Warnings',
            message: data.warnings.join(', '),
            persistent: false
          })
        }

        return true
      } else {
        addAlert({
          type: 'error',
          title: 'Circuit Invalid',
          message: data.errors.join(', '),
          persistent: true
        })
        return false
      }

    } catch (error) {
      handleApiError(error, 'Validation')
      return false
    }
  }

  const exportCircuit = (format: 'qasm' | 'json'): string => {
    if (!state.currentCircuit) {
      throw new Error('No circuit to export')
    }

    if (format === 'json') {
      return JSON.stringify(state.currentCircuit, null, 2)
    } else if (format === 'qasm') {
      // Simple QASM export
      let qasm = `OPENQASM 2.0;\ninclude "qelib1.inc";\n\n`
      qasm += `qreg q[${state.currentCircuit.qubits}];\n`
      qasm += `creg c[${state.currentCircuit.qubits}];\n\n`

      for (const op of state.currentCircuit.operations) {
        if (op.gate === 'H') {
          qasm += `h q[${op.targets[0]}];\n`
        } else if (op.gate === 'X') {
          qasm += `x q[${op.targets[0]}];\n`
        } else if (op.gate === 'CNOT') {
          qasm += `cx q[${op.targets[0]}],q[${op.targets[1]}];\n`
        }
        // Add more gates as needed
      }

      // Add measurements
      for (let i = 0; i < state.currentCircuit.qubits; i++) {
        if (state.currentCircuit.measurements[i]) {
          qasm += `measure q[${i}] -> c[${i}];\n`
        }
      }

      return qasm
    }

    throw new Error(`Unsupported format: ${format}`)
  }

  const value: QuantumContextType = {
    state,
    createCircuit,
    loadCircuit,
    updateCircuit,
    addGate,
    removeOperation,
    updateOperation,
    setMeasurements,
    simulateCircuit,
    selectGate,
    validateCircuit,
    exportCircuit
  }

  return (
    <QuantumContext.Provider value={value}>
      {children}
    </QuantumContext.Provider>
  )
}

export function useQuantum() {
  const context = useContext(QuantumContext)
  if (!context) {
    throw new Error('useQuantum must be used within a QuantumProvider')
  }
  return context
} 