'use client'

import React, { createContext, useContext, useReducer, ReactNode } from 'react'
import toast from 'react-hot-toast'

interface ErrorState {
  errors: QuantumError[]
  alerts: QuantumAlert[]
  isReporting: boolean
  reportHistory: ErrorReport[]
}

interface QuantumError {
  id: string
  type: string
  message: string
  userMessage: string
  code: string
  category: string
  suggestions: string[]
  timestamp: Date
  component?: string
  operation?: string
}

interface QuantumAlert {
  id: string
  type: 'info' | 'warning' | 'error' | 'success'
  title: string
  message: string
  timestamp: Date
  dismissed: boolean
  persistent: boolean
  actions?: AlertAction[]
}

interface AlertAction {
  label: string
  action: string
  style: 'primary' | 'secondary' | 'danger'
}

interface ErrorReport {
  id: string
  errorId: string
  userDescription: string
  reproductionSteps: string[]
  submitted: boolean
  submittedAt?: Date
}

type ErrorAction =
  | { type: 'ADD_ERROR'; error: QuantumError }
  | { type: 'CLEAR_ERROR'; id: string }
  | { type: 'ADD_ALERT'; alert: QuantumAlert }
  | { type: 'DISMISS_ALERT'; id: string }
  | { type: 'START_REPORTING' }
  | { type: 'FINISH_REPORTING' }
  | { type: 'ADD_REPORT'; report: ErrorReport }

const initialState: ErrorState = {
  errors: [],
  alerts: [],
  isReporting: false,
  reportHistory: []
}

function errorReducer(state: ErrorState, action: ErrorAction): ErrorState {
  switch (action.type) {
    case 'ADD_ERROR':
      return {
        ...state,
        errors: [...state.errors, action.error].slice(-10) // Keep last 10 errors
      }
    
    case 'CLEAR_ERROR':
      return {
        ...state,
        errors: state.errors.filter(error => error.id !== action.id)
      }
    
    case 'ADD_ALERT':
      return {
        ...state,
        alerts: [...state.alerts, action.alert]
      }
    
    case 'DISMISS_ALERT':
      return {
        ...state,
        alerts: state.alerts.map(alert =>
          alert.id === action.id ? { ...alert, dismissed: true } : alert
        )
      }
    
    case 'START_REPORTING':
      return {
        ...state,
        isReporting: true
      }
    
    case 'FINISH_REPORTING':
      return {
        ...state,
        isReporting: false
      }
    
    case 'ADD_REPORT':
      return {
        ...state,
        reportHistory: [...state.reportHistory, action.report]
      }
    
    default:
      return state
  }
}

interface ErrorContextType {
  state: ErrorState
  addError: (error: Omit<QuantumError, 'id' | 'timestamp'>) => void
  clearError: (id: string) => void
  addAlert: (alert: Omit<QuantumAlert, 'id' | 'timestamp' | 'dismissed'>) => void
  dismissAlert: (id: string) => void
  reportError: (errorId: string, description: string, steps: string[]) => Promise<void>
  handleApiError: (error: any, context?: string) => void
}

const ErrorContext = createContext<ErrorContextType | null>(null)

export function ErrorProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(errorReducer, initialState)

  const addError = (error: Omit<QuantumError, 'id' | 'timestamp'>) => {
    const newError: QuantumError = {
      ...error,
      id: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date()
    }

    dispatch({ type: 'ADD_ERROR', error: newError })

    // Show toast notification
    toast.error(error.userMessage || error.message, {
      duration: 5000,
      style: {
        background: '#dc2626',
        color: 'white'
      }
    })

    // Auto-create alert for errors
    addAlert({
      type: 'error',
      title: `${error.category} Error`,
      message: error.userMessage || error.message,
      persistent: true,
      actions: [
        { label: 'Dismiss', action: 'dismiss', style: 'secondary' },
        { label: 'Report', action: 'report', style: 'primary' }
      ]
    })
  }

  const clearError = (id: string) => {
    dispatch({ type: 'CLEAR_ERROR', id })
  }

  const addAlert = (alert: Omit<QuantumAlert, 'id' | 'timestamp' | 'dismissed'>) => {
    const newAlert: QuantumAlert = {
      ...alert,
      id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      dismissed: false
    }

    dispatch({ type: 'ADD_ALERT', alert: newAlert })

    // Auto-dismiss non-persistent alerts
    if (!alert.persistent) {
      setTimeout(() => {
        dismissAlert(newAlert.id)
      }, 5000)
    }
  }

  const dismissAlert = (id: string) => {
    dispatch({ type: 'DISMISS_ALERT', id })
  }

  const reportError = async (errorId: string, description: string, steps: string[]) => {
    dispatch({ type: 'START_REPORTING' })

    try {
      const error = state.errors.find(e => e.id === errorId)
      if (!error) {
        throw new Error('Error not found')
      }

      // Create error report
      const report: ErrorReport = {
        id: `report_${Date.now()}`,
        errorId,
        userDescription: description,
        reproductionSteps: steps,
        submitted: false
      }

      // Submit to API
      const response = await fetch('/api/errors/report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          error,
          report
        })
      })

      if (response.ok) {
        const updatedReport = { ...report, submitted: true, submittedAt: new Date() }
        dispatch({ type: 'ADD_REPORT', report: updatedReport })
        
        toast.success('Error report submitted successfully', {
          duration: 3000
        })
      } else {
        throw new Error('Failed to submit report')
      }

    } catch (error) {
      toast.error('Failed to submit error report', {
        duration: 5000
      })
      console.error('Error reporting failed:', error)
    } finally {
      dispatch({ type: 'FINISH_REPORTING' })
    }
  }

  const handleApiError = (error: any, context?: string) => {
    console.error('API Error:', error, 'Context:', context)

    let errorData: Omit<QuantumError, 'id' | 'timestamp'>

    if (error.response?.data?.error) {
      // Structured error from our API
      const apiError = error.response.data.error
      errorData = {
        type: apiError.type || 'APIError',
        message: apiError.message || 'Unknown API error',
        userMessage: apiError.message || 'An error occurred while processing your request',
        code: apiError.code || 'API_ERROR',
        category: apiError.category || 'internal',
        suggestions: apiError.suggestions || ['Try again later', 'Check your network connection'],
        component: context || 'API',
        operation: 'API Request'
      }
    } else if (error.message) {
      // JavaScript error
      errorData = {
        type: 'ClientError',
        message: error.message,
        userMessage: 'A client-side error occurred',
        code: 'CLIENT_ERROR',
        category: 'internal',
        suggestions: ['Refresh the page', 'Clear browser cache'],
        component: context || 'Client',
        operation: 'Client Operation'
      }
    } else {
      // Unknown error
      errorData = {
        type: 'UnknownError',
        message: 'An unknown error occurred',
        userMessage: 'Something went wrong. Please try again.',
        code: 'UNKNOWN_ERROR',
        category: 'internal',
        suggestions: ['Refresh the page', 'Try again later'],
        component: context || 'Unknown',
        operation: 'Unknown Operation'
      }
    }

    addError(errorData)
  }

  const value: ErrorContextType = {
    state,
    addError,
    clearError,
    addAlert,
    dismissAlert,
    reportError,
    handleApiError
  }

  return (
    <ErrorContext.Provider value={value}>
      {children}
    </ErrorContext.Provider>
  )
}

export function useError() {
  const context = useContext(ErrorContext)
  if (!context) {
    throw new Error('useError must be used within an ErrorProvider')
  }
  return context
} 