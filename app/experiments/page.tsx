'use client'

import React, { useState, useEffect } from 'react'
import { useError } from '@/contexts/ErrorContext'
import { 
  Play, 
  Square, 
  Trash2, 
  Eye, 
  Download,
  Plus,
  Filter,
  Search,
  Calendar,
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle,
  BarChart3,
  TrendingUp,
  Database,
  Atom,
  Settings
} from 'lucide-react'

interface Experiment {
  id: string
  name: string
  description?: string
  experiment_type: string
  status: string
  backend: string
  provider?: string
  device_name?: string
  shots: number
  total_runs: number
  successful_runs: number
  failed_runs: number
  created_at: string
  started_at?: string
  completed_at?: string
  created_by?: string
  tags?: string[]
  avg_execution_time?: number
  success_rate?: number
}

interface ExperimentSummary {
  experiment_id: string
  name: string
  experiment_type: string
  status: string
  backend: string
  total_runs: number
  successful_runs: number
  failed_runs: number
  avg_execution_time?: number
  created_at: string
  completed_at?: string
  success_rate: number
}

interface DatabaseStats {
  total_circuits: number
  total_experiments: number
  total_results: number
  experiments_by_status: Record<string, number>
  experiments_by_backend: Record<string, number>
  experiments_by_type: Record<string, number>
  recent_activity: {
    experiments_last_24h: number
    results_last_24h: number
  }
}

export default function ExperimentsPage() {
  const { addAlert } = useError()
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([])
  const [filteredExperiments, setFilteredExperiments] = useState<ExperimentSummary[]>([])
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [stats, setStats] = useState<DatabaseStats | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [backendFilter, setBackendFilter] = useState('all')
  const [typeFilter, setTypeFilter] = useState('all')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showDetailsModal, setShowDetailsModal] = useState(false)
  const [activeExperiments, setActiveExperiments] = useState<Record<string, any>>({})

  // Load experiments and stats on mount
  useEffect(() => {
    loadExperiments()
    loadStats()
    loadActiveExperiments()
    
    // Set up polling for active experiments
    const interval = setInterval(() => {
      loadActiveExperiments()
    }, 5000)
    
    return () => clearInterval(interval)
  }, [])

  // Filter experiments when search term or filters change
  useEffect(() => {
    let filtered = experiments

    if (searchTerm) {
      filtered = filtered.filter(exp => 
        exp.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        exp.experiment_type.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    if (statusFilter !== 'all') {
      filtered = filtered.filter(exp => exp.status === statusFilter)
    }

    if (backendFilter !== 'all') {
      filtered = filtered.filter(exp => exp.backend === backendFilter)
    }

    if (typeFilter !== 'all') {
      filtered = filtered.filter(exp => exp.experiment_type === typeFilter)
    }

    setFilteredExperiments(filtered)
  }, [experiments, searchTerm, statusFilter, backendFilter, typeFilter])

  const loadExperiments = async () => {
    try {
      setIsLoading(true)
      const response = await fetch('/api/experiments/list')
      const data = await response.json()
      
      if (data.success) {
        setExperiments(data.data.experiments)
      } else {
        addAlert({
          type: 'error',
          title: 'Failed to Load Experiments',
          message: data.error?.message || 'Unknown error occurred',
          persistent: true
        })
      }
    } catch (error) {
      addAlert({
        type: 'error',
        title: 'Network Error',
        message: 'Failed to connect to experiment API',
        persistent: true
      })
    } finally {
      setIsLoading(false)
    }
  }

  const loadStats = async () => {
    try {
      const response = await fetch('/api/experiments/stats')
      const data = await response.json()
      
      if (data.success) {
        setStats(data.data)
      }
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }

  const loadActiveExperiments = async () => {
    try {
      const response = await fetch('/api/experiments/active')
      const data = await response.json()
      
      if (data.success) {
        setActiveExperiments(data.data)
      }
    } catch (error) {
      console.error('Failed to load active experiments:', error)
    }
  }

  const handleRunExperiment = async (experimentId: string) => {
    try {
      const response = await fetch(`/api/experiments/${experimentId}/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ async: true })
      })
      
      const data = await response.json()
      
      if (data.success) {
        addAlert({
          type: 'success',
          title: 'Experiment Started',
          message: `Experiment ${experimentId} is now running`,
          persistent: false
        })
        loadActiveExperiments()
      } else {
        addAlert({
          type: 'error',
          title: 'Failed to Start Experiment',
          message: data.error?.message || 'Unknown error occurred',
          persistent: true
        })
      }
    } catch (error) {
      addAlert({
        type: 'error',
        title: 'Network Error',
        message: 'Failed to start experiment',
        persistent: true
      })
    }
  }

  const handleDeleteExperiment = async (experimentId: string) => {
    if (!confirm('Are you sure you want to delete this experiment? This action cannot be undone.')) {
      return
    }

    try {
      const response = await fetch(`/api/experiments/${experimentId}/delete`, {
        method: 'POST'
      })
      
      const data = await response.json()
      
      if (data.success) {
        addAlert({
          type: 'success',
          title: 'Experiment Deleted',
          message: 'Experiment has been permanently deleted',
          persistent: false
        })
        loadExperiments()
      } else {
        addAlert({
          type: 'error',
          title: 'Failed to Delete Experiment',
          message: data.error?.message || 'Unknown error occurred',
          persistent: true
        })
      }
    } catch (error) {
      addAlert({
        type: 'error',
        title: 'Network Error',
        message: 'Failed to delete experiment',
        persistent: true
      })
    }
  }

  const handleViewDetails = async (experimentId: string) => {
    try {
      const response = await fetch(`/api/experiments/${experimentId}`)
      const data = await response.json()
      
      if (data.success) {
        setSelectedExperiment(data.data)
        setShowDetailsModal(true)
      } else {
        addAlert({
          type: 'error',
          title: 'Failed to Load Experiment',
          message: data.error?.message || 'Unknown error occurred',
          persistent: true
        })
      }
    } catch (error) {
      addAlert({
        type: 'error',
        title: 'Network Error',
        message: 'Failed to load experiment details',
        persistent: true
      })
    }
  }

  const handleExportExperiment = async (experimentId: string, format: 'json' | 'csv') => {
    try {
      const response = await fetch(`/api/experiments/${experimentId}/export?format=${format}`)
      
      if (format === 'csv') {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `experiment_${experimentId}.csv`
        a.click()
        window.URL.revokeObjectURL(url)
      } else {
        const data = await response.json()
        if (data.success) {
          const blob = new Blob([JSON.stringify(data.data, null, 2)], { type: 'application/json' })
          const url = window.URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = `experiment_${experimentId}.json`
          a.click()
          window.URL.revokeObjectURL(url)
        }
      }
      
      addAlert({
        type: 'success',
        title: 'Export Complete',
        message: `Experiment data exported as ${format.toUpperCase()}`,
        persistent: false
      })
    } catch (error) {
      addAlert({
        type: 'error',
        title: 'Export Failed',
        message: 'Failed to export experiment data',
        persistent: true
      })
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-4 w-4 text-green-600" />
      case 'running': return <Play className="h-4 w-4 text-blue-600" />
      case 'failed': return <XCircle className="h-4 w-4 text-red-600" />
      case 'queued': return <Clock className="h-4 w-4 text-yellow-600" />
      default: return <AlertCircle className="h-4 w-4 text-gray-600" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800'
      case 'running': return 'bg-blue-100 text-blue-800'
      case 'failed': return 'bg-red-100 text-red-800'
      case 'queued': return 'bg-yellow-100 text-yellow-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const formatDuration = (startTime: string, endTime?: string) => {
    const start = new Date(startTime)
    const end = endTime ? new Date(endTime) : new Date()
    const duration = end.getTime() - start.getTime()
    
    const hours = Math.floor(duration / (1000 * 60 * 60))
    const minutes = Math.floor((duration % (1000 * 60 * 60)) / (1000 * 60))
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`
    } else {
      return `${minutes}m`
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="quantum-spinner w-12 h-12 mx-auto mb-4" />
          <p className="text-gray-600">Loading experiments...</p>
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
              <Database className="h-8 w-8 text-quantum-600" />
              <div>
                <h1 className="text-xl font-semibold text-gray-900">Experiment Tracker</h1>
                <p className="text-sm text-gray-500">
                  Manage and analyze quantum experiments
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={() => setShowCreateModal(true)}
                className="quantum-button flex items-center space-x-2"
              >
                <Plus className="h-4 w-4" />
                <span>New Experiment</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Stats Overview */}
      {stats && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Total Experiments</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.total_experiments}</p>
                </div>
                <BarChart3 className="h-8 w-8 text-blue-600" />
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Total Results</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.total_results}</p>
                </div>
                <TrendingUp className="h-8 w-8 text-green-600" />
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Circuits</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.total_circuits}</p>
                </div>
                <Atom className="h-8 w-8 text-purple-600" />
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Active Today</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.recent_activity.experiments_last_24h}</p>
                </div>
                <Calendar className="h-8 w-8 text-orange-600" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Filters and Search */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex flex-wrap gap-4 items-center">
            <div className="flex-1 min-w-64">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search experiments..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="quantum-input pl-10"
                />
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="quantum-select"
              >
                <option value="all">All Status</option>
                <option value="completed">Completed</option>
                <option value="running">Running</option>
                <option value="failed">Failed</option>
                <option value="queued">Queued</option>
              </select>
              
              <select
                value={backendFilter}
                onChange={(e) => setBackendFilter(e.target.value)}
                className="quantum-select"
              >
                <option value="all">All Backends</option>
                <option value="local_simulator">Local Simulator</option>
                <option value="noisy_simulator">Noisy Simulator</option>
                <option value="ibm_quantum">IBM Quantum</option>
              </select>
              
              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
                className="quantum-select"
              >
                <option value="all">All Types</option>
                <option value="single_shot">Single Shot</option>
                <option value="parameter_sweep">Parameter Sweep</option>
                <option value="optimization">Optimization</option>
                <option value="benchmarking">Benchmarking</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Experiments List */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-8">
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">
              Experiments ({filteredExperiments.length})
            </h2>
          </div>
          
          {filteredExperiments.length === 0 ? (
            <div className="text-center py-12">
              <Database className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No experiments found</p>
              <p className="text-sm text-gray-400">Try adjusting your filters or create a new experiment</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Backend
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Results
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Success Rate
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Created
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredExperiments.map((experiment) => (
                    <tr key={experiment.experiment_id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {experiment.name}
                          </div>
                          <div className="text-sm text-gray-500">
                            ID: {experiment.experiment_id.slice(0, 8)}...
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(experiment.status)}
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(experiment.status)}`}>
                            {experiment.status}
                          </span>
                          {activeExperiments[experiment.experiment_id] && (
                            <span className="text-xs text-blue-600">
                              {Math.round(activeExperiments[experiment.experiment_id].progress * 100)}%
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {experiment.experiment_type.replace('_', ' ')}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {experiment.backend.replace('_', ' ')}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {experiment.successful_runs}/{experiment.total_runs}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                            <div 
                              className="bg-green-600 h-2 rounded-full" 
                              style={{ width: `${experiment.success_rate * 100}%` }}
                            />
                          </div>
                          <span className="text-sm text-gray-900">
                            {Math.round(experiment.success_rate * 100)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {formatDate(experiment.created_at)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => handleViewDetails(experiment.experiment_id)}
                            className="text-blue-600 hover:text-blue-900"
                            title="View Details"
                          >
                            <Eye className="h-4 w-4" />
                          </button>
                          
                          {experiment.status !== 'running' && (
                            <button
                              onClick={() => handleRunExperiment(experiment.experiment_id)}
                              className="text-green-600 hover:text-green-900"
                              title="Run Experiment"
                            >
                              <Play className="h-4 w-4" />
                            </button>
                          )}
                          
                          <button
                            onClick={() => handleExportExperiment(experiment.experiment_id, 'json')}
                            className="text-purple-600 hover:text-purple-900"
                            title="Export Data"
                          >
                            <Download className="h-4 w-4" />
                          </button>
                          
                          <button
                            onClick={() => handleDeleteExperiment(experiment.experiment_id)}
                            className="text-red-600 hover:text-red-900"
                            title="Delete Experiment"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Create Experiment Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-4">Create New Experiment</h3>
            <p className="text-gray-600 mb-4">
              This feature will be available soon. For now, you can create experiments programmatically using the API.
            </p>
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowCreateModal(false)}
                className="quantum-button-secondary"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Experiment Details Modal */}
      {showDetailsModal && selectedExperiment && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-start mb-6">
              <div>
                <h3 className="text-lg font-semibold">{selectedExperiment.name}</h3>
                <p className="text-gray-600">{selectedExperiment.description}</p>
              </div>
              <button
                onClick={() => setShowDetailsModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <XCircle className="h-6 w-6" />
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3">Experiment Info</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">ID:</span>
                    <span className="font-mono">{selectedExperiment.id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Type:</span>
                    <span>{selectedExperiment.experiment_type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Backend:</span>
                    <span>{selectedExperiment.backend}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Shots:</span>
                    <span>{selectedExperiment.shots}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Status:</span>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(selectedExperiment.status)}`}>
                      {selectedExperiment.status}
                    </span>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium mb-3">Results Summary</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Total Runs:</span>
                    <span>{selectedExperiment.total_runs}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Successful:</span>
                    <span className="text-green-600">{selectedExperiment.successful_runs}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Failed:</span>
                    <span className="text-red-600">{selectedExperiment.failed_runs}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Success Rate:</span>
                    <span>{Math.round((selectedExperiment.success_rate || 0) * 100)}%</span>
                  </div>
                  {selectedExperiment.avg_execution_time && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Avg Time:</span>
                      <span>{selectedExperiment.avg_execution_time.toFixed(1)}ms</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            <div className="mt-6 flex justify-end space-x-3">
              <button
                onClick={() => handleExportExperiment(selectedExperiment.id, 'json')}
                className="quantum-button-secondary flex items-center space-x-2"
              >
                <Download className="h-4 w-4" />
                <span>Export JSON</span>
              </button>
              <button
                onClick={() => handleExportExperiment(selectedExperiment.id, 'csv')}
                className="quantum-button-secondary flex items-center space-x-2"
              >
                <Download className="h-4 w-4" />
                <span>Export CSV</span>
              </button>
              <button
                onClick={() => setShowDetailsModal(false)}
                className="quantum-button"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
} 