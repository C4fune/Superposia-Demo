'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { 
  Atom, 
  Zap, 
  Shield, 
  Code, 
  BarChart3, 
  Users, 
  ArrowRight,
  Play,
  Github,
  ExternalLink
} from 'lucide-react'

const features = [
  {
    icon: Atom,
    title: 'Quantum Circuit Design',
    description: 'Visual circuit builder with drag-and-drop interface and real-time validation.'
  },
  {
    icon: Zap,
    title: 'High-Performance Simulation',
    description: 'State-of-the-art quantum simulation engine supporting up to 30+ qubits.'
  },
  {
    icon: Code,
    title: 'Python DSL',
    description: 'Intuitive Python-based domain-specific language for quantum programming.'
  },
  {
    icon: Shield,
    title: 'Error Handling',
    description: 'Comprehensive error reporting and debugging tools for quantum development.'
  },
  {
    icon: BarChart3,
    title: 'Performance Analytics',
    description: 'Detailed profiling and optimization suggestions for quantum circuits.'
  },
  {
    icon: Users,
    title: 'Collaboration',
    description: 'Share circuits, collaborate on projects, and learn from the community.'
  }
]

const stats = [
  { label: 'Quantum Gates', value: '50+' },
  { label: 'Max Qubits', value: '30+' },
  { label: 'Circuit Depth', value: '1000+' },
  { label: 'Optimization Passes', value: '20+' }
]

export default function HomePage() {
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    setIsLoaded(true)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Navigation */}
      <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <Atom className="h-8 w-8 text-quantum-600" />
              <span className="text-xl font-bold text-gray-900">Quantum Platform</span>
            </div>
            <div className="hidden md:flex space-x-6">
              <Link href="/editor" className="text-gray-600 hover:text-gray-900 transition-colors">
                Circuit Editor
              </Link>
              <Link href="/examples" className="text-gray-600 hover:text-gray-900 transition-colors">
                Examples
              </Link>
              <Link href="/docs" className="text-gray-600 hover:text-gray-900 transition-colors">
                Documentation
              </Link>
              <Link href="/about" className="text-gray-600 hover:text-gray-900 transition-colors">
                About
              </Link>
            </div>
            <div className="flex space-x-3">
              <Link 
                href="/editor"
                className="quantum-button flex items-center space-x-2"
              >
                <Play className="h-4 w-4" />
                <span>Get Started</span>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 sm:py-32">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: isLoaded ? 1 : 0, y: isLoaded ? 0 : 20 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-4xl sm:text-6xl lg:text-7xl font-bold text-gray-900 mb-6">
              Next-Generation
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-quantum-600 to-purple-600">
                Quantum Computing
              </span>
              Platform
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-10">
              Design, simulate, and execute quantum circuits with our comprehensive platform. 
              From beginners to researchers, unlock the power of quantum computing.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link 
                href="/editor"
                className="quantum-button text-lg px-8 py-4 flex items-center justify-center space-x-2"
              >
                <Play className="h-5 w-5" />
                <span>Start Building</span>
                <ArrowRight className="h-5 w-5" />
              </Link>
              <a 
                href="https://github.com/quantum-platform"
                target="_blank"
                rel="noopener noreferrer"
                className="quantum-button-secondary text-lg px-8 py-4 flex items-center justify-center space-x-2"
              >
                <Github className="h-5 w-5" />
                <span>View on GitHub</span>
                <ExternalLink className="h-4 w-4" />
              </a>
            </div>
          </motion.div>
        </div>

        {/* Background decoration */}
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-quantum-200 rounded-full opacity-20 blur-3xl"></div>
          <div className="absolute top-1/4 right-1/4 w-64 h-64 bg-purple-200 rounded-full opacity-20 blur-3xl"></div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="text-3xl font-bold text-quantum-600 mb-2">{stat.value}</div>
                <div className="text-gray-600">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              Everything You Need for Quantum Development
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our platform provides a complete toolkit for quantum computing research and development.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="quantum-card p-6 hover:shadow-lg transition-shadow duration-300"
              >
                <feature.icon className="h-12 w-12 text-quantum-600 mb-4" />
                <h3 className="text-xl font-semibold text-gray-900 mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-quantum-600 to-purple-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
              Ready to Build Quantum Circuits?
            </h2>
            <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
              Join thousands of researchers and developers already using our platform 
              to explore the quantum frontier.
            </p>
            <Link 
              href="/editor"
              className="inline-flex items-center space-x-2 bg-white text-quantum-600 px-8 py-4 rounded-lg font-semibold hover:bg-gray-50 transition-colors duration-200"
            >
              <span>Start Your Quantum Journey</span>
              <ArrowRight className="h-5 w-5" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <Atom className="h-8 w-8 text-quantum-400" />
                <span className="text-xl font-bold">Quantum Platform</span>
              </div>
              <p className="text-gray-400">
                The most comprehensive quantum computing platform for research and development.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Platform</h3>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/editor" className="hover:text-white transition-colors">Circuit Editor</Link></li>
                <li><Link href="/simulator" className="hover:text-white transition-colors">Quantum Simulator</Link></li>
                <li><Link href="/examples" className="hover:text-white transition-colors">Examples</Link></li>
                <li><Link href="/docs" className="hover:text-white transition-colors">Documentation</Link></li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Resources</h3>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/tutorials" className="hover:text-white transition-colors">Tutorials</Link></li>
                <li><Link href="/api-docs" className="hover:text-white transition-colors">API Reference</Link></li>
                <li><Link href="/community" className="hover:text-white transition-colors">Community</Link></li>
                <li><Link href="/support" className="hover:text-white transition-colors">Support</Link></li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Connect</h3>
              <ul className="space-y-2 text-gray-400">
                <li><a href="https://github.com/quantum-platform" className="hover:text-white transition-colors">GitHub</a></li>
                <li><a href="https://twitter.com/quantumplatform" className="hover:text-white transition-colors">Twitter</a></li>
                <li><a href="https://discord.gg/quantum" className="hover:text-white transition-colors">Discord</a></li>
                <li><Link href="/contact" className="hover:text-white transition-colors">Contact</Link></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 Quantum Platform. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
} 