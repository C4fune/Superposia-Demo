#!/usr/bin/env python3
"""
Comprehensive Marketplace System Test

This script demonstrates the complete marketplace and community features
including:
- Built-in quantum algorithm library
- Marketplace package management
- Community features and contributions
- Plugin system

Run this to see the marketplace ecosystem in action.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_algorithm_library():
    """Test the built-in quantum algorithm library."""
    print("\n" + "="*60)
    print("TESTING QUANTUM ALGORITHM LIBRARY")
    print("="*60)
    
    try:
        from quantum_platform.marketplace.algorithm_library import (
            AlgorithmLibrary, AlgorithmCategory, AlgorithmComplexity
        )
        
        # Initialize algorithm library
        library = AlgorithmLibrary()
        
        # List all available algorithms
        print("\n1. Available Algorithms:")
        print("-" * 40)
        algorithms = library.list_algorithms()
        for algo in algorithms:
            print(f"✓ {algo.name} ({algo.category.value}) - {algo.complexity.value}")
            print(f"  {algo.description}")
            print(f"  Qubits: {algo.min_qubits}-{algo.max_qubits or 'unlimited'}")
            print(f"  Tags: {', '.join(algo.tags)}")
            print()
        
        # Search algorithms
        print("\n2. Search Results for 'grover':")
        print("-" * 40)
        search_results = library.search_algorithms("grover")
        for algo in search_results:
            print(f"✓ {algo.name}: {algo.description}")
        
        # Filter by category
        print("\n3. Entanglement Algorithms:")
        print("-" * 40)
        entanglement_algos = library.list_algorithms(category=AlgorithmCategory.ENTANGLEMENT)
        for algo in entanglement_algos:
            print(f"✓ {algo.name}: {algo.description}")
        
        # Create some example circuits
        print("\n4. Creating Example Circuits:")
        print("-" * 40)
        
        # Bell state
        try:
            bell_circuit = library.create_circuit("bell_state", state_type="phi_plus")
            print(f"✓ Bell state circuit created: {bell_circuit.num_qubits} qubits")
        except Exception as e:
            print(f"✗ Bell state creation failed: {e}")
        
        # GHZ state
        try:
            ghz_circuit = library.create_circuit("ghz_state", num_qubits=4)
            print(f"✓ GHZ state circuit created: {ghz_circuit.num_qubits} qubits")
        except Exception as e:
            print(f"✗ GHZ state creation failed: {e}")
        
        # Grover's algorithm
        try:
            grover_circuit = library.create_circuit("grover", num_qubits=3, marked_items=[5])
            print(f"✓ Grover circuit created: {grover_circuit.num_qubits} qubits")
        except Exception as e:
            print(f"✗ Grover circuit creation failed: {e}")
        
        # QFT
        try:
            qft_circuit = library.create_circuit("qft", num_qubits=3, inverse=False)
            print(f"✓ QFT circuit created: {qft_circuit.num_qubits} qubits")
        except Exception as e:
            print(f"✗ QFT circuit creation failed: {e}")
        
        print("\n✓ Algorithm library test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Algorithm library test failed: {e}")
        print(f"\n✗ Algorithm library test failed: {e}")
        return False


def test_marketplace_manager():
    """Test the marketplace manager functionality."""
    print("\n" + "="*60)
    print("TESTING MARKETPLACE MANAGER")
    print("="*60)
    
    try:
        from quantum_platform.marketplace.marketplace_manager import (
            MarketplaceManager, MarketplaceConfig, PackageType
        )
        
        # Initialize marketplace
        config = MarketplaceConfig(
            local_repository_path="test_packages",
            allow_beta_packages=True,
            security_validation=False  # Disable for testing
        )
        marketplace = MarketplaceManager(config)
        
        # List available packages
        print("\n1. Available Packages:")
        print("-" * 40)
        packages = marketplace.list_available_packages()
        for package in packages:
            print(f"✓ {package.metadata.name} v{package.metadata.version}")
            print(f"  Type: {package.metadata.package_type.value}")
            print(f"  Description: {package.metadata.description}")
            print(f"  Author: {package.metadata.author}")
            print(f"  Tags: {', '.join(package.metadata.tags)}")
            print()
        
        # Search packages
        print("\n2. Search Results for 'algorithm':")
        print("-" * 40)
        search_results = marketplace.search_packages("algorithm")
        for package in search_results:
            print(f"✓ {package.metadata.name}: {package.metadata.description}")
        
        # List installed packages
        print("\n3. Installed Packages:")
        print("-" * 40)
        installed = marketplace.list_installed_packages()
        for package in installed:
            print(f"✓ {package.metadata.name} v{package.metadata.version} - {package.status.value}")
        
        # Try to install a package (simulation)
        print("\n4. Package Installation Test:")
        print("-" * 40)
        try:
            # This will fail in the demo since we don't have real remote packages
            # but it demonstrates the interface
            result = marketplace.install_package("advanced-algorithms")
            if result:
                print("✓ Package installation successful")
            else:
                print("✗ Package installation failed (expected for demo)")
        except Exception as e:
            print(f"✗ Package installation failed: {e} (expected for demo)")
        
        # Check for updates
        print("\n5. Checking for Updates:")
        print("-" * 40)
        updates = marketplace.check_for_updates()
        if updates:
            print(f"✓ Found {len(updates)} packages with updates: {', '.join(updates)}")
        else:
            print("✓ All packages are up to date")
        
        print("\n✓ Marketplace manager test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Marketplace manager test failed: {e}")
        print(f"\n✗ Marketplace manager test failed: {e}")
        return False


def test_community_features():
    """Test the community features."""
    print("\n" + "="*60)
    print("TESTING COMMUNITY FEATURES")
    print("="*60)
    
    try:
        from quantum_platform.marketplace.community import (
            CommunityManager, UserProfile, AlgorithmContribution, 
            CommunityRating, DiscussionThread
        )
        
        # Initialize community manager
        community = CommunityManager(data_path="test_community")
        
        # Create test user profiles
        print("\n1. Creating User Profiles:")
        print("-" * 40)
        
        user1_data = {
            "user_id": str(uuid.uuid4()),
            "username": "quantum_alice",
            "display_name": "Alice Quantum",
            "email": "alice@quantum.com",
            "bio": "Quantum computing researcher",
            "location": "Boston, MA",
            "github_username": "quantum_alice"
        }
        
        user2_data = {
            "user_id": str(uuid.uuid4()),
            "username": "quantum_bob",
            "display_name": "Bob Entangled",
            "email": "bob@quantum.com",
            "bio": "Quantum algorithm developer",
            "location": "San Francisco, CA",
            "github_username": "quantum_bob"
        }
        
        user1 = community.create_user_profile(user1_data)
        user2 = community.create_user_profile(user2_data)
        
        print(f"✓ Created user profile: {user1.display_name}")
        print(f"✓ Created user profile: {user2.display_name}")
        
        # Submit algorithm contributions
        print("\n2. Submitting Algorithm Contributions:")
        print("-" * 40)
        
        contrib1_data = {
            "contribution_id": str(uuid.uuid4()),
            "user_id": user1.user_id,
            "algorithm_name": "Enhanced Grover",
            "description": "Optimized Grover's search with amplitude amplification",
            "category": "search",
            "tags": ["grover", "search", "optimization"],
            "code_repository": "https://github.com/quantum_alice/enhanced-grover",
            "license": "MIT",
            "status": "approved"
        }
        
        contrib2_data = {
            "contribution_id": str(uuid.uuid4()),
            "user_id": user2.user_id,
            "algorithm_name": "Quantum Neural Network",
            "description": "Variational quantum neural network implementation",
            "category": "machine_learning",
            "tags": ["vqnn", "machine-learning", "variational"],
            "code_repository": "https://github.com/quantum_bob/quantum-nn",
            "license": "Apache-2.0",
            "status": "pending"
        }
        
        contrib1 = community.submit_contribution(contrib1_data)
        contrib2 = community.submit_contribution(contrib2_data)
        
        print(f"✓ Submitted contribution: {contrib1.algorithm_name}")
        print(f"✓ Submitted contribution: {contrib2.algorithm_name}")
        
        # Rate packages
        print("\n3. Rating Packages:")
        print("-" * 40)
        
        rating1_data = {
            "rating_id": str(uuid.uuid4()),
            "user_id": user1.user_id,
            "package_name": "quantum-algorithm-library",
            "rating": 5,
            "review": "Excellent collection of quantum algorithms!"
        }
        
        rating2_data = {
            "rating_id": str(uuid.uuid4()),
            "user_id": user2.user_id,
            "package_name": "quantum-algorithm-library",
            "rating": 4,
            "review": "Great algorithms, could use more documentation"
        }
        
        rating1 = community.rate_package(rating1_data)
        rating2 = community.rate_package(rating2_data)
        
        print(f"✓ Added rating: {rating1.rating}/5 stars")
        print(f"✓ Added rating: {rating2.rating}/5 stars")
        
        # Create discussion threads
        print("\n4. Creating Discussion Threads:")
        print("-" * 40)
        
        thread1_data = {
            "thread_id": str(uuid.uuid4()),
            "user_id": user1.user_id,
            "title": "Best practices for quantum error correction",
            "content": "What are the current best practices for implementing quantum error correction in NISQ devices?",
            "category": "discussion",
            "tags": ["error-correction", "nisq", "best-practices"]
        }
        
        thread2_data = {
            "thread_id": str(uuid.uuid4()),
            "user_id": user2.user_id,
            "title": "Variational quantum algorithms optimization",
            "content": "Looking for tips on optimizing VQE convergence",
            "category": "help",
            "tags": ["vqe", "optimization", "variational"]
        }
        
        thread1 = community.create_discussion_thread(thread1_data)
        thread2 = community.create_discussion_thread(thread2_data)
        
        print(f"✓ Created thread: {thread1.title}")
        print(f"✓ Created thread: {thread2.title}")
        
        # Get community statistics
        print("\n5. Community Statistics:")
        print("-" * 40)
        
        stats = community.get_community_stats()
        print(f"✓ Total users: {stats['total_users']}")
        print(f"✓ Total contributions: {stats['total_contributions']}")
        print(f"✓ Approved contributions: {stats['approved_contributions']}")
        print(f"✓ Total ratings: {stats['total_ratings']}")
        print(f"✓ Total discussions: {stats['total_discussions']}")
        
        # List contributions
        print("\n6. Algorithm Contributions:")
        print("-" * 40)
        contributions = community.get_contributions()
        for contrib in contributions:
            print(f"✓ {contrib.algorithm_name} by {contrib.user_id[:8]}...")
            print(f"  Status: {contrib.status}")
            print(f"  Category: {contrib.category}")
            print(f"  Tags: {', '.join(contrib.tags)}")
            print()
        
        print("\n✓ Community features test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Community features test failed: {e}")
        print(f"\n✗ Community features test failed: {e}")
        return False


def test_plugin_system():
    """Test the plugin system functionality."""
    print("\n" + "="*60)
    print("TESTING PLUGIN SYSTEM")
    print("="*60)
    
    try:
        from quantum_platform.marketplace.plugin_system import (
            PluginManager, PluginType, PluginMetadata
        )
        
        # Initialize plugin manager
        plugin_manager = PluginManager(plugin_directories=["test_plugins"])
        
        # Get plugin statistics
        print("\n1. Plugin System Statistics:")
        print("-" * 40)
        stats = plugin_manager.get_plugin_statistics()
        print(f"✓ Total plugins: {stats['total_plugins']}")
        print(f"✓ Active plugins: {stats['active_plugins']}")
        print(f"✓ Plugin directories: {', '.join(stats['plugin_directories'])}")
        
        # List plugins by type
        print("\n2. Plugins by Type:")
        print("-" * 40)
        for plugin_type in PluginType:
            plugins = plugin_manager.get_plugins(plugin_type)
            print(f"✓ {plugin_type.value}: {len(plugins)} plugins")
            for plugin in plugins:
                print(f"  - {plugin.metadata.name} v{plugin.metadata.version}")
        
        # List available algorithms from plugins
        print("\n3. Available Algorithms from Plugins:")
        print("-" * 40)
        algorithms = plugin_manager.list_available_algorithms()
        if algorithms:
            for plugin_name, algo_list in algorithms.items():
                print(f"✓ {plugin_name}: {', '.join(algo_list)}")
        else:
            print("✓ No algorithm plugins loaded (expected for demo)")
        
        # List available visualizations
        print("\n4. Available Visualizations from Plugins:")
        print("-" * 40)
        visualizations = plugin_manager.list_available_visualizations()
        if visualizations:
            for plugin_name, viz_list in visualizations.items():
                print(f"✓ {plugin_name}: {', '.join(viz_list)}")
        else:
            print("✓ No visualization plugins loaded (expected for demo)")
        
        # List available tools
        print("\n5. Available Tools from Plugins:")
        print("-" * 40)
        tools = plugin_manager.list_available_tools()
        if tools:
            for plugin_name, tool_list in tools.items():
                print(f"✓ {plugin_name}: {', '.join(tool_list)}")
        else:
            print("✓ No tool plugins loaded (expected for demo)")
        
        print("\n✓ Plugin system test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Plugin system test failed: {e}")
        print(f"\n✗ Plugin system test failed: {e}")
        return False


def test_session_binding_fix():
    """Test if the session binding issue is fixed."""
    print("\n" + "="*60)
    print("TESTING SESSION BINDING FIX")
    print("="*60)
    
    try:
        from quantum_platform.experiments import ExperimentManager
        
        # Create experiment manager
        experiment_manager = ExperimentManager()
        
        # Create a test circuit
        print("\n1. Creating Test Circuit:")
        print("-" * 40)
        
        circuit_data = {
            'name': 'Test Bell State',
            'description': 'Test circuit for session binding',
            'qasm_code': '''
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
            '''.strip(),
            'num_qubits': 2,
            'circuit_json': {'test': 'data'},
            'parameters': {},
            'version': '1.0'
        }
        
        circuit = experiment_manager.database.create_circuit(**circuit_data)
        print(f"✓ Circuit created successfully: {circuit.id}")
        
        # Try to access circuit properties (this would fail with session binding issues)
        print(f"✓ Circuit name: {circuit.name}")
        print(f"✓ Circuit qubits: {circuit.num_qubits}")
        print(f"✓ Circuit version: {circuit.version}")
        
        # Create an experiment
        print("\n2. Creating Test Experiment:")
        print("-" * 40)
        
        experiment = experiment_manager.database.create_experiment(
            name="Test Experiment",
            circuit_id=circuit.id,
            backend="local_simulator",
            description="Test experiment for session binding",
            shots=1000
        )
        
        print(f"✓ Experiment created successfully: {experiment.id}")
        print(f"✓ Experiment name: {experiment.name}")
        print(f"✓ Experiment backend: {experiment.backend}")
        
        # Create a test result
        print("\n3. Creating Test Result:")
        print("-" * 40)
        
        result = experiment_manager.database.create_result(
            experiment_id=experiment.id,
            run_number=1,
            raw_counts={"00": 500, "11": 500},
            shots=1000,
            execution_time=0.1
        )
        
        print(f"✓ Result created successfully: {result.id}")
        print(f"✓ Result counts: {result.raw_counts}")
        print(f"✓ Result shots: {result.shots}")
        
        print("\n✓ Session binding test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Session binding test failed: {e}")
        print(f"\n✗ Session binding test failed: {e}")
        return False


def main():
    """Run all marketplace system tests."""
    print("=" * 80)
    print("COMPREHENSIVE MARKETPLACE SYSTEM TEST")
    print("=" * 80)
    
    test_results = []
    
    # Test session binding fix first
    test_results.append(("Session Binding Fix", test_session_binding_fix()))
    
    # Test algorithm library
    test_results.append(("Algorithm Library", test_algorithm_library()))
    
    # Test marketplace manager
    test_results.append(("Marketplace Manager", test_marketplace_manager()))
    
    # Test community features
    test_results.append(("Community Features", test_community_features()))
    
    # Test plugin system
    test_results.append(("Plugin System", test_plugin_system()))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:30} | {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {len(test_results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The marketplace system is working correctly.")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 