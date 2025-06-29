#!/usr/bin/env python3
"""
RBAC Usage Example

This script demonstrates practical usage of the Role-Based Access Control system
in various quantum computing scenarios.
"""

from quantum_platform.security import (
    initialize_security, SecurityContext, AdminContext, 
    UserRole, Permission, get_quantum_security,
    require_permission, require_role
)
from quantum_platform.compiler.language.dsl import QuantumProgram, allocate
from quantum_platform.compiler.language.operations import H, CNOT, RX
from quantum_platform.simulation.executor import SimulationExecutor
from quantum_platform.security.enforcement import PermissionDeniedException


def main():
    """Demonstrate RBAC usage scenarios."""
    
    print("🔐 Quantum Platform RBAC Usage Examples")
    print("=" * 50)
    
    # Initialize the security system
    security = initialize_security()
    
    # Scenario 1: Educational Setting
    print("\n📚 Scenario 1: Educational Setting")
    print("Professor manages quantum computing class")
    
    # Professor (admin) sets up the system
    with AdminContext():
        print("✅ Professor authenticated with admin privileges")
        
        # Create student accounts
        for i, student_name in enumerate(["alice", "bob", "charlie"], 1):
            security.create_user_session(student_name, UserRole.STANDARD_USER.value)
            print(f"  👨‍🎓 Created student account: {student_name}")
        
        # Create TA account
        security.create_user_session("teaching_assistant", UserRole.DEVELOPER.value)
        print("  👩‍🏫 Created teaching assistant account with advanced privileges")
    
    # Scenario 2: Student Working on Assignment
    print("\n📝 Scenario 2: Student Assignment Work")
    
    with SecurityContext("alice", UserRole.STANDARD_USER.value):
        print("Student Alice working on Bell state assignment...")
        
        try:
            # Student can create and simulate circuits
            with QuantumProgram(name="alice_bell_state") as qp:
                qubits = allocate(2)
                H(qubits[0])
                CNOT(qubits[0], qubits[1])
            
            executor = SimulationExecutor()
            result = executor.run(qp.circuit, shots=1000)
            print(f"  ✅ Assignment completed! Results: {len(result.measurement_counts)} outcomes")
            
        except PermissionDeniedException as e:
            print(f"  ❌ Permission denied: {e}")
    
    # Scenario 3: Research Collaboration
    print("\n🔬 Scenario 3: Research Collaboration")
    
    # Create a custom researcher role
    with AdminContext():
        role_manager = security.role_manager
        try:
            researcher_role = role_manager.create_role(
                name="quantum_researcher",
                description="Researcher with simulation and hardware access",
                permissions={
                    Permission.CREATE_CIRCUIT,
                    Permission.EDIT_CIRCUIT,
                    Permission.RUN_SIMULATION,
                    Permission.RUN_HARDWARE,
                    Permission.VIEW_CIRCUIT,
                    Permission.EXPORT_CIRCUIT
                },
                inherits_from=UserRole.STANDARD_USER.value
            )
            print("✅ Created custom 'quantum_researcher' role")
        except ValueError:
            print("ℹ️  Researcher role already exists")
    
    # Researcher using the system
    with SecurityContext("dr_quantum", "quantum_researcher"):
        print("Dr. Quantum conducting variational algorithm research...")
        
        # Create a parametric circuit
        with QuantumProgram(name="vqe_ansatz") as qp:
            qubits = allocate(2)
            H(qubits[0])  # Hadamard for superposition
            CNOT(qubits[0], qubits[1])
        
        # Check hardware access (would be available for this role)
        can_use_hardware = security.check_hardware_permission("run")
        print(f"  🖥️  Hardware access available: {can_use_hardware}")
        
        # Check plugin installation (should be denied)
        can_install_plugins = security.check_plugin_permission("install")
        print(f"  🔌 Plugin installation access: {can_install_plugins}")
    
    # Scenario 4: System Administration
    print("\n⚙️  Scenario 4: System Administration")
    
    @require_permission(Permission.MANAGE_USERS)
    def create_user_batch(usernames, role):
        """Admin function to create multiple users."""
        for username in usernames:
            security.create_user_session(username, role)
        return f"Created {len(usernames)} users with role {role}"
    
    @require_role(UserRole.ADMIN)
    def view_audit_logs():
        """Admin-only function to view audit logs."""
        summary = security.audit_logger.get_security_summary(1)
        return f"Found {summary['total_events']} events in last hour"
    
    with AdminContext():
        print("Administrator performing system management...")
        current_user = security.get_current_user_info()
        print(f"  Current admin user: {current_user['username']} ({current_user['role']})")
        
        try:
            result = create_user_batch(["researcher1", "researcher2"], UserRole.STANDARD_USER.value)
            print(f"  ✅ {result}")
            
            audit_info = view_audit_logs()
            print(f"  📊 {audit_info}")
            
        except Exception as e:
            print(f"  ❌ {e}")
    
    # Scenario 5: Guest Demonstration
    print("\n👥 Scenario 5: Public Demonstration")
    
    with SecurityContext("demo_guest", UserRole.GUEST.value):
        print("Guest viewing quantum demonstration...")
        
        # Guest can view existing circuits
        can_view = security.check_circuit_permission("view")
        can_create = security.check_circuit_permission("create")
        can_simulate = security.check_simulation_permission("run")
        
        print(f"  👀 Can view circuits: {can_view}")
        print(f"  ✏️  Can create circuits: {can_create}")
        print(f"  ▶️  Can run simulations: {can_simulate}")
    
    # Scenario 6: Security Monitoring
    print("\n🛡️  Scenario 6: Security Monitoring")
    
    # Get comprehensive security status
    security_status = security.get_security_summary()
    
    print("Current security status:")
    print(f"  👤 Active users: {security_status['user_system']['active_users']}")
    print(f"  🔒 Roles configured: {security_status['role_system']['total_roles']}")
    print(f"  📝 Recent events: {security_status['audit_summary']['total_events']}")
    print(f"  ⚠️  Permission denials: {security_status['audit_summary']['permission_denials']}")
    print(f"  🚨 Security violations: {security_status['audit_summary']['security_violations']}")
    
    # Show role capabilities
    print("\n📋 Role Capabilities Summary:")
    roles_to_check = [UserRole.GUEST.value, UserRole.READ_ONLY.value, 
                     UserRole.STANDARD_USER.value, UserRole.DEVELOPER.value, UserRole.ADMIN.value]
    
    key_permissions = [
        Permission.CREATE_CIRCUIT,
        Permission.RUN_SIMULATION,
        Permission.INSTALL_PLUGIN,
        Permission.RUN_HARDWARE,
        Permission.MANAGE_USERS
    ]
    
    for role_name in roles_to_check:
        role = security.role_manager.get_role(role_name)
        if role:
            perms = role.get_all_permissions()
            capabilities = []
            for perm in key_permissions:
                if perm in perms:
                    capabilities.append(perm.value.replace('_', ' ').title())
            
            print(f"  {role_name.title()}: {', '.join(capabilities) if capabilities else 'View only'}")
    
    print("\n✅ RBAC Examples Complete!")
    print("The quantum platform is now secured with role-based access control.")


if __name__ == "__main__":
    main() 