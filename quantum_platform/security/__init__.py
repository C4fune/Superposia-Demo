"""
Security Module

This module provides role-based access control (RBAC), user management,
and security enforcement for the quantum computing platform.
"""

from quantum_platform.security.rbac import (
    Role, Permission, UserRole, RoleManager
)
from quantum_platform.security.user import (
    User, UserManager, UserContext
)
from quantum_platform.security.enforcement import (
    require_permission, require_role, SecurityEnforcer
)
from quantum_platform.security.audit import (
    AuditEvent, AuditLogger, SecurityAuditLogger
)
from quantum_platform.security.integration import (
    QuantumPlatformSecurity, get_quantum_security, initialize_security,
    SecurityContext, AdminContext
)

__all__ = [
    'Role', 'Permission', 'UserRole', 'RoleManager',
    'User', 'UserManager', 'UserContext',
    'require_permission', 'require_role', 'SecurityEnforcer',
    'AuditEvent', 'AuditLogger', 'SecurityAuditLogger',
    'QuantumPlatformSecurity', 'get_quantum_security', 'initialize_security',
    'SecurityContext', 'AdminContext'
] 