"""
RBAC Integration Module

This module provides integration points to connect the RBAC security system
with existing quantum platform components, adding security enforcement to
simulation, plugin management, circuit operations, and more.
"""

import logging
from typing import Dict, Any, Optional, List
from functools import wraps

from quantum_platform.security.rbac import Permission, UserRole, RoleManager
from quantum_platform.security.user import UserManager, UserContext, User
from quantum_platform.security.enforcement import (
    SecurityEnforcer, require_permission, require_role, require_authentication
)
from quantum_platform.security.audit import SecurityAuditLogger, AuditEventType


class QuantumPlatformSecurity:
    """
    Main security integration class for the quantum platform.
    
    This class coordinates all security components and provides a unified
    interface for integrating RBAC with quantum platform operations.
    """
    
    def __init__(self):
        """Initialize the quantum platform security system."""
        self._logger = logging.getLogger(__name__)
        
        # Initialize core security components
        self.role_manager = RoleManager()
        self.user_manager = UserManager(role_manager=self.role_manager)
        self.security_enforcer = SecurityEnforcer(
            user_manager=self.user_manager,
            role_manager=self.role_manager
        )
        self.audit_logger = SecurityAuditLogger()
        
        # Set global instances
        SecurityEnforcer.set_instance(self.security_enforcer)
        
        # Initialize default admin session for single-user mode
        self._initialize_single_user_mode()
        
        self._logger.info("Quantum Platform Security System initialized")
    
    def _initialize_single_user_mode(self):
        """Initialize single-user mode with admin privileges."""
        admin_user = self.user_manager.get_user_by_username("admin")
        if admin_user:
            self.user_manager.authenticate_user("admin")
            self._logger.info("Single-user mode: Admin user authenticated")
    
    def get_current_user_info(self) -> Dict[str, Any]:
        """Get information about the current user."""
        current_user = UserContext.get_current_user()
        if not current_user:
            return {"authenticated": False}
        
        permissions = self.user_manager.get_user_permissions(current_user.user_id)
        
        return {
            "authenticated": True,
            "user_id": current_user.user_id,
            "username": current_user.username,
            "role": current_user.role,
            "permissions": [p.value for p in permissions],
            "session_duration": UserContext.get_session_duration(),
            "is_admin": current_user.role == UserRole.ADMIN.value
        }
    
    def create_user_session(self, username: str, role: str = UserRole.STANDARD_USER.value) -> bool:
        """
        Create a new user session.
        
        Args:
            username: Username for the session
            role: User role
            
        Returns:
            True if session created successfully
        """
        try:
            # Create user if doesn't exist
            existing_user = self.user_manager.get_user_by_username(username)
            if not existing_user:
                user = self.user_manager.create_user(username=username, role=role)
                self.audit_logger.log_user_management("create", username, True)
            else:
                user = existing_user
            
            # Authenticate user
            authenticated_user = self.user_manager.authenticate_user(username)
            if authenticated_user:
                self.audit_logger.log_authentication(username, True)
                return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"Failed to create user session: {e}")
            self.audit_logger.log_authentication(username, False, {"error": str(e)})
            return False
    
    def check_circuit_permission(self, operation: str) -> bool:
        """Check if current user can perform circuit operations."""
        permission_map = {
            "create": Permission.CREATE_CIRCUIT,
            "edit": Permission.EDIT_CIRCUIT,
            "delete": Permission.DELETE_CIRCUIT,
            "view": Permission.VIEW_CIRCUIT,
            "export": Permission.EXPORT_CIRCUIT,
            "import": Permission.IMPORT_CIRCUIT
        }
        
        permission = permission_map.get(operation)
        if not permission:
            return False
        
        has_permission = self.security_enforcer.check_current_user_permission(permission)
        
        # Log the permission check
        self.audit_logger.log_permission_check(
            permission.value,
            has_permission,
            resource="circuit",
            details={"operation": operation}
        )
        
        return has_permission
    
    def check_simulation_permission(self, operation: str) -> bool:
        """Check if current user can perform simulation operations."""
        permission_map = {
            "run": Permission.RUN_SIMULATION,
            "view_results": Permission.VIEW_SIMULATION_RESULTS,
            "configure": Permission.CONFIGURE_SIMULATION
        }
        
        permission = permission_map.get(operation)
        if not permission:
            return False
        
        has_permission = self.security_enforcer.check_current_user_permission(permission)
        
        # Log the permission check
        self.audit_logger.log_permission_check(
            permission.value,
            has_permission,
            resource="simulation",
            details={"operation": operation}
        )
        
        return has_permission
    
    def check_plugin_permission(self, operation: str) -> bool:
        """Check if current user can perform plugin operations."""
        permission_map = {
            "install": Permission.INSTALL_PLUGIN,
            "remove": Permission.REMOVE_PLUGIN,
            "configure": Permission.CONFIGURE_PLUGIN,
            "view": Permission.VIEW_PLUGINS
        }
        
        permission = permission_map.get(operation)
        if not permission:
            return False
        
        has_permission = self.security_enforcer.check_current_user_permission(permission)
        
        # Log the permission check
        self.audit_logger.log_permission_check(
            permission.value,
            has_permission,
            resource="plugin",
            details={"operation": operation}
        )
        
        return has_permission
    
    def check_hardware_permission(self, operation: str) -> bool:
        """Check if current user can perform hardware operations."""
        permission_map = {
            "run": Permission.RUN_HARDWARE,
            "configure": Permission.CONFIGURE_HARDWARE,
            "view_status": Permission.VIEW_HARDWARE_STATUS
        }
        
        permission = permission_map.get(operation)
        if not permission:
            return False
        
        has_permission = self.security_enforcer.check_current_user_permission(permission)
        
        # Log the permission check
        self.audit_logger.log_permission_check(
            permission.value,
            has_permission,
            resource="hardware",
            details={"operation": operation}
        )
        
        return has_permission
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security system summary."""
        current_user_info = self.get_current_user_info()
        role_stats = self.role_manager.get_stats()
        user_stats = self.user_manager.get_stats()
        audit_summary = self.audit_logger.get_security_summary(24)  # Last 24 hours
        
        return {
            "current_user": current_user_info,
            "role_system": role_stats,
            "user_system": user_stats,
            "audit_summary": audit_summary,
            "security_enabled": True,
            "single_user_mode": len(self.user_manager.list_users()) == 1
        }


# Global security instance
_quantum_security: Optional[QuantumPlatformSecurity] = None


def get_quantum_security() -> QuantumPlatformSecurity:
    """Get the global quantum platform security instance."""
    global _quantum_security
    if _quantum_security is None:
        _quantum_security = QuantumPlatformSecurity()
    return _quantum_security


def initialize_security() -> QuantumPlatformSecurity:
    """Initialize the quantum platform security system."""
    global _quantum_security
    _quantum_security = QuantumPlatformSecurity()
    return _quantum_security


# Decorators for quantum platform operations

def require_circuit_permission(operation: str):
    """Decorator to require circuit operation permissions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            security = get_quantum_security()
            if not security.check_circuit_permission(operation):
                from quantum_platform.security.enforcement import PermissionDeniedException
                raise PermissionDeniedException(f"Circuit {operation} permission required")
            
            # Log the operation
            security.audit_logger.log_system_operation(
                f"circuit_{operation}",
                f"circuit:{func.__name__}",
                True
            )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_simulation_permission(operation: str):
    """Decorator to require simulation operation permissions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            security = get_quantum_security()
            if not security.check_simulation_permission(operation):
                from quantum_platform.security.enforcement import PermissionDeniedException
                raise PermissionDeniedException(f"Simulation {operation} permission required")
            
            # Log the operation
            security.audit_logger.log_system_operation(
                f"simulation_{operation}",
                f"simulation:{func.__name__}",
                True
            )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_plugin_permission(operation: str):
    """Decorator to require plugin operation permissions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            security = get_quantum_security()
            if not security.check_plugin_permission(operation):
                from quantum_platform.security.enforcement import PermissionDeniedException
                raise PermissionDeniedException(f"Plugin {operation} permission required")
            
            # Log the operation
            security.audit_logger.log_system_operation(
                f"plugin_{operation}",
                f"plugin:{func.__name__}",
                True
            )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_hardware_permission(operation: str):
    """Decorator to require hardware operation permissions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            security = get_quantum_security()
            if not security.check_hardware_permission(operation):
                from quantum_platform.security.enforcement import PermissionDeniedException
                raise PermissionDeniedException(f"Hardware {operation} permission required")
            
            # Log the operation
            security.audit_logger.log_system_operation(
                f"hardware_{operation}",
                f"hardware:{func.__name__}",
                True
            )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Context managers for security operations

class SecurityContext:
    """Context manager for security operations."""
    
    def __init__(self, username: str, role: str = UserRole.STANDARD_USER.value):
        """
        Initialize security context.
        
        Args:
            username: Username for the context
            role: User role
        """
        self.username = username
        self.role = role
        self.security = get_quantum_security()
        self.previous_user = None
    
    def __enter__(self):
        """Enter security context."""
        self.previous_user = UserContext.get_current_user()
        self.security.create_user_session(self.username, self.role)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit security context."""
        if self.previous_user:
            UserContext.set_current_user(self.previous_user)
        else:
            UserContext.clear_current_user()


class AdminContext:
    """Context manager for administrative operations."""
    
    def __init__(self):
        """Initialize admin context."""
        self.security = get_quantum_security()
        self.previous_user = None
    
    def __enter__(self):
        """Enter admin context."""
        self.previous_user = UserContext.get_current_user()
        admin_user = self.security.user_manager.get_user_by_username("admin")
        if admin_user:
            UserContext.set_current_user(admin_user)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit admin context."""
        if self.previous_user:
            UserContext.set_current_user(self.previous_user)
        else:
            UserContext.clear_current_user() 