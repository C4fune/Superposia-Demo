"""
Security Enforcement System

This module provides decorators and enforcement mechanisms for checking
permissions and roles before allowing operations to proceed.
"""

import functools
import logging
from typing import Callable, Optional, Union, List, Any
from enum import Enum

from quantum_platform.security.rbac import Permission, UserRole
from quantum_platform.security.user import UserContext


class SecurityException(Exception):
    """Base exception for security-related errors."""
    pass


class PermissionDeniedException(SecurityException):
    """Raised when a user lacks required permissions."""
    pass


class AuthenticationRequiredException(SecurityException):
    """Raised when authentication is required but user is not logged in."""
    pass


class RoleRequiredException(SecurityException):
    """Raised when a specific role is required."""
    pass


def require_permission(permission: Union[Permission, List[Permission]], 
                      allow_admin_override: bool = True):
    """
    Decorator to require specific permission(s) for function execution.
    
    Args:
        permission: Required permission or list of permissions
        allow_admin_override: Whether admin role bypasses permission check
        
    Raises:
        AuthenticationRequiredException: If no user is logged in
        PermissionDeniedException: If user lacks required permissions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_user = UserContext.get_current_user()
            
            # Check authentication
            if not current_user:
                raise AuthenticationRequiredException(
                    f"Authentication required for {func.__name__}"
                )
            
            # Admin override if enabled
            if allow_admin_override and current_user.role == UserRole.ADMIN.value:
                return func(*args, **kwargs)
            
            # Check permissions
            required_perms = permission if isinstance(permission, list) else [permission]
            enforcer = SecurityEnforcer.get_instance()
            
            for perm in required_perms:
                if not enforcer.check_permission(current_user.user_id, perm):
                    raise PermissionDeniedException(
                        f"Permission '{perm.value}' required for {func.__name__}"
                    )
            
            return func(*args, **kwargs)
        
        # Add metadata for introspection
        wrapper._required_permissions = permission
        wrapper._allow_admin_override = allow_admin_override
        return wrapper
    
    return decorator


def require_role(role: Union[str, UserRole, List[Union[str, UserRole]]]):
    """
    Decorator to require specific role(s) for function execution.
    
    Args:
        role: Required role or list of roles
        
    Raises:
        AuthenticationRequiredException: If no user is logged in
        RoleRequiredException: If user doesn't have required role
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_user = UserContext.get_current_user()
            
            # Check authentication
            if not current_user:
                raise AuthenticationRequiredException(
                    f"Authentication required for {func.__name__}"
                )
            
            # Normalize roles to strings
            required_roles = role if isinstance(role, list) else [role]
            required_role_names = []
            
            for r in required_roles:
                if isinstance(r, UserRole):
                    required_role_names.append(r.value)
                else:
                    required_role_names.append(str(r))
            
            # Check role
            if current_user.role not in required_role_names:
                raise RoleRequiredException(
                    f"Role {required_role_names} required for {func.__name__}, "
                    f"but user has role '{current_user.role}'"
                )
            
            return func(*args, **kwargs)
        
        # Add metadata for introspection
        wrapper._required_roles = role
        return wrapper
    
    return decorator


def require_authentication(func: Callable) -> Callable:
    """
    Decorator to require user authentication for function execution.
    
    Raises:
        AuthenticationRequiredException: If no user is logged in
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        current_user = UserContext.get_current_user()
        
        if not current_user:
            raise AuthenticationRequiredException(
                f"Authentication required for {func.__name__}"
            )
        
        return func(*args, **kwargs)
    
    return wrapper


def allow_guest_access(func: Callable) -> Callable:
    """
    Decorator to explicitly mark functions that allow guest access.
    This is mainly for documentation and future enforcement policies.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    wrapper._allows_guest_access = True
    return wrapper


class SecurityEnforcer:
    """
    Central security enforcement system.
    
    This class provides the core security checking functionality and
    can be integrated with different user and role management systems.
    """
    
    _instance: Optional['SecurityEnforcer'] = None
    _logger = logging.getLogger(__name__)
    
    def __init__(self, user_manager=None, role_manager=None):
        """
        Initialize security enforcer.
        
        Args:
            user_manager: User manager instance
            role_manager: Role manager instance
        """
        self.user_manager = user_manager
        self.role_manager = role_manager
        self._audit_enabled = True
        
        # Set as global instance if none exists
        if SecurityEnforcer._instance is None:
            SecurityEnforcer._instance = self
    
    @classmethod
    def get_instance(cls) -> 'SecurityEnforcer':
        """Get the global security enforcer instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def set_instance(cls, instance: 'SecurityEnforcer'):
        """Set the global security enforcer instance."""
        cls._instance = instance
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            user_id: User ID to check
            permission: Permission to verify
            
        Returns:
            True if user has permission, False otherwise
        """
        if not self.user_manager:
            self._logger.warning("No user manager configured, denying permission")
            return False
        
        has_permission = self.user_manager.check_user_permission(user_id, permission)
        
        if self._audit_enabled:
            user = self.user_manager.get_user_by_id(user_id)
            username = user.username if user else "unknown"
            self._logger.debug(
                f"Permission check: {username} -> {permission.value} = {has_permission}"
            )
        
        return has_permission
    
    def check_role(self, user_id: str, role: str) -> bool:
        """
        Check if a user has a specific role.
        
        Args:
            user_id: User ID to check
            role: Role to verify
            
        Returns:
            True if user has role, False otherwise
        """
        if not self.user_manager:
            return False
        
        user = self.user_manager.get_user_by_id(user_id)
        has_role = user and user.role == role
        
        if self._audit_enabled:
            username = user.username if user else "unknown"
            self._logger.debug(
                f"Role check: {username} -> {role} = {has_role}"
            )
        
        return has_role
    
    def enforce_permission(self, permission: Permission, 
                          user_id: Optional[str] = None) -> bool:
        """
        Enforce a permission check, raising exception if denied.
        
        Args:
            permission: Permission to check
            user_id: User ID (uses current user if None)
            
        Returns:
            True if permission granted
            
        Raises:
            AuthenticationRequiredException: If no user specified or logged in
            PermissionDeniedException: If permission denied
        """
        if user_id is None:
            current_user = UserContext.get_current_user()
            if not current_user:
                raise AuthenticationRequiredException("Authentication required")
            user_id = current_user.user_id
        
        if not self.check_permission(user_id, permission):
            user = self.user_manager.get_user_by_id(user_id) if self.user_manager else None
            username = user.username if user else "unknown"
            
            raise PermissionDeniedException(
                f"User '{username}' lacks permission '{permission.value}'"
            )
        
        return True
    
    def enforce_role(self, role: str, user_id: Optional[str] = None) -> bool:
        """
        Enforce a role check, raising exception if denied.
        
        Args:
            role: Role to check
            user_id: User ID (uses current user if None)
            
        Returns:
            True if role granted
            
        Raises:
            AuthenticationRequiredException: If no user specified or logged in
            RoleRequiredException: If role requirement not met
        """
        if user_id is None:
            current_user = UserContext.get_current_user()
            if not current_user:
                raise AuthenticationRequiredException("Authentication required")
            user_id = current_user.user_id
        
        if not self.check_role(user_id, role):
            user = self.user_manager.get_user_by_id(user_id) if self.user_manager else None
            username = user.username if user else "unknown"
            current_role = user.role if user else "none"
            
            raise RoleRequiredException(
                f"User '{username}' has role '{current_role}' but requires '{role}'"
            )
        
        return True
    
    def check_current_user_permission(self, permission: Permission) -> bool:
        """Check if the current user has a specific permission."""
        current_user = UserContext.get_current_user()
        if not current_user:
            return False
        
        return self.check_permission(current_user.user_id, permission)
    
    def enforce_current_user_permission(self, permission: Permission) -> bool:
        """Enforce permission for the current user."""
        return self.enforce_permission(permission)
    
    def get_current_user_permissions(self) -> set:
        """Get all permissions for the current user."""
        current_user = UserContext.get_current_user()
        if not current_user or not self.user_manager:
            return set()
        
        return self.user_manager.get_user_permissions(current_user.user_id)
    
    def is_admin_user(self, user_id: Optional[str] = None) -> bool:
        """Check if a user (or current user) has admin role."""
        if user_id is None:
            current_user = UserContext.get_current_user()
            if not current_user:
                return False
            user_id = current_user.user_id
        
        return self.check_role(user_id, UserRole.ADMIN.value)
    
    def get_permission_summary(self, user_id: Optional[str] = None) -> dict:
        """
        Get a summary of permissions for a user.
        
        Args:
            user_id: User ID (uses current user if None)
            
        Returns:
            Dictionary with permission summary
        """
        if user_id is None:
            current_user = UserContext.get_current_user()
            if not current_user:
                return {"error": "No current user"}
            user_id = current_user.user_id
        
        if not self.user_manager:
            return {"error": "No user manager"}
        
        user = self.user_manager.get_user_by_id(user_id)
        if not user:
            return {"error": "User not found"}
        
        permissions = self.user_manager.get_user_permissions(user_id)
        
        return {
            "user_id": user_id,
            "username": user.username,
            "role": user.role,
            "is_active": user.is_active,
            "is_admin": user.role == UserRole.ADMIN.value,
            "permission_count": len(permissions),
            "permissions": [p.value for p in permissions],
            "can_manage_users": Permission.MANAGE_USERS in permissions,
            "can_run_hardware": Permission.RUN_HARDWARE in permissions,
            "can_install_plugins": Permission.INSTALL_PLUGIN in permissions
        }
    
    def set_audit_enabled(self, enabled: bool):
        """Enable or disable audit logging."""
        self._audit_enabled = enabled
        self._logger.info(f"Audit logging {'enabled' if enabled else 'disabled'}")
    
    def validate_function_access(self, func: Callable, user_id: Optional[str] = None) -> dict:
        """
        Validate if a user can access a decorated function.
        
        Args:
            func: Function to check access for
            user_id: User ID (uses current user if None)
            
        Returns:
            Dictionary with access validation results
        """
        if user_id is None:
            current_user = UserContext.get_current_user()
            if not current_user:
                return {"access_granted": False, "reason": "No current user"}
            user_id = current_user.user_id
        
        result = {
            "function_name": func.__name__,
            "access_granted": True,
            "required_permissions": [],
            "required_roles": [],
            "missing_permissions": [],
            "user_role": None,
            "reasons": []
        }
        
        if self.user_manager:
            user = self.user_manager.get_user_by_id(user_id)
            result["user_role"] = user.role if user else None
        
        # Check for permission requirements
        if hasattr(func, '_required_permissions'):
            perms = func._required_permissions
            if not isinstance(perms, list):
                perms = [perms]
            
            result["required_permissions"] = [p.value for p in perms]
            
            for perm in perms:
                if not self.check_permission(user_id, perm):
                    result["missing_permissions"].append(perm.value)
                    result["access_granted"] = False
                    result["reasons"].append(f"Missing permission: {perm.value}")
        
        # Check for role requirements
        if hasattr(func, '_required_roles'):
            roles = func._required_roles
            if not isinstance(roles, list):
                roles = [roles]
            
            role_names = []
            for role in roles:
                if isinstance(role, UserRole):
                    role_names.append(role.value)
                else:
                    role_names.append(str(role))
            
            result["required_roles"] = role_names
            
            user_role = result["user_role"]
            if user_role not in role_names:
                result["access_granted"] = False
                result["reasons"].append(f"Required role: {role_names}, but user has: {user_role}")
        
        return result 