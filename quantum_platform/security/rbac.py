"""
Role-Based Access Control (RBAC) System

This module defines the core RBAC functionality including roles,
permissions, and role management for the quantum computing platform.
"""

from enum import Enum
from typing import Set, Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from threading import RLock


class Permission(Enum):
    """System permissions for quantum platform operations."""
    
    # Circuit and Program Development
    CREATE_CIRCUIT = "create_circuit"
    EDIT_CIRCUIT = "edit_circuit"
    DELETE_CIRCUIT = "delete_circuit"
    VIEW_CIRCUIT = "view_circuit"
    EXPORT_CIRCUIT = "export_circuit"
    IMPORT_CIRCUIT = "import_circuit"
    
    # Simulation Operations
    RUN_SIMULATION = "run_simulation"
    VIEW_SIMULATION_RESULTS = "view_simulation_results"
    CONFIGURE_SIMULATION = "configure_simulation"
    
    # Hardware Operations (Future)
    RUN_HARDWARE = "run_hardware"
    CONFIGURE_HARDWARE = "configure_hardware"
    VIEW_HARDWARE_STATUS = "view_hardware_status"
    
    # Plugin Management
    INSTALL_PLUGIN = "install_plugin"
    REMOVE_PLUGIN = "remove_plugin"
    CONFIGURE_PLUGIN = "configure_plugin"
    VIEW_PLUGINS = "view_plugins"
    
    # System Administration
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    VIEW_SYSTEM_LOGS = "view_system_logs"
    CONFIGURE_SYSTEM = "configure_system"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    
    # Resource Management
    ALLOCATE_RESOURCES = "allocate_resources"
    VIEW_RESOURCE_USAGE = "view_resource_usage"
    SET_RESOURCE_LIMITS = "set_resource_limits"
    
    # Data Management
    BACKUP_DATA = "backup_data"
    RESTORE_DATA = "restore_data"
    EXPORT_USER_DATA = "export_user_data"
    DELETE_USER_DATA = "delete_user_data"


@dataclass
class Role:
    """Represents a user role with associated permissions."""
    
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    inherits_from: Optional['Role'] = None
    is_system_role: bool = False
    
    def __post_init__(self):
        """Initialize role with inherited permissions."""
        if self.inherits_from:
            self.permissions.update(self.inherits_from.get_all_permissions())
    
    def add_permission(self, permission: Permission):
        """Add a permission to this role."""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission):
        """Remove a permission from this role."""
        self.permissions.discard(permission)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has a specific permission."""
        return permission in self.get_all_permissions()
    
    def get_all_permissions(self) -> Set[Permission]:
        """Get all permissions including inherited ones."""
        all_perms = self.permissions.copy()
        if self.inherits_from:
            all_perms.update(self.inherits_from.get_all_permissions())
        return all_perms
    
    def __str__(self) -> str:
        return f"Role({self.name})"
    
    def __repr__(self) -> str:
        return f"Role(name='{self.name}', permissions={len(self.permissions)})"


class UserRole(Enum):
    """Predefined user roles for the platform."""
    
    ADMIN = "admin"
    DEVELOPER = "developer"
    STANDARD_USER = "standard_user"
    READ_ONLY = "read_only"
    GUEST = "guest"


class RoleManager:
    """Manages roles and their permissions."""
    
    def __init__(self):
        self._roles: Dict[str, Role] = {}
        self._lock = RLock()
        self._logger = logging.getLogger(__name__)
        
        # Initialize default roles
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize the default system roles."""
        
        # Guest Role - Very limited access
        guest_role = Role(
            name=UserRole.GUEST.value,
            description="Guest user with minimal read-only access",
            permissions={
                Permission.VIEW_CIRCUIT,
                Permission.VIEW_SIMULATION_RESULTS,
                Permission.VIEW_PLUGINS
            },
            is_system_role=True
        )
        
        # Read-Only Role - Can view most things but not modify
        readonly_role = Role(
            name=UserRole.READ_ONLY.value,
            description="Read-only user who can view but not modify",
            permissions={
                Permission.VIEW_CIRCUIT,
                Permission.VIEW_SIMULATION_RESULTS,
                Permission.VIEW_PLUGINS,
                Permission.VIEW_RESOURCE_USAGE,
                Permission.EXPORT_CIRCUIT,
                Permission.VIEW_HARDWARE_STATUS
            },
            inherits_from=guest_role,
            is_system_role=True
        )
        
        # Standard User Role - Can develop and run simulations
        standard_role = Role(
            name=UserRole.STANDARD_USER.value,
            description="Standard user who can develop and run quantum programs",
            permissions={
                Permission.CREATE_CIRCUIT,
                Permission.EDIT_CIRCUIT,
                Permission.DELETE_CIRCUIT,
                Permission.RUN_SIMULATION,
                Permission.CONFIGURE_SIMULATION,
                Permission.IMPORT_CIRCUIT,
                Permission.ALLOCATE_RESOURCES,
                Permission.EXPORT_USER_DATA
            },
            inherits_from=readonly_role,
            is_system_role=True
        )
        
        # Developer Role - Can manage plugins and advanced features
        developer_role = Role(
            name=UserRole.DEVELOPER.value,
            description="Developer who can manage plugins and advanced features",
            permissions={
                Permission.INSTALL_PLUGIN,
                Permission.REMOVE_PLUGIN,
                Permission.CONFIGURE_PLUGIN,
                Permission.RUN_HARDWARE,
                Permission.CONFIGURE_HARDWARE,
                Permission.SET_RESOURCE_LIMITS,
                Permission.BACKUP_DATA,
                Permission.RESTORE_DATA
            },
            inherits_from=standard_role,
            is_system_role=True
        )
        
        # Admin Role - Full system access
        admin_role = Role(
            name=UserRole.ADMIN.value,
            description="Administrator with full system access",
            permissions={
                Permission.MANAGE_USERS,
                Permission.MANAGE_ROLES,
                Permission.VIEW_SYSTEM_LOGS,
                Permission.CONFIGURE_SYSTEM,
                Permission.VIEW_AUDIT_LOGS,
                Permission.DELETE_USER_DATA
            },
            inherits_from=developer_role,
            is_system_role=True
        )
        
        # Register all default roles
        for role in [guest_role, readonly_role, standard_role, developer_role, admin_role]:
            self._roles[role.name] = role
        
        self._logger.info(f"Initialized {len(self._roles)} default roles")
    
    def create_role(self, name: str, description: str, 
                   permissions: Optional[Set[Permission]] = None,
                   inherits_from: Optional[str] = None) -> Role:
        """
        Create a new custom role.
        
        Args:
            name: Role name
            description: Role description
            permissions: Set of permissions for the role
            inherits_from: Name of role to inherit from
            
        Returns:
            Created role
            
        Raises:
            ValueError: If role already exists or inherits_from role not found
        """
        with self._lock:
            if name in self._roles:
                raise ValueError(f"Role '{name}' already exists")
            
            parent_role = None
            if inherits_from:
                parent_role = self._roles.get(inherits_from)
                if not parent_role:
                    raise ValueError(f"Parent role '{inherits_from}' not found")
            
            role = Role(
                name=name,
                description=description,
                permissions=permissions or set(),
                inherits_from=parent_role,
                is_system_role=False
            )
            
            self._roles[name] = role
            self._logger.info(f"Created custom role: {name}")
            return role
    
    def get_role(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        return self._roles.get(name)
    
    def delete_role(self, name: str) -> bool:
        """
        Delete a custom role.
        
        Args:
            name: Role name to delete
            
        Returns:
            True if deleted, False if not found or is system role
        """
        with self._lock:
            role = self._roles.get(name)
            if not role:
                return False
            
            if role.is_system_role:
                self._logger.warning(f"Cannot delete system role: {name}")
                return False
            
            del self._roles[name]
            self._logger.info(f"Deleted role: {name}")
            return True
    
    def list_roles(self) -> List[Role]:
        """List all available roles."""
        return list(self._roles.values())
    
    def get_role_permissions(self, role_name: str) -> Set[Permission]:
        """Get all permissions for a role."""
        role = self._roles.get(role_name)
        return role.get_all_permissions() if role else set()
    
    def check_permission(self, role_name: str, permission: Permission) -> bool:
        """Check if a role has a specific permission."""
        role = self._roles.get(role_name)
        return role.has_permission(permission) if role else False
    
    def get_role_hierarchy(self) -> Dict[str, Any]:
        """Get the role hierarchy structure."""
        hierarchy = {}
        
        for role in self._roles.values():
            hierarchy[role.name] = {
                "description": role.description,
                "permissions": [p.value for p in role.permissions],
                "inherits_from": role.inherits_from.name if role.inherits_from else None,
                "is_system_role": role.is_system_role,
                "total_permissions": len(role.get_all_permissions())
            }
        
        return hierarchy
    
    def validate_role_assignment(self, role_name: str) -> bool:
        """Validate that a role can be assigned to users."""
        role = self._roles.get(role_name)
        if not role:
            return False
        
        # Check that role has at least basic permissions
        basic_perms = {Permission.VIEW_CIRCUIT, Permission.VIEW_SIMULATION_RESULTS}
        return bool(role.get_all_permissions().intersection(basic_perms))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get role manager statistics."""
        system_roles = sum(1 for role in self._roles.values() if role.is_system_role)
        custom_roles = len(self._roles) - system_roles
        
        return {
            "total_roles": len(self._roles),
            "system_roles": system_roles,
            "custom_roles": custom_roles,
            "total_permissions": len(Permission),
            "role_names": list(self._roles.keys())
        } 