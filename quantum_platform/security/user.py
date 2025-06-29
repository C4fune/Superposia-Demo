"""
User Management System

This module provides user account management, authentication context,
and user session handling for the quantum computing platform.
"""

import uuid
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import logging
from threading import RLock

from quantum_platform.security.rbac import Permission, UserRole


@dataclass
class User:
    """Represents a user account in the system."""
    
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    role: str = UserRole.STANDARD_USER.value
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate user data."""
        if not self.username:
            raise ValueError("Username is required")
    
    def update_last_login(self):
        """Update the last login timestamp."""
        self.last_login = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary representation."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary representation."""
        user = cls(
            user_id=data["user_id"],
            username=data["username"],
            email=data.get("email", ""),
            role=data.get("role", UserRole.STANDARD_USER.value),
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {})
        )
        
        if data.get("created_at"):
            user.created_at = datetime.fromisoformat(data["created_at"])
        
        if data.get("last_login"):
            user.last_login = datetime.fromisoformat(data["last_login"])
        
        return user
    
    def __str__(self) -> str:
        return f"User({self.username})"
    
    def __repr__(self) -> str:
        return f"User(id='{self.user_id}', username='{self.username}', role='{self.role}')"


class UserContext:
    """
    Thread-local user context for tracking current user sessions.
    
    This provides a way to access the current user's information and
    permissions throughout the application without passing user objects
    around explicitly.
    """
    
    _context = threading.local()
    _logger = logging.getLogger(__name__)
    
    @classmethod
    def set_current_user(cls, user: User):
        """Set the current user for this thread."""
        cls._context.current_user = user
        cls._context.session_start = time.time()
        cls._logger.debug(f"Set current user: {user.username}")
    
    @classmethod
    def get_current_user(cls) -> Optional[User]:
        """Get the current user for this thread."""
        return getattr(cls._context, 'current_user', None)
    
    @classmethod
    def clear_current_user(cls):
        """Clear the current user context."""
        if hasattr(cls._context, 'current_user'):
            username = cls._context.current_user.username
            cls._logger.debug(f"Cleared current user: {username}")
            delattr(cls._context, 'current_user')
        
        if hasattr(cls._context, 'session_start'):
            delattr(cls._context, 'session_start')
    
    @classmethod
    def get_session_duration(cls) -> Optional[float]:
        """Get the duration of the current session in seconds."""
        session_start = getattr(cls._context, 'session_start', None)
        return time.time() - session_start if session_start else None
    
    @classmethod
    def has_current_user(cls) -> bool:
        """Check if there's a current user in context."""
        return hasattr(cls._context, 'current_user')
    
    @classmethod
    def get_current_user_id(cls) -> Optional[str]:
        """Get the current user's ID."""
        user = cls.get_current_user()
        return user.user_id if user else None
    
    @classmethod
    def get_current_username(cls) -> Optional[str]:
        """Get the current user's username."""
        user = cls.get_current_user()
        return user.username if user else None
    
    @classmethod
    def get_current_user_role(cls) -> Optional[str]:
        """Get the current user's role."""
        user = cls.get_current_user()
        return user.role if user else None


class UserManager:
    """Manages user accounts and authentication."""
    
    def __init__(self, role_manager=None):
        """
        Initialize user manager.
        
        Args:
            role_manager: Role manager instance for permission checking
        """
        self._users: Dict[str, User] = {}
        self._username_to_id: Dict[str, str] = {}
        self._lock = RLock()
        self._logger = logging.getLogger(__name__)
        self._role_manager = role_manager
        
        # Create default admin user for single-user scenarios
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user for initial setup."""
        admin_user = User(
            username="admin",
            email="admin@localhost",
            role=UserRole.ADMIN.value,
            metadata={"is_default": True, "created_by": "system"}
        )
        
        self._users[admin_user.user_id] = admin_user
        self._username_to_id[admin_user.username] = admin_user.user_id
        self._logger.info("Created default admin user")
    
    def create_user(self, username: str, email: str = "", 
                   role: str = UserRole.STANDARD_USER.value,
                   metadata: Optional[Dict[str, Any]] = None) -> User:
        """
        Create a new user account.
        
        Args:
            username: Unique username
            email: User email address
            role: User role name
            metadata: Additional user metadata
            
        Returns:
            Created user object
            
        Raises:
            ValueError: If username already exists or role is invalid
        """
        with self._lock:
            if username in self._username_to_id:
                raise ValueError(f"Username '{username}' already exists")
            
            # Validate role if role manager is available
            if self._role_manager and not self._role_manager.get_role(role):
                raise ValueError(f"Invalid role: {role}")
            
            user = User(
                username=username,
                email=email,
                role=role,
                metadata=metadata or {}
            )
            
            self._users[user.user_id] = user
            self._username_to_id[username] = user.user_id
            
            self._logger.info(f"Created user: {username} with role: {role}")
            return user
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by their ID."""
        return self._users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by their username."""
        user_id = self._username_to_id.get(username)
        return self._users.get(user_id) if user_id else None
    
    def update_user_role(self, user_id: str, new_role: str) -> bool:
        """
        Update a user's role.
        
        Args:
            user_id: User ID to update
            new_role: New role name
            
        Returns:
            True if updated successfully, False otherwise
        """
        with self._lock:
            user = self._users.get(user_id)
            if not user:
                return False
            
            # Validate role if role manager is available
            if self._role_manager and not self._role_manager.get_role(new_role):
                self._logger.warning(f"Invalid role for update: {new_role}")
                return False
            
            old_role = user.role
            user.role = new_role
            
            self._logger.info(f"Updated user {user.username} role: {old_role} -> {new_role}")
            return True
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account."""
        with self._lock:
            user = self._users.get(user_id)
            if user and user.is_active:
                user.is_active = False
                self._logger.info(f"Deactivated user: {user.username}")
                return True
            return False
    
    def activate_user(self, user_id: str) -> bool:
        """Activate a user account."""
        with self._lock:
            user = self._users.get(user_id)
            if user and not user.is_active:
                user.is_active = True
                self._logger.info(f"Activated user: {user.username}")
                return True
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user account.
        
        Args:
            user_id: User ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            user = self._users.get(user_id)
            if not user:
                return False
            
            # Don't delete default admin
            if user.metadata.get("is_default"):
                self._logger.warning("Cannot delete default admin user")
                return False
            
            # Remove from both mappings
            del self._users[user_id]
            if user.username in self._username_to_id:
                del self._username_to_id[user.username]
            
            self._logger.info(f"Deleted user: {user.username}")
            return True
    
    def list_users(self, include_inactive: bool = False) -> List[User]:
        """
        List all users.
        
        Args:
            include_inactive: Whether to include inactive users
            
        Returns:
            List of user objects
        """
        if include_inactive:
            return list(self._users.values())
        else:
            return [user for user in self._users.values() if user.is_active]
    
    def get_users_by_role(self, role: str) -> List[User]:
        """Get all users with a specific role."""
        return [user for user in self._users.values() 
                if user.role == role and user.is_active]
    
    def authenticate_user(self, username: str) -> Optional[User]:
        """
        Authenticate a user and set them as current user.
        
        Args:
            username: Username to authenticate
            
        Returns:
            User object if authentication successful, None otherwise
        """
        user = self.get_user_by_username(username)
        
        if user and user.is_active:
            user.update_last_login()
            UserContext.set_current_user(user)
            self._logger.info(f"User authenticated: {username}")
            return user
        
        self._logger.warning(f"Authentication failed for: {username}")
        return None
    
    def logout_current_user(self):
        """Logout the current user."""
        current_user = UserContext.get_current_user()
        if current_user:
            self._logger.info(f"User logged out: {current_user.username}")
        UserContext.clear_current_user()
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """
        Get all permissions for a user based on their role.
        
        Args:
            user_id: User ID
            
        Returns:
            Set of permissions
        """
        user = self._users.get(user_id)
        if not user or not user.is_active:
            return set()
        
        if self._role_manager:
            return self._role_manager.get_role_permissions(user.role)
        
        return set()
    
    def check_user_permission(self, user_id: str, permission: Permission) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            user_id: User ID
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        return permission in self.get_user_permissions(user_id)
    
    def get_current_user_permissions(self) -> Set[Permission]:
        """Get permissions for the current user."""
        current_user = UserContext.get_current_user()
        if current_user:
            return self.get_user_permissions(current_user.user_id)
        return set()
    
    def check_current_user_permission(self, permission: Permission) -> bool:
        """Check if the current user has a specific permission."""
        return permission in self.get_current_user_permissions()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get user manager statistics."""
        active_users = sum(1 for user in self._users.values() if user.is_active)
        inactive_users = len(self._users) - active_users
        
        # Count users by role
        role_counts = {}
        for user in self._users.values():
            if user.is_active:
                role_counts[user.role] = role_counts.get(user.role, 0) + 1
        
        return {
            "total_users": len(self._users),
            "active_users": active_users,
            "inactive_users": inactive_users,
            "users_by_role": role_counts,
            "current_user": UserContext.get_current_username(),
            "session_duration": UserContext.get_session_duration()
        } 