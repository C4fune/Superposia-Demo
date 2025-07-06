"""
Marketplace Manager

This module provides the core marketplace functionality for the quantum platform,
including package discovery, installation, management, and distribution.

The marketplace enables users to:
- Browse available packages and extensions
- Install packages from local and remote repositories
- Manage installed packages and dependencies
- Publish and share their own packages
- Access community ratings and reviews
"""

import os
import json
import shutil
import hashlib
import requests
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import zipfile
import tempfile

from quantum_platform.observability.logging import get_logger
from quantum_platform.security.validation import validate_package_security


class PackageType(Enum):
    """Types of packages available in the marketplace."""
    ALGORITHM = "algorithm"
    PLUGIN = "plugin"
    TOOL = "tool"
    EDUCATIONAL = "educational"
    VISUALIZATION = "visualization"
    BACKEND = "backend"
    COMPILER = "compiler"
    UTILITY = "utility"
    EXAMPLE = "example"
    LIBRARY = "library"


class PackageStatus(Enum):
    """Status of a package in the marketplace."""
    AVAILABLE = "available"
    INSTALLED = "installed"
    OUTDATED = "outdated"
    DEPRECATED = "deprecated"
    BETA = "beta"
    EXPERIMENTAL = "experimental"


@dataclass
class PackageMetadata:
    """Metadata for a marketplace package."""
    name: str
    version: str
    description: str
    author: str
    package_type: PackageType
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    min_platform_version: str = "1.0.0"
    max_platform_version: Optional[str] = None
    license: str = "MIT"
    repository_url: Optional[str] = None
    documentation_url: Optional[str] = None
    homepage_url: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    download_count: int = 0
    rating: float = 0.0
    rating_count: int = 0
    file_size: int = 0
    checksum: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "package_type": self.package_type.value,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "min_platform_version": self.min_platform_version,
            "max_platform_version": self.max_platform_version,
            "license": self.license,
            "repository_url": self.repository_url,
            "documentation_url": self.documentation_url,
            "homepage_url": self.homepage_url,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "download_count": self.download_count,
            "rating": self.rating,
            "rating_count": self.rating_count,
            "file_size": self.file_size,
            "checksum": self.checksum
        }


@dataclass
class Package:
    """Represents a package in the marketplace."""
    metadata: PackageMetadata
    status: PackageStatus
    local_path: Optional[Path] = None
    remote_url: Optional[str] = None
    
    @property
    def is_installed(self) -> bool:
        """Check if package is installed."""
        return self.status == PackageStatus.INSTALLED and self.local_path is not None
    
    @property
    def needs_update(self) -> bool:
        """Check if package needs updating."""
        return self.status == PackageStatus.OUTDATED


@dataclass
class MarketplaceConfig:
    """Configuration for the marketplace."""
    local_repository_path: str = "packages"
    remote_repositories: List[str] = field(default_factory=lambda: [
        "https://quantum-platform.org/packages",
        "https://github.com/quantum-platform/packages"
    ])
    cache_duration: int = 3600  # 1 hour
    auto_update_check: bool = True
    allow_beta_packages: bool = False
    security_validation: bool = True
    max_package_size: int = 100 * 1024 * 1024  # 100MB
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "local_repository_path": self.local_repository_path,
            "remote_repositories": self.remote_repositories,
            "cache_duration": self.cache_duration,
            "auto_update_check": self.auto_update_check,
            "allow_beta_packages": self.allow_beta_packages,
            "security_validation": self.security_validation,
            "max_package_size": self.max_package_size
        }


class MarketplaceManager:
    """
    Main marketplace manager for package discovery and management.
    
    This class provides the primary interface for all marketplace operations,
    including package discovery, installation, updates, and management.
    """
    
    def __init__(self, config: Optional[MarketplaceConfig] = None):
        """
        Initialize the marketplace manager.
        
        Args:
            config: Marketplace configuration
        """
        self.config = config or MarketplaceConfig()
        self.logger = get_logger("MarketplaceManager")
        
        # Initialize local repository
        self.local_repo_path = Path(self.config.local_repository_path)
        self.local_repo_path.mkdir(parents=True, exist_ok=True)
        
        # Package registry
        self._installed_packages: Dict[str, Package] = {}
        self._available_packages: Dict[str, Package] = {}
        self._package_cache: Dict[str, Any] = {}
        
        # Initialize package registry
        self._load_installed_packages()
        self._initialize_built_in_packages()
        
        self.logger.info("Marketplace manager initialized")
    
    def _load_installed_packages(self):
        """Load installed packages from local repository."""
        installed_file = self.local_repo_path / "installed.json"
        if installed_file.exists():
            try:
                with open(installed_file, 'r') as f:
                    installed_data = json.load(f)
                
                for package_data in installed_data.get("packages", []):
                    metadata = PackageMetadata(**package_data["metadata"])
                    package = Package(
                        metadata=metadata,
                        status=PackageStatus.INSTALLED,
                        local_path=Path(package_data["local_path"])
                    )
                    self._installed_packages[package.metadata.name] = package
                
                self.logger.info(f"Loaded {len(self._installed_packages)} installed packages")
            except Exception as e:
                self.logger.error(f"Failed to load installed packages: {e}")
    
    def _save_installed_packages(self):
        """Save installed packages to local registry."""
        installed_file = self.local_repo_path / "installed.json"
        
        try:
            installed_data = {
                "packages": [
                    {
                        "metadata": package.metadata.to_dict(),
                        "local_path": str(package.local_path)
                    }
                    for package in self._installed_packages.values()
                ]
            }
            
            with open(installed_file, 'w') as f:
                json.dump(installed_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save installed packages: {e}")
    
    def _initialize_built_in_packages(self):
        """Initialize built-in packages that come with the platform."""
        built_in_packages = [
            PackageMetadata(
                name="quantum-algorithm-library",
                version="1.0.0",
                description="Built-in quantum algorithm library with common algorithms",
                author="Quantum Platform Team",
                package_type=PackageType.ALGORITHM,
                tags=["built-in", "algorithms", "grover", "shor", "qft"],
                dependencies=[],
                license="MIT"
            ),
            PackageMetadata(
                name="basic-visualization",
                version="1.0.0",
                description="Basic quantum circuit and state visualization tools",
                author="Quantum Platform Team",
                package_type=PackageType.VISUALIZATION,
                tags=["built-in", "visualization", "circuits", "states"],
                dependencies=[],
                license="MIT"
            ),
            PackageMetadata(
                name="educational-examples",
                version="1.0.0",
                description="Educational examples and tutorials",
                author="Quantum Platform Team",
                package_type=PackageType.EDUCATIONAL,
                tags=["built-in", "education", "examples", "tutorials"],
                dependencies=[],
                license="MIT"
            )
        ]
        
        for metadata in built_in_packages:
            package = Package(
                metadata=metadata,
                status=PackageStatus.INSTALLED,
                local_path=Path("built-in")
            )
            self._installed_packages[package.metadata.name] = package
    
    def list_available_packages(self, package_type: Optional[PackageType] = None,
                               include_beta: Optional[bool] = None) -> List[Package]:
        """
        List available packages from all repositories.
        
        Args:
            package_type: Filter by package type
            include_beta: Include beta packages
            
        Returns:
            List of available packages
        """
        # Update package cache if needed
        self._update_package_cache()
        
        packages = list(self._available_packages.values())
        
        if package_type:
            packages = [p for p in packages if p.metadata.package_type == package_type]
        
        if include_beta is not None:
            if include_beta:
                packages = [p for p in packages if p.status in [PackageStatus.AVAILABLE, PackageStatus.BETA]]
            else:
                packages = [p for p in packages if p.status == PackageStatus.AVAILABLE]
        
        return packages
    
    def search_packages(self, query: str, package_type: Optional[PackageType] = None) -> List[Package]:
        """
        Search for packages by name, description, or tags.
        
        Args:
            query: Search query
            package_type: Filter by package type
            
        Returns:
            List of matching packages
        """
        packages = self.list_available_packages(package_type)
        query_lower = query.lower()
        
        results = []
        for package in packages:
            metadata = package.metadata
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                results.append(package)
        
        return results
    
    def get_package_info(self, package_name: str) -> Optional[Package]:
        """
        Get detailed information about a package.
        
        Args:
            package_name: Name of the package
            
        Returns:
            Package information or None if not found
        """
        # Check installed packages first
        if package_name in self._installed_packages:
            return self._installed_packages[package_name]
        
        # Check available packages
        return self._available_packages.get(package_name)
    
    def install_package(self, package_name: str, version: Optional[str] = None,
                       force: bool = False) -> bool:
        """
        Install a package from the marketplace.
        
        Args:
            package_name: Name of the package to install
            version: Specific version to install (latest if None)
            force: Force reinstallation if already installed
            
        Returns:
            True if installation succeeded
        """
        try:
            # Check if already installed
            if package_name in self._installed_packages and not force:
                self.logger.info(f"Package {package_name} is already installed")
                return True
            
            # Find package in available packages
            package = self._available_packages.get(package_name)
            if not package:
                self.logger.error(f"Package {package_name} not found in marketplace")
                return False
            
            # Check dependencies
            if not self._check_dependencies(package.metadata.dependencies):
                self.logger.error(f"Cannot install {package_name}: missing dependencies")
                return False
            
            # Download and install package
            if package.remote_url:
                success = self._download_and_install_package(package)
            else:
                success = self._install_local_package(package)
            
            if success:
                package.status = PackageStatus.INSTALLED
                self._installed_packages[package_name] = package
                self._save_installed_packages()
                self.logger.info(f"Successfully installed package {package_name}")
                return True
            else:
                self.logger.error(f"Failed to install package {package_name}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error installing package {package_name}: {e}")
            return False
    
    def uninstall_package(self, package_name: str) -> bool:
        """
        Uninstall a package.
        
        Args:
            package_name: Name of the package to uninstall
            
        Returns:
            True if uninstallation succeeded
        """
        try:
            if package_name not in self._installed_packages:
                self.logger.warning(f"Package {package_name} is not installed")
                return False
            
            package = self._installed_packages[package_name]
            
            # Check if other packages depend on this one
            dependents = self._find_dependents(package_name)
            if dependents:
                self.logger.error(f"Cannot uninstall {package_name}: required by {dependents}")
                return False
            
            # Remove package files
            if package.local_path and package.local_path.exists():
                shutil.rmtree(package.local_path)
            
            # Remove from registry
            del self._installed_packages[package_name]
            self._save_installed_packages()
            
            self.logger.info(f"Successfully uninstalled package {package_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error uninstalling package {package_name}: {e}")
            return False
    
    def update_package(self, package_name: str) -> bool:
        """
        Update a package to the latest version.
        
        Args:
            package_name: Name of the package to update
            
        Returns:
            True if update succeeded
        """
        try:
            if package_name not in self._installed_packages:
                self.logger.warning(f"Package {package_name} is not installed")
                return False
            
            # Check for updates
            available_package = self._available_packages.get(package_name)
            if not available_package:
                self.logger.warning(f"Package {package_name} not found in marketplace")
                return False
            
            installed_package = self._installed_packages[package_name]
            if installed_package.metadata.version >= available_package.metadata.version:
                self.logger.info(f"Package {package_name} is already up to date")
                return True
            
            # Perform update (uninstall then install)
            self.uninstall_package(package_name)
            return self.install_package(package_name, force=True)
            
        except Exception as e:
            self.logger.error(f"Error updating package {package_name}: {e}")
            return False
    
    def check_for_updates(self) -> List[str]:
        """
        Check for package updates.
        
        Returns:
            List of packages that can be updated
        """
        self._update_package_cache()
        
        outdated_packages = []
        for name, installed_package in self._installed_packages.items():
            available_package = self._available_packages.get(name)
            if available_package and installed_package.metadata.version < available_package.metadata.version:
                outdated_packages.append(name)
        
        return outdated_packages
    
    def list_installed_packages(self) -> List[Package]:
        """
        List all installed packages.
        
        Returns:
            List of installed packages
        """
        return list(self._installed_packages.values())
    
    def _update_package_cache(self):
        """Update the package cache from remote repositories."""
        try:
            self._available_packages.clear()
            
            # Add built-in packages
            for package in self._installed_packages.values():
                if package.local_path == Path("built-in"):
                    self._available_packages[package.metadata.name] = package
            
            # Fetch from remote repositories
            for repo_url in self.config.remote_repositories:
                try:
                    packages = self._fetch_packages_from_repository(repo_url)
                    for package in packages:
                        self._available_packages[package.metadata.name] = package
                except Exception as e:
                    self.logger.warning(f"Failed to fetch from repository {repo_url}: {e}")
            
            self.logger.info(f"Updated package cache with {len(self._available_packages)} packages")
            
        except Exception as e:
            self.logger.error(f"Failed to update package cache: {e}")
    
    def _fetch_packages_from_repository(self, repo_url: str) -> List[Package]:
        """Fetch packages from a remote repository."""
        # This is a simplified implementation
        # In a real system, this would use proper API calls
        packages = []
        
        # Mock data for demonstration
        if "quantum-platform.org" in repo_url:
            mock_packages = [
                {
                    "name": "advanced-algorithms",
                    "version": "2.1.0",
                    "description": "Advanced quantum algorithms and protocols",
                    "author": "Quantum Research Group",
                    "package_type": "algorithm",
                    "tags": ["advanced", "research", "algorithms"],
                    "dependencies": [],
                    "license": "Apache-2.0"
                },
                {
                    "name": "quantum-ml",
                    "version": "1.5.0",
                    "description": "Quantum machine learning algorithms",
                    "author": "ML Quantum Team",
                    "package_type": "algorithm",
                    "tags": ["machine-learning", "vqe", "qaoa"],
                    "dependencies": ["advanced-algorithms"],
                    "license": "MIT"
                },
                {
                    "name": "circuit-optimizer",
                    "version": "3.0.0",
                    "description": "Advanced circuit optimization tools",
                    "author": "Optimization Labs",
                    "package_type": "tool",
                    "tags": ["optimization", "compiler", "efficiency"],
                    "dependencies": [],
                    "license": "MIT"
                }
            ]
            
            for package_data in mock_packages:
                metadata = PackageMetadata(
                    name=package_data["name"],
                    version=package_data["version"],
                    description=package_data["description"],
                    author=package_data["author"],
                    package_type=PackageType(package_data["package_type"]),
                    tags=package_data["tags"],
                    dependencies=package_data["dependencies"],
                    license=package_data["license"]
                )
                
                package = Package(
                    metadata=metadata,
                    status=PackageStatus.AVAILABLE,
                    remote_url=f"{repo_url}/packages/{package_data['name']}"
                )
                packages.append(package)
        
        return packages
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if all dependencies are satisfied."""
        for dep in dependencies:
            if dep not in self._installed_packages:
                return False
        return True
    
    def _find_dependents(self, package_name: str) -> List[str]:
        """Find packages that depend on the given package."""
        dependents = []
        for name, package in self._installed_packages.items():
            if package_name in package.metadata.dependencies:
                dependents.append(name)
        return dependents
    
    def _download_and_install_package(self, package: Package) -> bool:
        """Download and install a package from remote URL."""
        try:
            # Create temporary directory for download
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download package
                response = requests.get(package.remote_url, stream=True)
                response.raise_for_status()
                
                # Save to temporary file
                package_file = temp_path / f"{package.metadata.name}.zip"
                with open(package_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify checksum if available
                if package.metadata.checksum:
                    if not self._verify_checksum(package_file, package.metadata.checksum):
                        self.logger.error(f"Checksum verification failed for {package.metadata.name}")
                        return False
                
                # Security validation
                if self.config.security_validation:
                    if not validate_package_security(package_file):
                        self.logger.error(f"Security validation failed for {package.metadata.name}")
                        return False
                
                # Extract and install
                install_path = self.local_repo_path / package.metadata.name
                install_path.mkdir(parents=True, exist_ok=True)
                
                with zipfile.ZipFile(package_file, 'r') as zip_ref:
                    zip_ref.extractall(install_path)
                
                package.local_path = install_path
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to download and install package: {e}")
            return False
    
    def _install_local_package(self, package: Package) -> bool:
        """Install a local package."""
        # For built-in packages, no installation needed
        if package.local_path == Path("built-in"):
            return True
        
        # For other local packages, copy to local repository
        if package.local_path and package.local_path.exists():
            install_path = self.local_repo_path / package.metadata.name
            if install_path.exists():
                shutil.rmtree(install_path)
            
            shutil.copytree(package.local_path, install_path)
            package.local_path = install_path
            return True
        
        return False
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum."""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash == expected_checksum
        except Exception:
            return False 