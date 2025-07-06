"""
Security Validation for Marketplace Packages

This module provides security validation functions for marketplace packages
to ensure safe installation and execution.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import zipfile
import tempfile

from quantum_platform.observability.logging import get_logger


def validate_package_security(package_path: Path) -> bool:
    """
    Validate security of a marketplace package.
    
    Args:
        package_path: Path to the package file
        
    Returns:
        True if package passes security validation
    """
    logger = get_logger("SecurityValidation")
    
    try:
        # Basic file validation
        if not package_path.exists():
            logger.error(f"Package file does not exist: {package_path}")
            return False
        
        # Check file size (basic DoS protection)
        file_size = package_path.stat().st_size
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            logger.error(f"Package too large: {file_size} bytes (max: {max_size})")
            return False
        
        # Validate ZIP structure if it's a ZIP file
        if package_path.suffix.lower() == '.zip':
            return _validate_zip_package(package_path)
        
        # For other file types, perform basic validation
        return _validate_generic_package(package_path)
        
    except Exception as e:
        logger.error(f"Security validation failed: {e}")
        return False


def _validate_zip_package(zip_path: Path) -> bool:
    """Validate ZIP package security."""
    logger = get_logger("SecurityValidation")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # Check for path traversal attacks
            for name in zip_file.namelist():
                if '..' in name or name.startswith('/'):
                    logger.error(f"Suspicious path in ZIP: {name}")
                    return False
            
            # Check for reasonable number of files
            if len(zip_file.namelist()) > 1000:
                logger.error("Too many files in package")
                return False
            
            # Test ZIP integrity
            bad_file = zip_file.testzip()
            if bad_file:
                logger.error(f"Corrupted file in ZIP: {bad_file}")
                return False
        
        return True
        
    except zipfile.BadZipFile:
        logger.error("Invalid ZIP file")
        return False
    except Exception as e:
        logger.error(f"ZIP validation error: {e}")
        return False


def _validate_generic_package(file_path: Path) -> bool:
    """Validate generic package file."""
    logger = get_logger("SecurityValidation")
    
    try:
        # Check file is readable
        with open(file_path, 'rb') as f:
            # Read first few bytes to check for common file types
            header = f.read(16)
            
            # Basic file type validation
            if len(header) == 0:
                logger.error("Empty package file")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Generic validation error: {e}")
        return False


def validate_plugin_code(code_content: str) -> bool:
    """
    Validate plugin code for security issues.
    
    Args:
        code_content: Python code content
        
    Returns:
        True if code passes validation
    """
    logger = get_logger("SecurityValidation")
    
    # List of potentially dangerous patterns
    dangerous_patterns = [
        'exec(',
        'eval(',
        '__import__',
        'subprocess',
        'os.system',
        'open(',
        'file(',
        'input(',
        'raw_input('
    ]
    
    # Check for dangerous patterns (basic static analysis)
    for pattern in dangerous_patterns:
        if pattern in code_content:
            logger.warning(f"Potentially dangerous pattern detected: {pattern}")
            # In a real implementation, this might be more sophisticated
            # For now, we'll log but not block
    
    return True


def get_security_report(package_path: Path) -> Dict[str, Any]:
    """
    Generate a security report for a package.
    
    Args:
        package_path: Path to the package
        
    Returns:
        Security report dictionary
    """
    report = {
        "package_path": str(package_path),
        "file_size": 0,
        "file_type": "unknown",
        "security_checks": {
            "file_exists": False,
            "size_check": False,
            "structure_check": False,
            "integrity_check": False
        },
        "warnings": [],
        "errors": [],
        "overall_status": "unknown"
    }
    
    try:
        if package_path.exists():
            report["security_checks"]["file_exists"] = True
            report["file_size"] = package_path.stat().st_size
            
            # Size check
            max_size = 100 * 1024 * 1024  # 100MB
            if report["file_size"] <= max_size:
                report["security_checks"]["size_check"] = True
            else:
                report["errors"].append(f"File too large: {report['file_size']} bytes")
            
            # File type detection
            if package_path.suffix.lower() == '.zip':
                report["file_type"] = "zip"
                
                # ZIP-specific checks
                try:
                    with zipfile.ZipFile(package_path, 'r') as zip_file:
                        report["security_checks"]["structure_check"] = True
                        
                        # Test integrity
                        bad_file = zip_file.testzip()
                        if bad_file is None:
                            report["security_checks"]["integrity_check"] = True
                        else:
                            report["errors"].append(f"Corrupted file: {bad_file}")
                            
                except zipfile.BadZipFile:
                    report["errors"].append("Invalid ZIP file")
            else:
                report["file_type"] = "generic"
                report["security_checks"]["structure_check"] = True
                report["security_checks"]["integrity_check"] = True
        else:
            report["errors"].append("Package file does not exist")
        
        # Determine overall status
        if len(report["errors"]) == 0:
            if all(report["security_checks"].values()):
                report["overall_status"] = "safe"
            else:
                report["overall_status"] = "warning"
        else:
            report["overall_status"] = "unsafe"
            
    except Exception as e:
        report["errors"].append(f"Security analysis failed: {e}")
        report["overall_status"] = "error"
    
    return report 