"""
Multi-Provider Support Module

This module provides unified access to multiple quantum hardware providers.
"""

# Import only what exists
try:
    from .provider_manager import (
        ProviderManager,
        ProviderInfo,
        ProviderStatus,
        get_provider_manager,
        switch_provider,
        get_active_backend,
        list_available_providers,
        list_available_devices
    )
    __all__ = [
        'ProviderManager',
        'ProviderInfo',
        'ProviderStatus',
        'get_provider_manager',
        'switch_provider',
        'get_active_backend',
        'list_available_providers',
        'list_available_devices'
    ]
except ImportError:
    __all__ = []
