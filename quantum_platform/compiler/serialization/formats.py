"""
Serialization Format Definitions

This module defines the supported serialization formats and their capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

class SerializationFormat(Enum):
    """Supported serialization formats for quantum circuits."""
    QASM = "qasm"
    QASM2 = "qasm2"
    QASM3 = "qasm3" 
    JSON = "json"
    QOBJ = "qobj"
    CIRQ = "cirq"
    QISKIT = "qiskit"

@dataclass
class ExportOptions:
    """Options for exporting quantum circuits to various formats."""
    format: SerializationFormat = SerializationFormat.QASM3
    include_comments: bool = True
    flatten_custom_gates: bool = False
    use_includes: bool = True
    precision: int = 15
    indent: str = "    "
    gate_decomposition_level: int = 0
    optimize_output: bool = False

@dataclass
class ImportOptions:
    """Options for importing quantum circuits from various formats."""
    strict_parsing: bool = True
    include_barriers: bool = True
    custom_gate_definitions: Optional[Dict[str, Any]] = None
    parameter_substitution: Optional[Dict[str, float]] = None
    validate_circuit: bool = True
    preserve_comments: bool = False

class SerializationCapability(Enum):
    """Capabilities supported by serialization formats."""
    GATES = "gates"
    MEASUREMENTS = "measurements"
    CLASSICAL_CONTROL = "classical_control"  
    PARAMETERS = "parameters"
    METADATA = "metadata"
    CUSTOM_GATES = "custom_gates"

class FormatCapabilities:
    """Defines capabilities for each serialization format."""
    
    CAPABILITIES: Dict[SerializationFormat, List[SerializationCapability]] = {
        SerializationFormat.QASM: [
            SerializationCapability.GATES,
            SerializationCapability.MEASUREMENTS,
            SerializationCapability.CLASSICAL_CONTROL,
            SerializationCapability.PARAMETERS,
            SerializationCapability.CUSTOM_GATES
        ],
        SerializationFormat.QASM2: [
            SerializationCapability.GATES,
            SerializationCapability.MEASUREMENTS,
            SerializationCapability.PARAMETERS,
            SerializationCapability.CUSTOM_GATES
        ],
        SerializationFormat.QASM3: [
            SerializationCapability.GATES,
            SerializationCapability.MEASUREMENTS,
            SerializationCapability.CLASSICAL_CONTROL,
            SerializationCapability.PARAMETERS,
            SerializationCapability.CUSTOM_GATES,
            SerializationCapability.METADATA
        ],
        SerializationFormat.JSON: [
            SerializationCapability.GATES,
            SerializationCapability.MEASUREMENTS,
            SerializationCapability.CLASSICAL_CONTROL,
            SerializationCapability.PARAMETERS,
            SerializationCapability.METADATA,
            SerializationCapability.CUSTOM_GATES
        ],
        SerializationFormat.QOBJ: [
            SerializationCapability.GATES,
            SerializationCapability.MEASUREMENTS,
            SerializationCapability.PARAMETERS
        ],
        SerializationFormat.CIRQ: [
            SerializationCapability.GATES,
            SerializationCapability.MEASUREMENTS,
            SerializationCapability.PARAMETERS
        ],
        SerializationFormat.QISKIT: [
            SerializationCapability.GATES,
            SerializationCapability.MEASUREMENTS,
            SerializationCapability.PARAMETERS
        ]
    }
    
    @classmethod
    def supports_capability(cls, format: SerializationFormat, capability: SerializationCapability) -> bool:
        """Check if a format supports a specific capability."""
        return capability in cls.CAPABILITIES.get(format, [])
    
    @classmethod
    def get_capabilities(cls, format: SerializationFormat) -> List[SerializationCapability]:
        """Get all capabilities for a format."""
        return cls.CAPABILITIES.get(format, [])

class SerializerBase(ABC):
    """Abstract base class for circuit serializers."""
    
    def __init__(self, format: SerializationFormat):
        self.format = format
        self.capabilities = FormatCapabilities.get_capabilities(format)
    
    @abstractmethod
    def serialize(self, circuit, **kwargs) -> str:
        """Serialize a quantum circuit to string format."""
        pass
    
    @abstractmethod
    def deserialize(self, data: str, **kwargs):
        """Deserialize a string to quantum circuit."""
        pass
    
    def supports_capability(self, capability: SerializationCapability) -> bool:
        """Check if this serializer supports a capability."""
        return capability in self.capabilities
    
    def validate_circuit_compatibility(self, circuit) -> List[str]:
        """Validate if circuit is compatible with this format."""
        warnings = []
        
        # This would be implemented by specific serializers
        # to check circuit features against format capabilities
        
        return warnings

class SerializerRegistry:
    """Registry for managing available serializers."""
    
    def __init__(self):
        self._serializers: Dict[SerializationFormat, SerializerBase] = {}
    
    def register(self, serializer: SerializerBase):
        """Register a serializer."""
        self._serializers[serializer.format] = serializer
    
    def get_serializer(self, format: SerializationFormat) -> Optional[SerializerBase]:
        """Get a serializer for the specified format."""
        return self._serializers.get(format)
    
    def list_formats(self) -> List[SerializationFormat]:
        """List all available formats."""
        return list(self._serializers.keys())
    
    def is_format_supported(self, format: SerializationFormat) -> bool:
        """Check if a format is supported."""
        return format in self._serializers

# Global serializer registry
_registry = SerializerRegistry()

def get_registry() -> SerializerRegistry:
    """Get the global serializer registry."""
    return _registry

# Predefined export configurations
STRICT_QASM2_EXPORT = ExportOptions(
    format=SerializationFormat.QASM2,
    include_comments=False,
    flatten_custom_gates=True,
    use_includes=True,
    optimize_output=True
)

COMPATIBLE_QASM3_EXPORT = ExportOptions(
    format=SerializationFormat.QASM3,
    include_comments=True,
    flatten_custom_gates=False,
    use_includes=True,
    optimize_output=False
)

VERBOSE_QASM3_EXPORT = ExportOptions(
    format=SerializationFormat.QASM3,
    include_comments=True,
    flatten_custom_gates=False,
    use_includes=True,
    optimize_output=False,
    precision=18
)

JSON_EXPORT = ExportOptions(
    format=SerializationFormat.JSON,
    include_comments=True,
    flatten_custom_gates=False
) 