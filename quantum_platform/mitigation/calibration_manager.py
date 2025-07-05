"""
Calibration Data Management

This module manages calibration data for error mitigation techniques,
including caching, persistence, and automatic refresh policies.
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
import hashlib

from ..hardware.hal import QuantumHardwareBackend
from ..hardware.results import AggregatedResult
from ..errors import MitigationError
from ..observability.logging import get_logger
from .measurement_mitigation import CalibrationMatrix, MeasurementMitigator


@dataclass
class CalibrationData:
    """Stored calibration data for a device."""
    
    # Device identification
    backend_name: str
    device_id: str
    num_qubits: int
    
    # Calibration metadata
    calibration_type: str  # "measurement_mitigation", "zne_characterization", etc.
    created_at: datetime
    expires_at: datetime
    
    # Calibration parameters
    calibration_shots: int
    calibration_method: str
    
    # Quality metrics
    average_fidelity: float
    confidence_score: float
    
    # Stored data (serialized)
    data: Dict[str, Any]
    
    # Validation metadata
    checksum: str
    
    def is_valid(self) -> bool:
        """Check if calibration is still valid."""
        return datetime.now() < self.expires_at
    
    def is_expired(self) -> bool:
        """Check if calibration has expired."""
        return datetime.now() >= self.expires_at
    
    def get_age_hours(self) -> float:
        """Get age of calibration in hours."""
        age = datetime.now() - self.created_at
        return age.total_seconds() / 3600
    
    def validate_checksum(self) -> bool:
        """Validate data integrity using checksum."""
        current_checksum = self._calculate_checksum()
        return current_checksum == self.checksum
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for data integrity."""
        data_str = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'backend_name': self.backend_name,
            'device_id': self.device_id,
            'num_qubits': self.num_qubits,
            'calibration_type': self.calibration_type,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'calibration_shots': self.calibration_shots,
            'calibration_method': self.calibration_method,
            'average_fidelity': self.average_fidelity,
            'confidence_score': self.confidence_score,
            'data': self.data,
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationData':
        """Create from dictionary."""
        return cls(
            backend_name=data['backend_name'],
            device_id=data['device_id'],
            num_qubits=data['num_qubits'],
            calibration_type=data['calibration_type'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']),
            calibration_shots=data['calibration_shots'],
            calibration_method=data['calibration_method'],
            average_fidelity=data['average_fidelity'],
            confidence_score=data['confidence_score'],
            data=data['data'],
            checksum=data['checksum']
        )


@dataclass
class CalibrationResult:
    """Result of calibration process."""
    
    # Calibration success
    success: bool
    error_message: Optional[str] = None
    
    # Calibration data
    calibration_data: Optional[CalibrationData] = None
    
    # Execution metadata
    execution_time: float = 0.0
    circuits_executed: int = 0
    total_shots: int = 0
    
    # Quality assessment
    quality_score: float = 0.0
    recommended_refresh_hours: float = 24.0


class CalibrationManager:
    """Manages calibration data for error mitigation."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.logger = get_logger(__name__)
        self._cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "calibration_cache"
        self._cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache
        self._memory_cache: Dict[str, CalibrationData] = {}
        self._cache_lock = threading.Lock()
        
        # Calibration policies
        self._default_expiry_hours = 24.0
        self._max_cache_age_hours = 72.0
        self._auto_refresh_threshold = 0.7  # Refresh when 70% of lifetime passed
        
        # Load existing calibrations
        self._load_cached_calibrations()
        
    def _get_cache_key(self, backend_name: str, device_id: str, 
                      num_qubits: int, calibration_type: str) -> str:
        """Generate cache key for calibration data."""
        return f"{backend_name}_{device_id}_{num_qubits}_{calibration_type}"
    
    def _get_cache_filename(self, cache_key: str) -> Path:
        """Get cache filename for a cache key."""
        return self._cache_dir / f"{cache_key}.json"
    
    def _load_cached_calibrations(self):
        """Load calibrations from disk cache."""
        try:
            for cache_file in self._cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    
                    calibration = CalibrationData.from_dict(data)
                    
                    # Validate integrity
                    if not calibration.validate_checksum():
                        self.logger.warning(f"Corrupted calibration data: {cache_file}")
                        continue
                    
                    # Check if expired
                    if calibration.is_expired():
                        self.logger.info(f"Expired calibration removed: {cache_file}")
                        cache_file.unlink()
                        continue
                    
                    # Add to memory cache
                    cache_key = self._get_cache_key(
                        calibration.backend_name,
                        calibration.device_id,
                        calibration.num_qubits,
                        calibration.calibration_type
                    )
                    
                    with self._cache_lock:
                        self._memory_cache[cache_key] = calibration
                    
                    self.logger.info(f"Loaded calibration: {cache_key}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load calibration from {cache_file}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error loading calibrations: {e}")
    
    def _save_calibration(self, calibration: CalibrationData):
        """Save calibration to disk cache."""
        try:
            cache_key = self._get_cache_key(
                calibration.backend_name,
                calibration.device_id,
                calibration.num_qubits,
                calibration.calibration_type
            )
            
            cache_file = self._get_cache_filename(cache_key)
            
            with open(cache_file, 'w') as f:
                json.dump(calibration.to_dict(), f, indent=2)
            
            self.logger.info(f"Saved calibration to {cache_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save calibration: {e}")
    
    def get_calibration(self, backend_name: str, device_id: str, 
                       num_qubits: int, calibration_type: str) -> Optional[CalibrationData]:
        """Get calibration data if available and valid."""
        
        cache_key = self._get_cache_key(backend_name, device_id, num_qubits, calibration_type)
        
        with self._cache_lock:
            if cache_key in self._memory_cache:
                calibration = self._memory_cache[cache_key]
                
                # Check if valid
                if calibration.is_valid():
                    return calibration
                else:
                    # Remove expired calibration
                    del self._memory_cache[cache_key]
                    
                    # Remove from disk
                    cache_file = self._get_cache_filename(cache_key)
                    if cache_file.exists():
                        cache_file.unlink()
                    
                    self.logger.info(f"Expired calibration removed: {cache_key}")
        
        return None
    
    def store_calibration(self, backend_name: str, device_id: str, 
                         num_qubits: int, calibration_type: str,
                         calibration_data: Any, calibration_shots: int = 1000,
                         calibration_method: str = "default",
                         expiry_hours: Optional[float] = None) -> CalibrationData:
        """Store calibration data."""
        
        # Determine expiry time
        expiry_hours = expiry_hours or self._default_expiry_hours
        expires_at = datetime.now() + timedelta(hours=expiry_hours)
        
        # Calculate quality metrics
        average_fidelity = self._calculate_average_fidelity(calibration_data)
        confidence_score = self._calculate_confidence_score(calibration_data, calibration_shots)
        
        # Serialize calibration data
        if hasattr(calibration_data, 'to_dict'):
            data_dict = calibration_data.to_dict()
        else:
            data_dict = calibration_data
        
        # Create calibration data object
        calibration = CalibrationData(
            backend_name=backend_name,
            device_id=device_id,
            num_qubits=num_qubits,
            calibration_type=calibration_type,
            created_at=datetime.now(),
            expires_at=expires_at,
            calibration_shots=calibration_shots,
            calibration_method=calibration_method,
            average_fidelity=average_fidelity,
            confidence_score=confidence_score,
            data=data_dict,
            checksum=""  # Will be calculated
        )
        
        # Calculate and set checksum
        calibration.checksum = calibration._calculate_checksum()
        
        # Store in memory cache
        cache_key = self._get_cache_key(backend_name, device_id, num_qubits, calibration_type)
        with self._cache_lock:
            self._memory_cache[cache_key] = calibration
        
        # Save to disk
        self._save_calibration(calibration)
        
        self.logger.info(f"Stored calibration: {cache_key}")
        
        return calibration
    
    def _calculate_average_fidelity(self, calibration_data: Any) -> float:
        """Calculate average fidelity from calibration data."""
        try:
            if isinstance(calibration_data, CalibrationMatrix):
                # Use readout fidelity
                if calibration_data.readout_fidelity:
                    return float(sum(calibration_data.readout_fidelity.values()) / len(calibration_data.readout_fidelity))
            
            # Default fidelity
            return 0.95
            
        except Exception:
            return 0.95
    
    def _calculate_confidence_score(self, calibration_data: Any, shots: int) -> float:
        """Calculate confidence score based on calibration data and shots."""
        try:
            # Base confidence on number of shots
            base_confidence = min(shots / 10000, 1.0)  # Max confidence at 10k shots
            
            # Adjust based on data quality
            if isinstance(calibration_data, CalibrationMatrix):
                # Higher fidelity = higher confidence
                avg_fidelity = self._calculate_average_fidelity(calibration_data)
                fidelity_factor = avg_fidelity
                
                return base_confidence * fidelity_factor
            
            return base_confidence
            
        except Exception:
            return 0.5
    
    def needs_refresh(self, backend_name: str, device_id: str, 
                     num_qubits: int, calibration_type: str) -> bool:
        """Check if calibration needs to be refreshed."""
        
        calibration = self.get_calibration(backend_name, device_id, num_qubits, calibration_type)
        
        if calibration is None:
            return True
        
        # Check if approaching expiry
        age_hours = calibration.get_age_hours()
        expiry_hours = (calibration.expires_at - calibration.created_at).total_seconds() / 3600
        
        age_ratio = age_hours / expiry_hours
        
        return age_ratio > self._auto_refresh_threshold
    
    def refresh_calibration(self, backend: QuantumHardwareBackend, 
                          num_qubits: int, calibration_type: str,
                          shots: int = 1000) -> CalibrationResult:
        """Refresh calibration data for a backend."""
        
        start_time = datetime.now()
        
        try:
            device_id = getattr(backend, 'device_name', backend.name)
            
            if calibration_type == "measurement_mitigation":
                # Perform measurement calibration
                result = self._refresh_measurement_calibration(backend, num_qubits, shots)
            else:
                raise ValueError(f"Unknown calibration type: {calibration_type}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result.success:
                self.logger.info(f"Calibration refreshed: {backend.name}, {num_qubits} qubits, {calibration_type}")
            else:
                self.logger.error(f"Calibration refresh failed: {result.error_message}")
            
            result.execution_time = execution_time
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return CalibrationResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _refresh_measurement_calibration(self, backend: QuantumHardwareBackend, 
                                       num_qubits: int, shots: int) -> CalibrationResult:
        """Refresh measurement error calibration."""
        
        try:
            # Use measurement mitigator to perform calibration
            mitigator = MeasurementMitigator()
            
            # Generate and execute calibration circuits
            calibration_circuits = mitigator.generate_calibration_circuits(num_qubits)
            calibration_results = []
            
            circuits_executed = 0
            total_shots = 0
            
            for circuit in calibration_circuits:
                # Execute circuit
                result = backend.submit_and_wait(circuit, shots)
                
                # Convert to AggregatedResult
                aggregated_result = AggregatedResult(
                    counts=result.counts,
                    total_shots=shots,
                    successful_shots=shots,
                    backend_name=backend.name
                )
                calibration_results.append(aggregated_result)
                
                circuits_executed += 1
                total_shots += shots
            
            # Build calibration matrix
            calibration_matrix = mitigator.build_calibration_matrix(calibration_results)
            
            # Store calibration
            device_id = getattr(backend, 'device_name', backend.name)
            calibration_data = self.store_calibration(
                backend_name=backend.name,
                device_id=device_id,
                num_qubits=num_qubits,
                calibration_type="measurement_mitigation",
                calibration_data=calibration_matrix,
                calibration_shots=total_shots,
                calibration_method="confusion_matrix"
            )
            
            # Calculate quality score
            quality_score = calibration_data.confidence_score
            
            return CalibrationResult(
                success=True,
                calibration_data=calibration_data,
                circuits_executed=circuits_executed,
                total_shots=total_shots,
                quality_score=quality_score,
                recommended_refresh_hours=self._default_expiry_hours
            )
            
        except Exception as e:
            return CalibrationResult(
                success=False,
                error_message=str(e)
            )
    
    def get_cached_calibrations(self) -> List[CalibrationData]:
        """Get all cached calibrations."""
        with self._cache_lock:
            return list(self._memory_cache.values())
    
    def clear_expired_calibrations(self) -> int:
        """Clear expired calibrations from cache."""
        cleared_count = 0
        
        with self._cache_lock:
            expired_keys = []
            for key, calibration in self._memory_cache.items():
                if calibration.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
                cleared_count += 1
                
                # Remove from disk
                cache_file = self._get_cache_filename(key)
                if cache_file.exists():
                    cache_file.unlink()
        
        if cleared_count > 0:
            self.logger.info(f"Cleared {cleared_count} expired calibrations")
        
        return cleared_count
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get statistics about calibration cache."""
        with self._cache_lock:
            total_calibrations = len(self._memory_cache)
            
            # Count by type
            type_counts = {}
            quality_scores = []
            ages = []
            
            for calibration in self._memory_cache.values():
                cal_type = calibration.calibration_type
                type_counts[cal_type] = type_counts.get(cal_type, 0) + 1
                
                quality_scores.append(calibration.confidence_score)
                ages.append(calibration.get_age_hours())
            
            return {
                'total_calibrations': total_calibrations,
                'calibration_types': type_counts,
                'average_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                'average_age_hours': sum(ages) / len(ages) if ages else 0.0,
                'cache_directory': str(self._cache_dir)
            }


# Global instance
_calibration_manager = None
_manager_lock = threading.Lock()


def get_calibration_manager() -> CalibrationManager:
    """Get global calibration manager instance."""
    global _calibration_manager
    if _calibration_manager is None:
        with _manager_lock:
            if _calibration_manager is None:
                _calibration_manager = CalibrationManager()
    return _calibration_manager


def refresh_calibration(backend: QuantumHardwareBackend, 
                       num_qubits: int, calibration_type: str,
                       shots: int = 1000) -> CalibrationResult:
    """Refresh calibration data for a backend."""
    manager = get_calibration_manager()
    return manager.refresh_calibration(backend, num_qubits, calibration_type, shots)


def get_cached_calibration(backend_name: str, device_id: str, 
                          num_qubits: int, calibration_type: str) -> Optional[CalibrationData]:
    """Get cached calibration data."""
    manager = get_calibration_manager()
    return manager.get_calibration(backend_name, device_id, num_qubits, calibration_type) 