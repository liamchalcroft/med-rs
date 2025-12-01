"""
Custom exceptions for medrs Python API.

Provides enterprise-grade error handling with specific error types,
detailed error information, and recovery suggestions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
from pathlib import Path


class MedrsError(Exception):
    """Base exception for all medrs operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, suggestions: Optional[list[str]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        """Enhanced error string with suggestions."""
        base_msg = super().__str__()
        if self.suggestions:
            suggestions = "\n  ".join(f" {suggestion}" for suggestion in self.suggestions)
            return f"{base_msg}\n\nSuggestions:\n  {suggestions}"
        return base_msg

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "suggestions": self.suggestions
        }


class LoadError(MedrsError):
    """Raised when file loading fails."""

    def __init__(
        self,
        path: Union[str, Path],
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.path = str(path)
        self.reason = reason

        message = f"Failed to load '{self.path}': {reason}"
        suggestions = [
            f"Check if '{self.path}' exists and is readable",
            "Verify the file is a valid NIfTI format (.nii or .nii.gz)",
            "Ensure you have sufficient permissions to read the file",
            "Try running: file command to verify file format"
        ]

        if "corrupted" in reason.lower():
            suggestions.append("The file may be corrupted - try re-downloading or re-creating")

        super().__init__(message, details, suggestions)


class ValidationError(MedrsError):
    """Raised when input validation fails."""

    def __init__(
        self,
        parameter: str,
        value: Any,
        expected: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.parameter = parameter
        self.value = value
        self.expected = expected

        message = f"Invalid '{parameter}': {value} (expected {expected})"
        suggestions = [
            f"Check the documentation for valid '{parameter}' values",
            "Ensure the parameter type matches the expected type",
            f"Expected: {expected}"
        ]

        super().__init__(message, details, suggestions)


class DeviceError(MedrsError):
    """Raised when device operations fail."""

    def __init__(
        self,
        device: str,
        operation: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.device = device
        self.operation = operation
        self.reason = reason

        message = f"Device operation failed on '{device}' during {operation}: {reason}"
        suggestions = []

        if "cuda" in device.lower():
            suggestions.extend([
                "Ensure CUDA is properly installed and configured",
                "Check GPU availability with: torch.cuda.is_available()",
                "Verify GPU memory is sufficient for the operation",
                "Try using CPU device first: device='cpu'"
            ])

        if "not available" in reason.lower():
            suggestions.append("Install the appropriate deep learning framework (PyTorch/JAX)")

        super().__init__(message, details, suggestions)


class TransformError(MedrsError):
    """Raised when transform operations fail."""

    def __init__(
        self,
        operation: str,
        reason: str,
        input_shape: Optional[tuple] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.reason = reason
        self.input_shape = input_shape

        message = f"Transform '{operation}' failed: {reason}"
        suggestions = [
            f"Check input data compatibility with '{operation}'",
            "Verify the image orientation and spacing are valid",
            "Ensure sufficient memory for the transform operation"
        ]

        if input_shape:
            suggestions.append(f"Input shape: {input_shape}")

        if "memory" in reason.lower():
            suggestions.extend([
                "Try processing smaller patches",
                "Use crop-first loading to reduce memory usage",
                "Consider using a smaller output shape"
            ])

        super().__init__(message, details, suggestions)


class MemoryError(MedrsError):
    """Raised when memory allocation fails."""

    def __init__(
        self,
        operation: str,
        requested_mb: float,
        available_mb: float,
        details: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.requested_mb = requested_mb
        self.available_mb = available_mb

        message = f"Memory allocation failed during '{operation}': requested {requested_mb:.1f}MB, available {available_mb:.1f}MB"
        suggestions = [
            "Reduce input size or use smaller patches",
            "Use crop-first loading to minimize memory usage",
            "Process data in smaller batches",
            "Close other applications to free memory",
            f"Available memory: {available_mb:.1f}MB, needed: {requested_mb:.1f}MB"
        ]

        super().__init__(message, details, suggestions)


class ConfigurationError(MedrsError):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        component: str,
        setting: str,
        value: Any,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.component = component
        self.setting = setting
        self.value = value
        self.reason = reason

        message = f"Invalid configuration for {component}.{setting}: {value} - {reason}"
        suggestions = [
            f"Check {component} documentation for valid {setting} values",
            "Ensure configuration types match expected types",
            "Verify the setting is compatible with other configuration options"
        ]

        super().__init__(message, details, suggestions)


# Utility functions for error handling
def validate_path(path: Union[str, Path], must_exist: bool = True) -> Path:
    """Validate file path and raise appropriate errors."""
    path_obj = Path(path)

    if must_exist and not path_obj.exists():
        raise LoadError(
            path_obj,
            "File does not exist",
            {"absolute_path": str(path_obj.absolute())}
        )

    if path_obj.suffix not in {'.nii', '.nii.gz'}:
        raise ValidationError(
            "path",
            str(path_obj),
            "NIfTI file (.nii or .nii.gz)"
        )

    return path_obj


def validate_device(device: str) -> str:
    """Validate device string and raise appropriate errors."""
    if not isinstance(device, str):
        raise ValidationError(
            "device",
            type(device).__name__,
            "string"
        )

    device = device.lower()

    if device == "cpu":
        return device

    if device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                raise DeviceError(
                    device,
                    "device_check",
                    "CUDA not available",
                    {"cuda_available": False}
                )

            if device != "cuda" and ":" in device:
                # Check specific CUDA device
                try:
                    device_id = int(device.split(":")[1])
                    if device_id >= torch.cuda.device_count():
                        raise DeviceError(
                            device,
                            "device_check",
                            f"CUDA device {device_id} not available (0-{torch.cuda.device_count()-1})",
                            {"available_devices": torch.cuda.device_count()}
                        )
                except ValueError:
                    raise ValidationError(
                        "device",
                        device,
                        "valid CUDA device format (e.g., 'cuda' or 'cuda:0')"
                    )
        except ImportError:
            raise DeviceError(
                device,
                "device_check",
                "PyTorch not available for CUDA device validation"
            )

        return device

    raise ValidationError(
        "device",
        device,
        "valid device ('cpu', 'cuda', 'cuda:0', etc.)"
    )


def validate_shape(shape: Union[tuple, list], name: str = "shape") -> tuple:
    """Validate shape tuple/list and raise appropriate errors."""
    if not isinstance(shape, (tuple, list)):
        raise ValidationError(
            name,
            type(shape).__name__,
            "tuple or list of positive integers"
        )

    if len(shape) != 3:
        raise ValidationError(
            name,
            shape,
            "3-tuple/list of positive integers (x, y, z)"
        )

    try:
        shape_tuple = tuple(int(s) for s in shape)
    except (ValueError, TypeError):
        raise ValidationError(
            name,
            shape,
            "tuple/list of integers"
        )

    if any(s <= 0 for s in shape_tuple):
        raise ValidationError(
            name,
            shape_tuple,
            "tuple/list of positive integers"
        )

    return shape_tuple
