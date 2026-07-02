
Error Handling
===============

This guide covers comprehensive error handling strategies and debugging techniques for medrs applications.


Exception Hierarchy
-------------------

medrs's Python API (``medrs.exceptions``) provides a structured exception hierarchy, all deriving from ``MedrsError``, that carries structured ``details`` and actionable ``suggestions`` alongside the message.

Exception Classes
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MedrsError(Exception):
       """Base exception for all medrs operations."""
       def __init__(self, message, details=None, suggestions=None):
           ...
           self.details = details or {}
           self.suggestions = suggestions or []

   class LoadError(MedrsError):
       """Raised when file loading fails (missing file, invalid format,
       corrupted data, etc.). Carries `.path` and `.reason`."""

   class ValidationError(MedrsError):
       """Raised when an input parameter fails validation. Carries
       `.parameter`, `.value`, `.expected`."""

   class DeviceError(MedrsError):
       """Raised when a device (e.g. CUDA) operation fails. Carries
       `.device`, `.operation`, `.reason`."""

   class TransformError(MedrsError):
       """Raised when a transform operation fails. Carries `.operation`,
       `.reason`, `.input_shape`."""

   class MemoryError(MedrsError):
       """Raised when memory allocation fails. Carries `.operation`,
       `.requested_mb`, `.available_mb`. Shadows the Python builtin
       intentionally; import it from `medrs.exceptions` explicitly if you
       need to disambiguate."""

   class ConfigurationError(MedrsError):
       """Raised when configuration is invalid. Carries `.component`,
       `.setting`, `.value`, `.reason`."""

See ``src/python/medrs/exceptions.py`` for the full implementation, including the ``validate_path``, ``validate_device``, and ``validate_shape`` helpers.

Common Error Scenarios
----------------------

File Loading Errors
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import medrs
   from medrs.exceptions import LoadError

   def safe_load_volume(path: str) -> medrs.MedicalImage:
       try:
           return medrs.load(path)
       except LoadError as e:
           print(f"Failed to load '{e.path}': {e.reason}")
           for suggestion in e.suggestions:
               print(f"  - {suggestion}")
           raise

Device Errors
~~~~~~~~~~~~~

.. code-block:: python

   from medrs.exceptions import DeviceError, validate_device

   def load_to_device(path: str, device: str) -> "torch.Tensor":
       """Validate the device string before touching CUDA."""
       device = validate_device(device)  # raises ValidationError/DeviceError early
       return medrs.load_to_torch(path, device=device)

Memory-Constrained Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from medrs.exceptions import MemoryError as MedrsMemoryError

   def load_with_fallback(path: str, patch_size=(64, 64, 64)) -> medrs.MedicalImage:
       """Fall back to crop-first loading if the full volume doesn't fit."""
       try:
           return medrs.load(path)
       except MedrsMemoryError as e:
           print(f"Full-volume load failed ({e.requested_mb:.0f}MB requested, "
                 f"{e.available_mb:.0f}MB available); falling back to a "
                 f"{patch_size} crop.")
           return medrs.load_cropped(path, [0, 0, 0], list(patch_size))

Advanced Error Recovery
-----------------------

Retry Mechanisms
~~~~~~~~~~~~~~~~

Implement intelligent retry logic for transient errors:

.. code-block:: python

   import time
   import random
   from typing import Optional, Callable

   def with_retry(max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True) -> Callable:
       """Decorator for retry logic with exponential backoff"""

       def decorator(func: Callable):
           def wrapper(*args, **kwargs):
               last_exception = None

               for attempt in range(max_attempts):
                   try:
                       return func(*args, **kwargs)
                   except (medrs.LoadError, medrs.MemoryError) as e:
                       last_exception = e

                       if attempt == max_attempts - 1:
                           raise

                       # Calculate delay with exponential backoff
                       delay = min(base_delay * (exponential_base ** attempt), max_delay)

                       if jitter:
                           delay *= (0.5 + random.random() * 0.5)

                       print(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s...")
                       time.sleep(delay)

               raise last_exception
           return wrapper
       return decorator

   # Usage
   @with_retry(max_attempts=5, base_delay=0.5)
   def robust_load(path: str) -> medrs.MedicalImage:
       return medrs.load(path)

Fallback Strategies
~~~~~~~~~~~~~~~~~~~

Prefer explicit failure over hidden fallbacks so issues surface quickly. Keep a single loading path per workflow and add targeted retries only when they add value (for example transient I/O errors).

Debugging Techniques
--------------------

Error Context Collection
~~~~~~~~~~~~~~~~~~~~~~~~~

Collect detailed context for debugging:

.. code-block:: python

   import sys
   import traceback
   from pathlib import Path
   import psutil

   class DebugContext:
       def __init__(self):
           self.system_info = self._collect_system_info()
           self.memory_info = self._collect_memory_info()
           self.file_info = None

       def _collect_system_info(self) -> dict:
           return {
               'platform': sys.platform,
               'python_version': sys.version,
               'cpu_count': psutil.cpu_count(),
               'memory_total_gb': psutil.virtual_memory().total / (1024**3)
           }

       def _collect_memory_info(self) -> dict:
           process = psutil.Process()
           return {
               'memory_mb': process.memory_info().rss / (1024**2),
               'memory_percent': process.memory_percent(),
               'available_memory_gb': psutil.virtual_memory().available / (1024**3)
           }

       def analyze_file(self, path: str) -> dict:
           path_obj = Path(path)

           if not path_obj.exists():
               return {'exists': False, 'path': str(path_obj.absolute())}

           stat = path_obj.stat()

           return {
               'exists': True,
               'path': str(path_obj.absolute()),
               'size_mb': stat.st_size / (1024**2),
               'extension': path_obj.suffix,
               'readable': path_obj.is_file(),
               'permissions': oct(stat.st_mode)[-3:]
           }

       def generate_error_report(self, error: Exception,
                               file_path: str = None) -> str:
           report = []
           report.append("=== MEDRS ERROR REPORT ===")
           report.append(f"Error: {type(error).__name__}: {error}")
           report.append(f"Traceback: {traceback.format_exc()}")

           if file_path:
               report.append("\n=== FILE ANALYSIS ===")
               file_info = self.analyze_file(file_path)
               for key, value in file_info.items():
                   report.append(f"{key}: {value}")

           report.append("\n=== SYSTEM INFO ===")
           for key, value in self.system_info.items():
               report.append(f"{key}: {value}")

           report.append("\n=== MEMORY INFO ===")
           for key, value in self.memory_info.items():
               report.append(f"{key}: {value}")

           return "\n".join(report)

   # Usage
   def debug_load(path: str) -> medrs.MedicalImage:
       debug_context = DebugContext()

       try:
           return medrs.load(path)
       except Exception as e:
           report = debug_context.generate_error_report(e, path)
           print(report)

           # Optionally save to file
           with open("medrs_error_report.txt", "w") as f:
               f.write(report)

           raise

Performance Debugging
~~~~~~~~~~~~~~~~~~~~~

Debug performance-related issues:

.. code-block:: python

   import time
   from contextlib import contextmanager

   @contextmanager
   def performance_monitor(operation_name: str):
       start_time = time.time()
       start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

       try:
           yield
       finally:
           end_time = time.time()
           end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

           duration = end_time - start_time
           memory_delta = (end_memory - start_memory) / (1024**2)  # MB

           print(f"{operation_name}:")
           print(f"  Duration: {duration:.3f}s")
           print(f"  Memory delta: {memory_delta:+.1f}MB")

   def debug_load_performance(path: str):
       with performance_monitor(f"Loading {path}"):
           img = medrs.load(path)

       with performance_monitor(f"Converting to tensor"):
           tensor = img.to_torch()

       return tensor

Validation and Testing
----------------------

Input Validation
~~~~~~~~~~~~~~~~

Implement comprehensive input validation:

.. code-block:: python

   from typing import Sequence, Union
   import numpy as np

   def validate_load_params(path: str,
                           crop_offset: Sequence[int] = None,
                           crop_shape: Sequence[int] = None) -> None:
       """Validate parameters for loading operations"""

       # Validate path
       if not isinstance(path, str):
           raise TypeError(f"Path must be string, got {type(path)}")

       if not path.strip():
           raise ValueError("Path cannot be empty")

       # Validate crop parameters
       if crop_offset is not None:
           if not isinstance(crop_offset, (list, tuple)):
               raise TypeError("crop_offset must be list or tuple")
           if len(crop_offset) != 3:
               raise ValueError("crop_offset must have 3 elements")
           if any(x < 0 for x in crop_offset):
               raise ValueError("crop_offset values must be non-negative")

       if crop_shape is not None:
           if not isinstance(crop_shape, (list, tuple)):
               raise TypeError("crop_shape must be list or tuple")
           if len(crop_shape) != 3:
               raise ValueError("crop_shape must have 3 elements")
           if any(x <= 0 for x in crop_shape):
               raise ValueError("crop_shape values must be positive")

   def safe_load_with_validation(path: str, **kwargs) -> medrs.MedicalImage:
       """Load with comprehensive input validation"""

       validate_load_params(path,
                          kwargs.get('crop_offset'),
                          kwargs.get('crop_shape'))

       return medrs.load(path, **kwargs)

Property-Based Testing
~~~~~~~~~~~~~~~~~~~~~~

Use property-based testing for robust error handling:

.. code-block:: python

   import hypothesis
   from hypothesis import given, strategies as st

   @given(st.lists(st.integers(min_value=1, max_value=1000), min_size=3, max_size=3))
   def test_crop_shape_validation(shape):
       """Test crop shape validation with various inputs"""

       # Valid shapes should pass
       try:
           validate_load_params("test.nii", crop_shape=shape)
       except (ValueError, TypeError):
           pass  # Expected for some invalid shapes

   @given(st.text())
   def test_path_validation(path):
       """Test path validation with various inputs"""

       if not path.strip():
           with pytest.raises(ValueError):
               validate_load_params(path)
       else:
           # Should not raise validation errors for path format
           validate_load_params(path)

Error Logging and Monitoring
------------------------------

Structured Error Logging
~~~~~~~~~~~~~~~~~~~~~~~~

Implement structured logging for production environments:

.. code-block:: python

   import logging
   import json
   from datetime import datetime

   class MedrsLogger:
       def __init__(self, name: str = "medrs"):
           self.logger = logging.getLogger(name)
           self.logger.setLevel(logging.INFO)

           # Create handler
           handler = logging.StreamHandler()
           formatter = logging.Formatter(
               '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
           )
           handler.setFormatter(formatter)
           self.logger.addHandler(handler)

       def log_error(self, error: Exception, context: dict = None):
           error_data = {
               'timestamp': datetime.utcnow().isoformat(),
               'error_type': type(error).__name__,
               'error_message': str(error),
               'context': context or {}
           }

           if hasattr(error, 'suggestions'):
               error_data['suggestions'] = error.suggestions

           self.logger.error(f"MEDRS Error: {json.dumps(error_data, indent=2)}")

   # Usage
   logger = MedrsLogger()

   def logged_load(path: str) -> medrs.MedicalImage:
       try:
           return medrs.load(path)
       except Exception as e:
           logger.log_error(e, context={'file_path': path})
           raise

Monitoring and Alerting
~~~~~~~~~~~~~~~~~~~~~~~

Set up monitoring for error rates:

.. code-block:: python

   from collections import defaultdict, deque
   import time

   class ErrorMonitor:
       def __init__(self, window_size: int = 100):
           self.error_counts = defaultdict(lambda: deque(maxlen=window_size))
           self.window_size = window_size

       def record_error(self, error_type: str, timestamp: float = None):
           if timestamp is None:
               timestamp = time.time()

           self.error_counts[error_type].append(timestamp)

       def get_error_rate(self, error_type: str, window_seconds: int = 60) -> float:
           timestamps = self.error_counts[error_type]
           current_time = time.time()

           recent_errors = [
               ts for ts in timestamps
               if current_time - ts <= window_seconds
           ]

           return len(recent_errors) / window_seconds

       def check_alert_thresholds(self):
           """Check if error rates exceed alert thresholds"""
           for error_type in self.error_counts:
               rate = self.get_error_rate(error_type)
               if rate > 0.1:  # More than 0.1 errors per second
                   print(f"ALERT: High error rate for {error_type}: {rate:.2f}/s")

   # Global error monitor
   error_monitor = ErrorMonitor()

   def monitored_operation(func):
       def wrapper(*args, **kwargs):
           try:
               return func(*args, **kwargs)
           except Exception as e:
               error_monitor.record_error(type(e).__name__)
               error_monitor.check_alert_thresholds()
               raise
       return wrapper

This comprehensive error handling guide provides strategies for building robust medrs applications that can gracefully handle errors and provide actionable debugging information.
