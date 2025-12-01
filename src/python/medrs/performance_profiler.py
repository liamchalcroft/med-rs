"""
Performance profiling utilities for medrs operations.

Provides comprehensive performance monitoring, memory tracking,
and optimization recommendations for medrs workflows.
"""

from __future__ import annotations

import time
import gc
import threading
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict, deque

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    operation: str
    duration_ms: float
    memory_mb: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSummary:
    """Summary of performance metrics."""
    operation: str
    count: int
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    total_duration_ms: float
    avg_memory_mb: float
    peak_memory_mb: float


class PerformanceProfiler:
    """Comprehensive performance profiler for medrs operations."""

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: deque[PerformanceMetric] = deque(maxlen=max_history)
        self.active_operations: Dict[str, float] = {}
        self._lock = threading.Lock()

    def start_operation(self, operation: str) -> str:
        """Start timing an operation."""
        operation_id = f"{operation}_{int(time.time() * 1000000)}"
        with self._lock:
            self.active_operations[operation_id] = time.time()
        return operation_id

    def end_operation(
        self,
        operation_id: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """End timing an operation and record metrics."""
        end_time = time.time()

        with self._lock:
            if operation_id not in self.active_operations:
                return 0.0

            start_time = self.active_operations.pop(operation_id)
            duration_ms = (end_time - start_time) * 1000
            memory_mb = self._get_memory_usage()

            metric = PerformanceMetric(
                operation=operation,
                duration_ms=duration_ms,
                memory_mb=memory_mb,
                timestamp=start_time,
                metadata=metadata or {}
            )

            self.metrics.append(metric)
            return duration_ms

    @contextmanager
    def profile(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for profiling operations."""
        operation_id = self.start_operation(operation)
        try:
            yield
        finally:
            self.end_operation(operation_id, operation, metadata)

    def get_summary(self, operation: Optional[str] = None) -> Dict[str, PerformanceSummary]:
        """Get performance summary for operations."""
        with self._lock:
            filtered_metrics = [
                m for m in self.metrics
                if operation is None or m.operation == operation
            ]

        if not filtered_metrics:
            return {}

        summaries = defaultdict(list)
        for metric in filtered_metrics:
            summaries[metric.operation].append(metric)

        result = {}
        for op_name, op_metrics in summaries.items():
            durations = [m.duration_ms for m in op_metrics]
            memories = [m.memory_mb for m in op_metrics]

            result[op_name] = PerformanceSummary(
                operation=op_name,
                count=len(op_metrics),
                avg_duration_ms=sum(durations) / len(durations),
                min_duration_ms=min(durations),
                max_duration_ms=max(durations),
                total_duration_ms=sum(durations),
                avg_memory_mb=sum(memories) / len(memories),
                peak_memory_mb=max(memories)
            )

        return result

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self._get_memory_usage()

    def _get_memory_usage(self) -> float:
        """Internal memory usage calculation."""
        total_mb = 0.0

        # PyTorch CUDA memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            total_mb += torch.cuda.memory_allocated() / (1024**2)

        # Add other memory tracking as needed
        return total_mb

    def clear(self):
        """Clear all recorded metrics."""
        with self._lock:
            self.metrics.clear()
            self.active_operations.clear()

    def export_metrics(self) -> List[Dict[str, Any]]:
        """Export metrics as a list of dictionaries."""
        with self._lock:
            return [
                {
                    "operation": m.operation,
                    "duration_ms": m.duration_ms,
                    "memory_mb": m.memory_mb,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata
                }
                for m in self.metrics
            ]

    def find_bottlenecks(self, threshold_ms: float = 100.0) -> List[PerformanceSummary]:
        """Find operations that exceed the performance threshold."""
        summaries = self.get_summary()
        bottlenecks = []

        for summary in summaries.values():
            if summary.avg_duration_ms > threshold_ms:
                bottlenecks.append(summary)

        return sorted(bottlenecks, key=lambda x: x.avg_duration_ms, reverse=True)


# Global profiler instance
_global_profiler = PerformanceProfiler()


def profile(operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator for profiling functions."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with _global_profiler.profile(operation, metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def benchmark_operation(
    operation: Callable,
    num_runs: int = 10,
    warmup_runs: int = 2,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """Benchmark an operation with multiple runs."""
    # Warmup runs
    for _ in range(warmup_runs):
        operation()

    gc.collect()  # Clear memory before benchmarking

    # Timed runs
    durations = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        operation()
        end_time = time.perf_counter()
        durations.append((end_time - start_time) * 1000)  # Convert to ms

    return {
        "avg_duration_ms": sum(durations) / len(durations),
        "min_duration_ms": min(durations),
        "max_duration_ms": max(durations),
        "std_duration_ms": (sum((d - sum(durations)/len(durations))**2 for d in durations) / len(durations))**0.5,
        "total_runs": num_runs
    }


class MemoryMonitor:
    """Monitor memory usage during operations."""

    def __init__(self):
        self.baseline_memory = 0.0
        self.peak_memory = 0.0
        self.memory_history = []

    def start_monitoring(self):
        """Start memory monitoring."""
        gc.collect()
        self.baseline_memory = _global_profiler._get_memory_usage()
        self.peak_memory = self.baseline_memory

    def record_memory(self):
        """Record current memory usage."""
        current = _global_profiler._get_memory_usage()
        self.memory_history.append(current)
        self.peak_memory = max(self.peak_memory, current)

    def get_memory_delta(self) -> float:
        """Get memory increase from baseline."""
        return _global_profiler._get_memory_usage() - self.baseline_memory

    def get_peak_increase(self) -> float:
        """Get peak memory increase from baseline."""
        return self.peak_memory - self.baseline_memory


@contextmanager
def monitor_memory():
    """Context manager for memory monitoring."""
    monitor = MemoryMonitor()
    monitor.start_monitoring()

    try:
        yield monitor
    finally:
        monitor.record_memory()


# Utility functions
def format_duration(ms: float) -> str:
    """Format duration in human readable format."""
    if ms < 1:
        return f"{ms*1000:.1f}mus"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"


def format_memory(mb: float) -> str:
    """Format memory size in human readable format."""
    if mb < 1024:
        return f"{mb:.1f}MB"
    else:
        return f"{mb/1024:.1f}GB"
