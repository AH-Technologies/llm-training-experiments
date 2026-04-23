#!/usr/bin/env python3
"""
GPU memory monitor using nvidia-smi polling.

Polls nvidia-smi in a background thread to capture peak GPU memory usage
across process boundaries (works for subprocesses that torch.cuda can't see).
"""

import os
import re
import subprocess
import threading
import time


class GPUMemoryMonitor:
    """
    Context manager that polls nvidia-smi to track peak GPU memory usage.

    Works across process boundaries - captures memory from subprocesses
    that torch.cuda.max_memory_allocated() cannot see.

    Usage:
        with GPUMemoryMonitor(poll_interval=1.0) as monitor:
            subprocess.run(["torchrun", ...])
        print(f"Peak memory: {monitor.peak_memory_gb:.1f} GB")
    """

    def __init__(self, poll_interval: float = 1.0, gpu_indices: list[int] | None = None):
        """
        Args:
            poll_interval: Seconds between nvidia-smi polls.
            gpu_indices: GPU indices to monitor. If None, parses CUDA_VISIBLE_DEVICES.
        """
        self.poll_interval = poll_interval
        self.gpu_indices = gpu_indices or self._get_gpu_indices()
        self.peak_memory_gb: float | None = None
        self._peak_memory_mib = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _get_gpu_indices() -> list[int]:
        """Parse CUDA_VISIBLE_DEVICES to determine which GPUs to monitor."""
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            try:
                return [int(idx.strip()) for idx in cuda_visible.split(",") if idx.strip()]
            except ValueError:
                pass
        # Fallback: monitor all GPUs
        return []

    def _poll_memory(self):
        """Background thread: poll nvidia-smi for memory usage."""
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(",")
                        if len(parts) != 2:
                            continue
                        try:
                            gpu_idx = int(parts[0].strip())
                            mem_mib = float(parts[1].strip())
                        except ValueError:
                            continue

                        # Only track our allocated GPUs (if specified)
                        if self.gpu_indices and gpu_idx not in self.gpu_indices:
                            continue

                        with self._lock:
                            if mem_mib > self._peak_memory_mib:
                                self._peak_memory_mib = mem_mib
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                # nvidia-smi not available or timed out
                pass

            self._stop_event.wait(self.poll_interval)

    def __enter__(self):
        self._stop_event.clear()
        self._peak_memory_mib = 0.0
        self._thread = threading.Thread(target=self._poll_memory, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            if self._peak_memory_mib > 0:
                self.peak_memory_gb = self._peak_memory_mib / 1024.0
            else:
                self.peak_memory_gb = None
        return False
