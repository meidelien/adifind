#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Utilities and Monitoring Module
=======================================

Provides system monitoring, memory management, and cross-instance coordination
for AdiFind WSI analysis.
"""

import os
import gc
import errno
import time
import logging
import datetime
import tempfile
import subprocess
import threading
from typing import Optional, Dict, Any

# Import configuration
from config import config

# Try GPU/system monitoring imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# File locking imports
try:
    import fcntl  # For file locking (Unix/Linux)
except ImportError:
    fcntl = None
try:
    import msvcrt  # For file locking (Windows)
except ImportError:
    msvcrt = None


class _LabelMappingLockBusy(Exception):
    """Raised when the label-mapping byte-range lock is held elsewhere."""


class _LabelMappingLockHandle:
    """Small wrapper around the lock file so release can unlock before close."""

    def __init__(self, file_obj, path):
        self.file_obj = file_obj
        self.path = path
        self.locked = True


def _is_retryable_lock_error(error):
    """Return True only for OS errors that represent lock contention."""
    retryable_errnos = {errno.EACCES, errno.EAGAIN}
    if hasattr(errno, 'EDEADLK'):
        retryable_errnos.add(errno.EDEADLK)
    if hasattr(errno, 'EWOULDBLOCK'):
        retryable_errnos.add(errno.EWOULDBLOCK)
    return (
        isinstance(error, (BlockingIOError, OSError))
        and getattr(error, 'errno', None) in retryable_errnos
    )


def _open_label_mapping_lock_file(lock_path):
    """Open or create the label-mapping lock file without taking the OS lock."""
    lock_dir = os.path.dirname(os.path.abspath(lock_path))
    if lock_dir and not os.path.isdir(lock_dir):
        raise RuntimeError(
            f"Label mapping lock directory does not exist: {lock_dir}. "
            f"Lock path: {lock_path}"
        )

    try:
        return open(lock_path, 'r+', encoding='utf-8')
    except FileNotFoundError:
        try:
            return open(lock_path, 'w+', encoding='utf-8')
        except OSError as e:
            raise RuntimeError(
                f"Could not create label mapping lock file at {lock_path}: {e}"
            ) from e
    except OSError as e:
        raise RuntimeError(
            f"Could not open label mapping lock file at {lock_path}: {e}"
        ) from e


def _lock_label_mapping_file(lock_file, lock_path):
    """Try to take the platform lock, separating contention from hard errors."""
    try:
        lock_file.seek(0)
        if os.name == 'nt':
            if msvcrt is not None:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if _is_retryable_lock_error(e):
            raise _LabelMappingLockBusy(
                f"Label mapping lock is currently held at {lock_path}: {e}"
            ) from e
        raise RuntimeError(
            f"Could not acquire OS label mapping lock at {lock_path}: {e}"
        ) from e


def _write_label_mapping_lock_metadata(lock_file):
    """Record owner metadata after the OS lock has been acquired."""
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"PID: {os.getpid()}\nTime: {datetime.datetime.now()}\n")
    lock_file.flush()


def _unlock_label_mapping_file(lock_file):
    """Release the platform lock while the file handle is still open."""
    if lock_file is None or lock_file.closed:
        return

    try:
        lock_file.seek(0)
        if os.name == 'nt':
            if msvcrt is not None:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except (OSError, IOError) as e:
        logging.debug(f"Info: Error unlocking label mapping lock: {e}")


def _remove_label_mapping_lock_file(lock_path):
    """Best-effort cleanup of the lock marker file after unlocking."""
    if not lock_path:
        return

    max_attempts = 5 if os.name == 'nt' else 3
    for attempt in range(max_attempts):
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
                logging.debug(
                    f"\U0001F513 Lock file cleaned up successfully on attempt {attempt + 1}"
                )
            return
        except (OSError, IOError, PermissionError) as e:
            if attempt < max_attempts - 1:
                wait_time = 0.2 * (2 ** attempt) if os.name == 'nt' else 0.5
                time.sleep(wait_time)
                continue
            logging.debug(
                f"Info: Could not remove lock file after {max_attempts} attempts: {e}"
            )
            logging.debug(
                "\U0001F513 Lock released (file cleanup delayed, but process can continue)"
            )


def _cleanup_memory_before_label_mapping(get_memory_usage):
    """Reduce memory pressure before retrying the memory-heavy mapping step."""
    memory_usage = get_memory_usage()
    if memory_usage <= 94:
        return

    logging.info(f"\U0001F6A6 High memory ({memory_usage:.1f}%) - brief cleanup...")
    gc.collect()
    if TORCH_AVAILABLE:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    time.sleep(2)


def _acquire_label_mapping_lock_handle(lock_path, get_memory_usage, timeout=300):
    """Acquire the label-mapping lock with actionable failures and bounded retry."""
    start_time = time.time()
    next_wait_log = start_time + 30
    last_lock_error = None

    while True:
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            detail = f" Last lock error: {last_lock_error}" if last_lock_error else ""
            raise TimeoutError(
                f"Could not acquire label mapping lock at {lock_path} within {timeout} seconds.{detail}"
            )

        _cleanup_memory_before_label_mapping(get_memory_usage)
        lock_file = _open_label_mapping_lock_file(lock_path)

        try:
            _lock_label_mapping_file(lock_file, lock_path)
            try:
                _write_label_mapping_lock_metadata(lock_file)
            except Exception as e:
                _unlock_label_mapping_file(lock_file)
                raise RuntimeError(
                    f"Acquired label mapping lock at {lock_path}, but could not write metadata: {e}"
                ) from e

            logging.info("\U0001F512 Acquired label mapping lock at %s", lock_path)
            return _LabelMappingLockHandle(lock_file, lock_path)

        except _LabelMappingLockBusy as e:
            last_lock_error = e.__cause__ or e
            try:
                lock_file.close()
            except (OSError, IOError):
                pass

            now = time.time()
            if now >= next_wait_log:
                logging.info(
                    "\U0001F6A6 Waiting for label mapping lock at %s "
                    "(elapsed %.0fs, timeout %ss). Last error: %s",
                    lock_path,
                    now - start_time,
                    timeout,
                    last_lock_error,
                )
                next_wait_log = now + 30

            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                continue
            time.sleep(min(3, remaining))

        except Exception:
            try:
                lock_file.close()
            except (OSError, IOError):
                pass
            raise


def _release_label_mapping_lock_handle(lock_handle, fallback_path):
    """Release the OS lock before closing, then remove the marker file."""
    if not lock_handle:
        return

    lock_file = getattr(lock_handle, 'file_obj', lock_handle)
    lock_path = getattr(lock_handle, 'path', fallback_path)

    try:
        _unlock_label_mapping_file(lock_file)
        if hasattr(lock_handle, 'locked'):
            lock_handle.locked = False
    finally:
        try:
            if lock_file is not None and not lock_file.closed:
                lock_file.close()
        except (OSError, IOError) as e:
            logging.debug(f"Info: Error closing lock file (expected on some systems): {e}")

    if os.name == 'nt':
        time.sleep(0.1)
    _remove_label_mapping_lock_file(lock_path)
    logging.info("\U0001F513 Lock released, continuing processing...")


# ================================================================
# SYSTEM MONITORING
# ================================================================

class SystemMonitor:
    """Professional system monitoring with multiple fallback methods."""
    
    def __init__(self):
        self.gpu_handle = None
        self.monitoring_method = "none"
        self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """Initialize monitoring with multiple fallback options."""
        # Method 1: Try pynvml (NVIDIA official library)
        if self._try_pynvml():
            return
        
        # Method 2: Try nvidia-ml-py3 (alternative package name)
        if self._try_nvidia_ml_py3():
            return
        
        # Method 3: Try GPUtil (simple cross-platform library)
        if self._try_gputil():
            return
        
        # Method 4: Try psutil for CPU and basic system info
        if self._try_psutil():
            return
        
        # Method 5: Try subprocess calls to nvidia-smi
        if self._try_nvidia_smi():
            return
        
        logging.warning("\u26A0\uFE0F  No GPU monitoring libraries available. Using basic PyTorch monitoring only.")
        self.monitoring_method = "pytorch_basic"
    
    def _try_pynvml(self):
        """Try pynvml (py3nvml) library."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.pynvml = pynvml
            self.monitoring_method = "pynvml"
            logging.info("\u2705 GPU monitoring initialized with pynvml")
            return True
        except ImportError:
            logging.debug("pynvml not available")
            return False
        except Exception as e:
            logging.debug(f"pynvml initialization failed: {e}")
            return False
    
    def _try_nvidia_ml_py3(self):
        """Try nvidia-ml-py3 (alternative package)."""
        try:
            import nvidia_ml_py3.nvidia_ml_py as nvml
            nvml.nvmlInit()
            self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml = nvml
            self.monitoring_method = "nvidia_ml_py3"
            logging.info("\u2705 GPU monitoring initialized with nvidia-ml-py3")
            return True
        except ImportError:
            logging.debug("nvidia-ml-py3 not available")
            return False
        except Exception as e:
            logging.debug(f"nvidia-ml-py3 initialization failed: {e}")
            return False
    
    def _try_gputil(self):
        """Try GPUtil library (simple cross-platform)."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gputil = GPUtil
                self.monitoring_method = "gputil"
                logging.info("\u2705 GPU monitoring initialized with GPUtil")
                return True
            else:
                logging.debug("No GPUs found with GPUtil")
                return False
        except ImportError:
            logging.debug("GPUtil not available")
            return False
        except Exception as e:
            logging.debug(f"GPUtil initialization failed: {e}")
            return False
    
    def _try_psutil(self):
        """Try psutil for system monitoring."""
        try:
            import psutil
            self.psutil = psutil
            self.monitoring_method = "psutil"
            logging.info("\u2705 System monitoring initialized with psutil")
            return True
        except ImportError:
            logging.debug("psutil not available")
            return False
    
    def _try_nvidia_smi(self):
        """Try nvidia-smi command line tool."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.monitoring_method = "nvidia_smi"
                logging.info("\u2705 GPU monitoring initialized with nvidia-smi")
                return True
            else:
                logging.debug("nvidia-smi command failed")
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            logging.debug(f"nvidia-smi not available: {e}")
            return False
    
    def log_gpu_status(self, context: str = ""):
        """Log current GPU utilization and memory usage using available monitoring method."""
        # Check if GPU monitoring is enabled in configuration
        if not config.ENABLE_GPU_MONITORING:
            return
            
        try:
            if self.monitoring_method == "pynvml":
                self._log_with_pynvml(context)
            elif self.monitoring_method == "nvidia_ml_py3":
                self._log_with_nvidia_ml_py3(context)
            elif self.monitoring_method == "gputil":
                self._log_with_gputil(context)
            elif self.monitoring_method == "nvidia_smi":
                self._log_with_nvidia_smi(context)
            elif self.monitoring_method == "psutil":
                self._log_with_psutil(context)
            else:
                self._log_with_pytorch_basic(context)
        except Exception as e:
            logging.debug(f"GPU monitoring error with {self.monitoring_method}: {e}")
            self._log_with_pytorch_basic(context)
    
    def _log_with_pynvml(self, context: str):
        """Log using pynvml."""
        try:
            rates = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            memory_used_gb = memory_info.used / 1024**3
            memory_total_gb = memory_info.total / 1024**3
            logging.debug(f"\uD83D\uDD27 {context} | GPU: {rates.gpu:3d}% | Memory: {rates.memory:3d}% ({memory_used_gb:.1f}GB/{memory_total_gb:.1f}GB) [pynvml]")
        except Exception as e:
            logging.debug(f"pynvml error: {e}")
            self._log_with_pytorch_basic(context)
    
    def _log_with_nvidia_ml_py3(self, context: str):
        """Log using nvidia-ml-py3."""
        try:
            rates = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            memory_info = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            memory_used_gb = memory_info.used / 1024**3
            memory_total_gb = memory_info.total / 1024**3
            logging.debug(f"\uD83D\uDD27 {context} | GPU: {rates.gpu:3d}% | Memory: {rates.memory:3d}% ({memory_used_gb:.1f}GB/{memory_total_gb:.1f}GB) [nvidia-ml-py3]")
        except Exception as e:
            logging.debug(f"nvidia-ml-py3 error: {e}")
            self._log_with_pytorch_basic(context)
    
    def _log_with_gputil(self, context: str):
        """Log using GPUtil."""
        try:
            gpus = self.gputil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                memory_used_gb = gpu.memoryUsed / 1024  # GPUtil returns MB
                memory_total_gb = gpu.memoryTotal / 1024
                logging.debug(f"\uD83D\uDD27 {context} | GPU: {gpu.load*100:3.0f}% | Memory: {gpu.memoryUtil*100:3.0f}% ({memory_used_gb:.1f}GB/{memory_total_gb:.1f}GB) [GPUtil]")
            else:
                self._log_with_pytorch_basic(context)
        except Exception as e:
            logging.debug(f"GPUtil error: {e}")
            self._log_with_pytorch_basic(context)
    
    def _log_with_nvidia_smi(self, context: str):
        """Log using nvidia-smi command line."""
        try:
            # Get GPU utilization and memory info
            cmd = ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                gpu_util = int(values[0])
                memory_used_mb = int(values[1])
                memory_total_mb = int(values[2])
                memory_used_gb = memory_used_mb / 1024
                memory_total_gb = memory_total_mb / 1024
                memory_util = (memory_used_mb / memory_total_mb) * 100
                logging.debug(f"\uD83D\uDD27 {context} | GPU: {gpu_util:3d}% | Memory: {memory_util:3.0f}% ({memory_used_gb:.1f}GB/{memory_total_gb:.1f}GB) [nvidia-smi]")
            else:
                self._log_with_pytorch_basic(context)
        except Exception as e:
            logging.debug(f"nvidia-smi error: {e}")
            self._log_with_pytorch_basic(context)
    
    def _log_with_psutil(self, context: str):
        """Log using psutil (CPU + basic system info)."""
        try:
            cpu_percent = self.psutil.cpu_percent(interval=0.1)
            memory = self.psutil.virtual_memory()
            memory_used_gb = memory.used / 1024**3
            memory_total_gb = memory.total / 1024**3
            logging.debug(f"\uD83D\uDD27 {context} | CPU: {cpu_percent:3.0f}% | RAM: {memory.percent:3.0f}% ({memory_used_gb:.1f}GB/{memory_total_gb:.1f}GB) [psutil]")
            # Also try to get GPU info via PyTorch
            self._log_pytorch_gpu_info(context, prefix="     ")
        except Exception as e:
            logging.debug(f"psutil error: {e}")
            self._log_with_pytorch_basic(context)
    
    def _log_with_pytorch_basic(self, context: str):
        """Fallback using only PyTorch and basic system info."""
        if config.USE_GPU_INFERENCE:
            self._log_pytorch_gpu_info(context)
        else:
            logging.debug(f"\uD83D\uDD27 {context} | GPU: Inference disabled in configuration")
    
    def _log_pytorch_gpu_info(self, context: str, prefix: str = "\uD83D\uDD27"):
        """Log GPU info using PyTorch."""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3   # GB
                
                # Try to get additional info from CuPy if available
                cupy_info = ""
                try:
                    if CUPY_AVAILABLE:
                        pool = cp.get_default_memory_pool()
                        cupy_used = pool.used_bytes() / 1024**3
                        cupy_info = f" | CuPy: {cupy_used:.1f}GB"
                except:
                    pass
                
                logging.debug(f"{prefix} {context} | GPU: {device_name} | PyTorch: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved{cupy_info}")
            else:
                logging.debug(f"{prefix} {context} | GPU: CUDA not available")
        except Exception as e:
            logging.debug(f"{prefix} {context} | GPU: Status unavailable ({str(e)})")
    
    def log_system_info(self):
        """Log comprehensive system information."""
        if not config.ENABLE_GPU_MONITORING:
            logging.debug("\uD83D\uDD27 GPU monitoring disabled in configuration")
            return
            
        logging.debug("\uD83D\uDDA5\uFE0F  System Information:")
        logging.debug(f"   \u2022 CPU Cores: {os.cpu_count()}")
        logging.debug(f"   \u2022 I/O Workers: {config.MAX_IO_WORKERS}")
        logging.debug(f"   \u2022 GPU Inference: {'Enabled' if config.USE_GPU_INFERENCE else 'Disabled'}")
        logging.debug(f"   \u2022 CuPy Ops: {'Enabled' if config.USE_CUPY else 'Disabled'}")
        logging.debug(f"   \u2022 GPU Preprocessing: {'Enabled' if config.USE_GPU_PREPROCESSING else 'Disabled'}")
        logging.debug(f"   \u2022 GPU Label Mapping: {'Enabled' if config.ENABLE_GPU_LABEL_MAPPING else 'Disabled'}")
        logging.debug(f"   \u2022 CuPy Available: {'Yes' if CUPY_AVAILABLE else 'No'}")
        logging.debug(f"   \u2022 Monitoring Method: {self.monitoring_method}")
        
        # Show available monitoring libraries
        available_libs = []
        try:
            import pynvml
            available_libs.append("pynvml")
        except ImportError:
            pass
        
        try:
            import GPUtil
            available_libs.append("GPUtil")
        except ImportError:
            pass
        
        try:
            import psutil
            available_libs.append("psutil")
        except ImportError:
            pass
        
        try:
            result = subprocess.run(['nvidia-smi', '--version'], capture_output=True, timeout=2)
            if result.returncode == 0:
                available_libs.append("nvidia-smi")
        except:
            pass
        
        if available_libs:
            logging.debug(f"   \u2022 Available Libraries: {', '.join(available_libs)}")
        else:
            logging.debug("   \u2022 Available Libraries: None (using PyTorch basic monitoring)")
        
        # Show installation suggestions if no advanced monitoring is available
        if self.monitoring_method in ["pytorch_basic", "none"]:
            self._log_installation_suggestions()
    
    def start_monitoring(self):
        """Start system monitoring (compatibility method)."""
        logging.debug("\uD83D\uDDA5\uFE0F System monitoring active")
        if hasattr(self, 'monitoring_method'):
            logging.debug(f"\uD83D\uDCCA Monitoring method: {self.monitoring_method}")
    
    def stop_monitoring(self):
        """Stop system monitoring (compatibility method)."""
        logging.debug("\uD83D\uDDA5\uFE0F System monitoring stopped")
        try:
            if self.monitoring_method == "pynvml" and hasattr(self, 'pynvml'):
                self.pynvml.nvmlShutdown()
        except Exception as e:
            logging.debug(f"Error stopping monitoring: {e}")
    
    def _log_installation_suggestions(self):
        """Log suggestions for installing monitoring libraries."""
        logging.debug("\uD83D\uDCA1 To enable advanced GPU/CPU monitoring, install one of:")
        logging.debug("   \u2022 pip install nvidia-ml-py3      # NVIDIA official Python bindings")
        logging.debug("   \u2022 pip install GPUtil             # Simple cross-platform GPU monitoring")
        logging.debug("   \u2022 pip install psutil             # System and process monitoring")
        logging.debug("   \u2022 Ensure nvidia-smi is in PATH   # Command-line GPU monitoring")


# ================================================================
# MEMORY MANAGEMENT
# ================================================================

class MemoryManager:
    """Efficient memory management for large-scale processing."""
    
    @staticmethod
    def flush_gpu_memory():
        """Clean up GPU memory from PyTorch and CuPy."""
        try:
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            if CUPY_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                cp.cuda.Device().synchronize()
        except Exception:
            pass
    
    @staticmethod
    def cleanup_variables(*variables):
        """Safely delete variables and trigger garbage collection."""
        for var in variables:
            if var is not None:
                del var
        gc.collect()
    
    def get_system_memory_usage(self):
        """Get current system memory usage percentage."""
        try:
            if PSUTIL_AVAILABLE:
                import psutil
                return psutil.virtual_memory().percent
            else:
                return 50.0  # Default if can't check
        except Exception:
            return 50.0  # Default if can't check
    
    def start_monitoring(self):
        """Start memory monitoring (compatibility method)."""
        logging.info("\uD83D\uDCBE Memory monitoring active")
    
    def stop_monitoring(self):
        """Stop memory monitoring (compatibility method)."""
        logging.info("\uD83D\uDCBE Memory monitoring stopped")
    
    def __init__(self, lock_dir=None):
        """Initialize MemoryManager with lock directory."""
        self.lock_dir = lock_dir or tempfile.gettempdir()
        self.label_mapping_lock_path = os.path.join(self.lock_dir, "adifind_label_mapping.lock")
    
    def acquire_label_mapping_lock(self, timeout=300):
        """Acquire exclusive lock for label mapping operations."""
        return _acquire_label_mapping_lock_handle(
            self.label_mapping_lock_path,
            self.get_system_memory_usage,
            timeout=timeout,
        )
    
    def release_label_mapping_lock(self, lock_file):
        """Release the label mapping lock with robust Windows-compatible error handling."""
        try:
            _release_label_mapping_lock_handle(lock_file, self.label_mapping_lock_path)
        except Exception as e:
            logging.warning(f"\u26A0\uFE0F Error during lock release: {e}")


# ================================================================
# CROSS-INSTANCE MEMORY MANAGEMENT
# ================================================================

class CrossInstanceMemoryManager:
    """
    Manages memory-intensive operations across multiple AdiFind instances.
    Prevents simultaneous memory-intensive operations that can exhaust system memory.
    """
    
    def __init__(self, lock_dir=None):
        self.lock_dir = lock_dir or tempfile.gettempdir()
        self.label_mapping_lock_path = os.path.join(self.lock_dir, "adifind_label_mapping.lock")
        self.processing_lock_path = os.path.join(self.lock_dir, "adifind_processing.lock")
        self.max_memory_percent = 90
        self.emergency_memory_percent = 95
        self.memory_check_interval = 2
        
        # OOM tracking system - initially use temp directory
        self.oom_log_path = os.path.join(self.lock_dir, "adifind_oom_tracking.csv")
        self.oom_summary_path = os.path.join(self.lock_dir, "adifind_oom_summary.txt")
        self.current_run_output_dir = None  # Will be set per run
        
        # Clean up any stale lock files on startup
        self.cleanup_stale_locks()
    
    def set_run_output_directory(self, output_dir):
        """
        Set the output directory for the current run's OOM tracking.
        This allows OOM files to be saved in the specific run's summary folder.
        """
        if output_dir:
            self.current_run_output_dir = output_dir
            # Update OOM file paths to use run-specific directory
            run_name = os.path.basename(output_dir)
            self.oom_log_path = os.path.join(self.lock_dir, f"adifind_oom_tracking_{run_name}.csv")
            self.oom_summary_path = os.path.join(output_dir, "oom_summary_report.txt")
        else:
            # Reset to default temp directory paths
            self.current_run_output_dir = None
            self.oom_log_path = os.path.join(self.lock_dir, "adifind_oom_tracking.csv")
            self.oom_summary_path = os.path.join(self.lock_dir, "adifind_oom_summary.txt")
        
    def cleanup_stale_locks(self):
        """Clean up any stale lock files from previous runs with Windows-compatible handling."""
        try:
            for lock_path in [self.label_mapping_lock_path, self.processing_lock_path]:
                if os.path.exists(lock_path):
                    # Use multiple attempts for Windows file system delays
                    max_attempts = 3 if os.name == 'nt' else 1
                    for attempt in range(max_attempts):
                        try:
                            os.remove(lock_path)
                            logging.debug(f"\uD83E\uDDF9 Cleaned up stale lock file: {os.path.basename(lock_path)}")
                            break
                        except (OSError, PermissionError) as e:
                            if attempt < max_attempts - 1:
                                # Brief wait for Windows file system
                                time.sleep(0.2)
                                continue
                            else:
                                # Final attempt failed - this is often normal on Windows
                                logging.debug(f"\uD83D\uDD12 Lock file busy (likely in use): {os.path.basename(lock_path)}")
        except Exception as e:
            logging.debug(f"Info: Minor cleanup issue (non-critical): {e}")
        
    def get_system_memory_usage(self):
        """Get current system memory usage percentage."""
        try:
            if PSUTIL_AVAILABLE:
                import psutil
                return psutil.virtual_memory().percent
            else:
                return 50  # Default if can't check
        except:
            return 50  # Default if can't check
    
    def wait_for_memory_availability(self, max_wait_time=60):
        """Wait until system memory usage is below threshold with timeout to prevent infinite stalling."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            memory_usage = self.get_system_memory_usage()
            
            # Only wait if memory is CRITICALLY high (above 92%)
            if memory_usage > 92:
                logging.info(f"\uD83D\uDEA6 High memory usage ({memory_usage:.1f}%) - brief wait...")
                time.sleep(self.memory_check_interval)
                continue
            else:
                return True
        
        logging.warning(f"\u26A0\uFE0F Memory usage still high after {max_wait_time}s timeout. Proceeding anyway.")
        return False
    
    def acquire_label_mapping_lock(self, timeout=300):
        """Acquire exclusive lock for label mapping operations with minimal blocking."""
        return _acquire_label_mapping_lock_handle(
            self.label_mapping_lock_path,
            self.get_system_memory_usage,
            timeout=timeout,
        )
    
    def release_label_mapping_lock(self, lock_file):
        """Release the label mapping lock with robust Windows-compatible error handling."""
        try:
            _release_label_mapping_lock_handle(lock_file, self.label_mapping_lock_path)
        except Exception as e:
            logging.warning(f"\u26A0\uFE0F Error during lock release: {e}")
    
    def log_oom_error(self, error_message, image_file=None, window_coords=None, memory_usage=None):
        """Log an Out of Memory (OOM) error with detailed context information."""
        try:
            import csv
            from datetime import datetime
            
            # Get current memory usage if not provided
            if memory_usage is None:
                memory_usage = self.get_system_memory_usage()
            
            # Prepare log entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pid = os.getpid()
            
            log_entry = {
                'timestamp': timestamp,
                'pid': pid,
                'error_type': 'OOM',
                'error_message': error_message,
                'image_file': image_file or 'Unknown',
                'window_x': window_coords[0] if window_coords else 'N/A',
                'window_y': window_coords[1] if window_coords else 'N/A',
                'memory_usage_percent': f"{memory_usage:.1f}",
                'system_info': f"PID-{pid}"
            }
            
            # Write to CSV (create header if file doesn't exist)
            file_exists = os.path.exists(self.oom_log_path)
            with open(self.oom_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_entry)
            
            # Log to console as well
            location_info = f" at ({window_coords[0]}, {window_coords[1]})" if window_coords else ""
            image_info = f" in {os.path.basename(image_file)}" if image_file else ""
            logging.error(f"\uD83D\uDEA8 OOM Error{image_info}{location_info}: {error_message} (Memory: {memory_usage:.1f}%)")
            
        except Exception as e:
            logging.warning(f"\u26A0\uFE0F Failed to log OOM error: {e}")
    
    def generate_oom_summary(self, output_dir=None):
        """Generate a summary of all OOM errors from the current session."""
        try:
            if not os.path.exists(self.oom_log_path):
                logging.info("\uD83D\uDCCA No OOM errors logged - summary not needed")
                return
            
            import csv
            from collections import defaultdict, Counter
            
            # Read all OOM logs
            oom_data = []
            with open(self.oom_log_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                oom_data = list(reader)
            
            if not oom_data:
                logging.info("\uD83D\uDCCA OOM log file empty - summary not needed")
                return
            
            # Analyze data
            total_errors = len(oom_data)
            error_by_pid = Counter(row['pid'] for row in oom_data)
            error_by_image = Counter(row['image_file'] for row in oom_data)
            
            # Memory usage statistics
            memory_values = [float(row['memory_usage_percent']) for row in oom_data if row['memory_usage_percent'] != 'N/A']
            avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0
            
            # Spatial distribution
            spatial_data = [(row['window_x'], row['window_y']) for row in oom_data if row['window_x'] != 'N/A']
            
            # Generate summary report
            summary_path = output_dir if output_dir else self.oom_summary_path
            if output_dir:
                summary_path = os.path.join(output_dir, "oom_summary_report.txt")
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("AdiFind WSI - Out of Memory (OOM) Analysis Report\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Analysis Period: {oom_data[0]['timestamp']} to {oom_data[-1]['timestamp']}\n\n")
                
                f.write("SUMMARY STATISTICS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total OOM Errors: {total_errors}\n")
                f.write(f"Average Memory Usage: {avg_memory:.1f}%\n")
                f.write(f"Affected Images: {len(error_by_image)}\n")
                f.write(f"Process Instances: {len(error_by_pid)}\n\n")
                
                f.write("ERROR DISTRIBUTION BY PROCESS:\n")
                f.write("-" * 35 + "\n")
                for pid, count in error_by_pid.most_common():
                    f.write(f"PID {pid}: {count} errors\n")
                
                f.write("\nERROR DISTRIBUTION BY IMAGE:\n")
                f.write("-" * 32 + "\n")
                for image, count in error_by_image.most_common():
                    f.write(f"{os.path.basename(image)}: {count} errors\n")
                
                if spatial_data:
                    f.write(f"\nSPATIAL DISTRIBUTION:\n")
                    f.write("-" * 25 + "\n")
                    f.write(f"Windows with errors: {len(spatial_data)}\n")
                    f.write("Sample error locations (X, Y):\n")
                    for i, (x, y) in enumerate(spatial_data[:10]):
                        f.write(f"  ({x}, {y})\n")
                    if len(spatial_data) > 10:
                        f.write(f"  ... and {len(spatial_data) - 10} more\n")
                
                f.write(f"\nDETAILED LOG INFORMATION:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Log File: {self.oom_log_path}\n")
                f.write(f"Log Entries: {total_errors}\n")
                f.write(f"Log Size: {os.path.getsize(self.oom_log_path) if os.path.exists(self.oom_log_path) else 0} bytes\n")
            
            logging.info(f"\uD83D\uDCCA OOM summary report generated: {summary_path}")
            logging.info(f"\uD83D\uDCCA Summary: {total_errors} OOM errors across {len(error_by_pid)} processes")
            
            # Clean up temporary log file
            try:
                if os.path.exists(self.oom_log_path) and output_dir:
                    os.remove(self.oom_log_path)
                    logging.debug("\uD83E\uDDF9 Temporary OOM log cleaned up")
            except Exception as e:
                logging.debug(f"Info: Could not clean up OOM log: {e}")
            
        except Exception as e:
            logging.warning(f"\u26A0\uFE0F Failed to generate OOM summary: {e}")
    
    def start_monitoring(self):
        """Start cross-instance monitoring (compatibility method)."""
        logging.info("\uD83D\uDD17 Cross-instance memory monitoring active")
    
    def stop_monitoring(self):
        """Stop cross-instance monitoring (compatibility method)."""
        logging.info("\uD83D\uDD17 Cross-instance memory monitoring stopped")


# ================================================================
# GLOBAL INSTANCES
# ================================================================

# Initialize global instances
monitor = SystemMonitor()
memory_manager = MemoryManager()
cross_instance_memory_manager = CrossInstanceMemoryManager()

### Test GPU monitoring immediately after initialization (if enabled)
# if config.ENABLE_GPU_MONITORING:
#     print("GPU MONITORING - This should appear in console")
#     monitor.log_gpu_status("SystemMonitor initialization test")
#     print("GPU MONITORING TEST COMPLETE")
# else:
#     print("\uD83D\uDD27 GPU MONITORING DISABLED")

# Test system info display
monitor.log_system_info()

# Export commonly used functions
flush_gpu_memory = memory_manager.flush_gpu_memory


# ================================================================
# SLIDE METADATA FUNCTIONS (from original BIBLE)
# ================================================================

def get_mpp(image_handler):
    """
    Extract microns per pixel (MPP) from slide metadata.
    
    Args:
        image_handler: ImageHandler object with slide access
        
    Returns:
        float: Microns per pixel value
    """
    try:
        from image_processing import get_mpp as _get_mpp

        return _get_mpp(image_handler)
    except Exception as e:
        logging.warning(f"\u26A0\uFE0F  Error extracting MPP: {e}. Using default: {config.DEFAULT_MPP}")
        return config.DEFAULT_MPP


def setup_gpu_device(gpu_id):
    """
    Setup GPU device for processing.
    
    Args:
        gpu_id: GPU device ID to use
    """
    import torch
    
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device_name = torch.cuda.get_device_name(gpu_id)
        logging.info("\uD83D\uDDA5\uFE0F Using GPU %d: %s", gpu_id, device_name)
    else:
        logging.warning("\u26A0\uFE0F CUDA not available, using CPU")


def get_gpu_runtime_summary(gpu_id: Optional[int] = None) -> Dict[str, Any]:
    """Return a small summary of the effective GPU runtime stack."""
    summary: Dict[str, Any] = {
        'gpu_id': int(gpu_id) if gpu_id is not None else 0,
        'device_name': 'CPU / CUDA unavailable',
        'torch_available': TORCH_AVAILABLE,
        'cuda_available': bool(TORCH_AVAILABLE and torch.cuda.is_available()),
        'cupy_available': CUPY_AVAILABLE,
        'gpu_inference': bool(getattr(config, 'USE_GPU_INFERENCE', False)),
        'cupy_ops': bool(getattr(config, 'USE_CUPY', False)),
        'gpu_preprocessing': bool(getattr(config, 'USE_GPU_PREPROCESSING', False)),
        'gpu_label_mapping': bool(getattr(config, 'ENABLE_GPU_LABEL_MAPPING', False)),
    }

    if not summary['cuda_available']:
        return summary

    try:
        device_count = torch.cuda.device_count()
        device_index = summary['gpu_id']
        if device_count <= 0:
            summary['device_name'] = 'CUDA reported available but no devices were enumerated'
            return summary
        if device_index < 0 or device_index >= device_count:
            device_index = torch.cuda.current_device()
            summary['gpu_id'] = int(device_index)
        summary['device_name'] = torch.cuda.get_device_name(device_index)
    except Exception as exc:
        summary['device_name'] = f'CUDA device lookup failed: {exc}'

    return summary


def format_gpu_runtime_summary(summary: Dict[str, Any]) -> str:
    """Format a GPU runtime summary into a compact diagnostics line."""
    return (
        "inference={inference} | cupy_ops={cupy_ops} | gpu_preprocessing={gpu_preprocessing} | "
        "gpu_label_mapping={gpu_label_mapping} | gpu_id={gpu_id} | device={device_name} | "
        "torch_available={torch_available} | cuda_available={cuda_available} | cupy_available={cupy_available}"
    ).format(
        inference='on' if summary.get('gpu_inference') else 'off',
        cupy_ops='on' if summary.get('cupy_ops') else 'off',
        gpu_preprocessing='on' if summary.get('gpu_preprocessing') else 'off',
        gpu_label_mapping='on' if summary.get('gpu_label_mapping') else 'off',
        gpu_id=summary.get('gpu_id'),
        device_name=summary.get('device_name'),
        torch_available='yes' if summary.get('torch_available') else 'no',
        cuda_available='yes' if summary.get('cuda_available') else 'no',
        cupy_available='yes' if summary.get('cupy_available') else 'no',
    )


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'SystemMonitor',
    'MemoryManager', 
    'CrossInstanceMemoryManager',
    'monitor',
    'memory_manager',
    'cross_instance_memory_manager',
    'flush_gpu_memory',
    'format_gpu_runtime_summary',
    'get_gpu_runtime_summary',
    'get_mpp',
    'setup_gpu_device'
]
