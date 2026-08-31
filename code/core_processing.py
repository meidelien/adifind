#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Processing Module
======================

Main window processing and inference logic for AdiFind WSI analysis.
Contains the core parallel processing pipeline and union-find operations.
"""

import os
import gc
import csv
import time
import logging
import threading
import tempfile
import concurrent.futures
import numpy as np
import cv2
import torch
from torchvision.ops import nms
from skimage.measure import regionprops_table
from tqdm import tqdm
from collections import defaultdict, deque

try:
    import psutil
except ImportError:
    psutil = None

# Import from other modules
from config import config
from system_utils import monitor, memory_manager, flush_gpu_memory
from image_processing import is_window_predominantly_black
from progress_utils import ColorChangingTqdm


# ================================================================
# MEMORY-MAPPED MASK UTILITIES
# ================================================================

def create_mask_array(height, width, output_dir=None):
    """
    Create a mask array, using memory-mapping if low-memory mode is enabled.
    
    Args:
        height: Mask height in pixels
        width: Mask width in pixels
        output_dir: Output directory for temporary files (optional)
        
    Returns:
        Tuple of (mask_array, cleanup_function)
        - mask_array: numpy array (either in-memory or memory-mapped)
        - cleanup_function: Function to call when done (removes temp file for memmap)
    """
    use_memmap = getattr(config, 'USE_MEMMAP_MASK', False)
    
    if use_memmap:
        # Calculate expected size
        expected_size_gb = (height * width * 4) / (1024 ** 3)  # uint32 = 4 bytes
        logging.debug(f"\uD83D\uDCBE Creating memory-mapped mask: {height:,} \u00D7 {width:,} ({expected_size_gb:.1f}GB on disk)")
        
        # Create temp file for memmap
        temp_dir = output_dir if output_dir else tempfile.gettempdir()
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.memmap', 
            dir=temp_dir, 
            delete=False,
            prefix='adifind_mask_'
        )
        temp_path = temp_file.name
        temp_file.close()
        
        # Create memory-mapped array
        mask = np.memmap(temp_path, dtype=np.uint32, mode='w+', shape=(height, width))
        mask[:] = 0  # Initialize to zero
        mask_ref = mask
        
        def cleanup():
            """Clean up temporary memmap file."""
            nonlocal mask_ref
            if mask_ref is None:
                return
            try:
                # Flush and close the memmap before deleting the file
                try:
                    mask_ref.flush()
                except Exception:
                    pass
                try:
                    if hasattr(mask_ref, "_mmap") and mask_ref._mmap is not None:
                        mask_ref._mmap.close()
                except Exception:
                    pass

                mask_ref = None  # Release memmap reference
                gc.collect()

                if os.path.exists(temp_path):
                    # Windows can hold file handles briefly; retry a few times
                    for attempt in range(5):
                        try:
                            os.unlink(temp_path)
                            logging.debug(f"\uD83D\uDDD1\uFE0F Cleaned up memmap file: {temp_path}")
                            break
                        except PermissionError:
                            time.sleep(0.2)
                    else:
                        logging.warning(f"\u26A0\uFE0F Could not delete memmap file after retries: {temp_path}")
            except Exception as e:
                logging.warning(f"\u26A0\uFE0F Could not clean up memmap file {temp_path}: {e}")
        
        logging.debug(f"   \u2022 Temp file: {temp_path}")
        return mask, cleanup
    else:
        # Standard in-memory array
        expected_size_gb = (height * width * 4) / (1024 ** 3)
        logging.debug(f"\uD83E\uDDE0 Creating in-memory mask: {height:,} \u00D7 {width:,} ({expected_size_gb:.1f}GB)")
        mask = np.zeros((height, width), dtype=np.uint32)
        return mask, lambda: None  # No cleanup needed

# Import GPU optimizations
try:
    from optimizations.gpu_label_mapping import apply_gpu_optimized_label_mapping, GPUUnionFind, GPULabelMapper
    from optimizations.gpu_acceleration import (
        GPUAcceleratedOperations, OptimizedWindowBatcher, AdvancedGPUProcessor,
        GPUMemoryProfiler, apply_gpu_morphological_operation, compute_gpu_mask_statistics,
        apply_gpu_connected_components, process_gpu_batch_arrays
    )
    GPU_LABEL_MAPPING_AVAILABLE = True
    GPU_ACCELERATION_AVAILABLE = True
    logging.debug("\u2705 GPU label mapping optimization loaded")
    logging.debug("\u2705 GPU acceleration extensions loaded")
except ImportError as e:
    GPU_LABEL_MAPPING_AVAILABLE = False
    GPU_ACCELERATION_AVAILABLE = False
    logging.debug(f"\uD83D\uDDA5\uFE0F  Using CPU processing (GPU optimizations not available: {e})")

# Import Async I/O optimizations
try:
    from optimizations.async_integration import (
        integrate_async_io, benchmark_async_performance, 
        patch_image_handler_with_async, restore_original_image_handler
    )
    ASYNC_IO_AVAILABLE = True
    logging.debug("\u2705 Async I/O optimizations loaded")
except ImportError as e:
    ASYNC_IO_AVAILABLE = False
    logging.debug(f"\uD83D\uDCC1 Using standard I/O (async optimizations not available: {e})")


LAST_EXECUTION_DIAGNOSTICS = {}


def _set_execution_diagnostics(diagnostics):
    """Store the most recent execution diagnostics for retrieval in main.py."""
    global LAST_EXECUTION_DIAGNOSTICS
    LAST_EXECUTION_DIAGNOSTICS = dict(diagnostics or {})


def get_last_execution_diagnostics():
    """Return a copy of the most recent execution diagnostics."""
    return dict(LAST_EXECUTION_DIAGNOSTICS)


def _capture_resource_snapshot():
    """Capture a lightweight CPU/RAM/GPU snapshot for diagnostics."""
    snapshot = {}
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            snapshot.update({
                'cpu_percent': psutil.cpu_percent(interval=None),
                'ram_percent': float(vm.percent),
                'ram_used_gb': vm.used / (1024 ** 3),
                'ram_total_gb': vm.total / (1024 ** 3),
                'ram_available_gb': vm.available / (1024 ** 3),
            })
        except Exception:
            pass

    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_bytes = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            snapshot.update({
                'gpu_allocated_gb': allocated / (1024 ** 3),
                'gpu_reserved_gb': reserved / (1024 ** 3),
                'gpu_total_gb': total_bytes / (1024 ** 3),
                'gpu_percent': (reserved / total_bytes * 100.0) if total_bytes else 0.0,
            })
    except Exception:
        pass

    return snapshot


def _update_peak_usage(diagnostics, snapshot):
    """Track peak RAM/GPU usage observed during a stage."""
    if not diagnostics or not snapshot:
        return
    if 'ram_percent' in snapshot:
        diagnostics['peak_ram_percent'] = max(
            float(diagnostics.get('peak_ram_percent', 0.0)),
            float(snapshot['ram_percent'])
        )
    if 'ram_used_gb' in snapshot:
        diagnostics['peak_ram_used_gb'] = max(
            float(diagnostics.get('peak_ram_used_gb', 0.0)),
            float(snapshot['ram_used_gb'])
        )
    if 'gpu_reserved_gb' in snapshot:
        diagnostics['peak_gpu_reserved_gb'] = max(
            float(diagnostics.get('peak_gpu_reserved_gb', 0.0)),
            float(snapshot['gpu_reserved_gb'])
        )
    if 'gpu_percent' in snapshot:
        diagnostics['peak_gpu_percent'] = max(
            float(diagnostics.get('peak_gpu_percent', 0.0)),
            float(snapshot['gpu_percent'])
        )


def _format_resource_snapshot(snapshot):
    """Render a compact resource summary for file logging."""
    if not snapshot:
        return "resource snapshot unavailable"

    parts = []
    if 'cpu_percent' in snapshot:
        parts.append(f"CPU {snapshot['cpu_percent']:.0f}%")
    if 'ram_percent' in snapshot:
        parts.append(
            f"RAM {snapshot['ram_used_gb']:.1f}/{snapshot['ram_total_gb']:.1f}GB "
            f"({snapshot['ram_percent']:.1f}%, avail {snapshot['ram_available_gb']:.1f}GB)"
        )
    if 'gpu_total_gb' in snapshot:
        parts.append(
            f"GPU {snapshot['gpu_reserved_gb']:.1f}/{snapshot['gpu_total_gb']:.1f}GB reserved "
            f"({snapshot['gpu_allocated_gb']:.1f}GB alloc, {snapshot['gpu_percent']:.1f}%)"
        )
    return " | ".join(parts) if parts else "resource snapshot unavailable"


def log_resource_snapshot(stage, diagnostics=None):
    """Log a stage boundary resource snapshot and update diagnostic peaks."""
    snapshot = _capture_resource_snapshot()
    _update_peak_usage(diagnostics, snapshot)
    logging.info("RESOURCE SNAPSHOT [%s] %s", stage, _format_resource_snapshot(snapshot))
    return snapshot


def _estimate_mask_bytes(height, width):
    """Estimate uint32 full-mask size in bytes."""
    return int(height) * int(width) * 4


def _prepare_window_arrays(window):
    """Convert a PIL window into model-ready arrays, skipping predominantly black tiles."""
    if is_window_predominantly_black(window):
        return None, None

    raw_window_array = np.array(window)[:, :, :3]
    window_np = raw_window_array.copy()
    if config.APPLY_IMAGE_INVERSION:
        if config.USE_GPU_PREPROCESSING:
            from optimizations.gpu_acceleration import gpu_image_inversion
            try:
                window_np = gpu_image_inversion(window_np)
            except Exception as e:
                print(f"Warning: GPU inversion failed, using CPU fallback: {e}")
                window_np = 255 - window_np
        else:
            window_np = 255 - window_np

    if config.APPLY_SOBEL_FILTER:
        if config.USE_GPU_PREPROCESSING:
            from optimizations.gpu_acceleration import gpu_sobel_preprocessing
            try:
                window_array = gpu_sobel_preprocessing(
                    window_np,
                    apply_bilateral=config.APPLY_BILATERAL_FILTER,
                )
            except Exception as e:
                print(f"Warning: GPU Sobel failed, using CPU fallback: {e}")
                window_cv = cv2.cvtColor(window_np, cv2.COLOR_RGB2BGR)
                window_cv = window_cv.astype(np.float32)
                sobelx = cv2.Sobel(window_cv, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(window_cv, cv2.CV_64F, 0, 1, ksize=3)
                sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
                sobel_magnitude = np.uint8(sobel_magnitude)
                if config.APPLY_BILATERAL_FILTER:
                    window_cv = cv2.bilateralFilter(sobel_magnitude, 20, 20, 20)
                else:
                    window_cv = sobel_magnitude
                window_array = cv2.cvtColor(window_cv, cv2.COLOR_BGR2RGB)
        else:
            window_cv = cv2.cvtColor(window_np, cv2.COLOR_RGB2BGR)
            window_cv = window_cv.astype(np.float32)
            sobelx = cv2.Sobel(window_cv, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(window_cv, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            sobel_magnitude = np.uint8(sobel_magnitude)
            if config.APPLY_BILATERAL_FILTER:
                window_cv = cv2.bilateralFilter(sobel_magnitude, 20, 20, 20)
            else:
                window_cv = sobel_magnitude
            window_array = cv2.cvtColor(window_cv, cv2.COLOR_BGR2RGB)
    else:
        if config.APPLY_BILATERAL_FILTER:
            window_np = cv2.bilateralFilter(window_np, 20, 20, 20)
        window_array = window_np[:, :, :3]

    return window_array, raw_window_array


def _initialize_processing_runtime(image_handler, output_dir, allow_async_patch=True):
    """Initialize async I/O and optional GPU profiling for a processing run."""
    async_enabled = False
    original_image_handler = image_handler

    if allow_async_patch and ASYNC_IO_AVAILABLE and config.ENABLE_ASYNC_IO:
        try:
            image_handler = patch_image_handler_with_async(
                image_handler,
                config={'cache_size_mb': config.ASYNC_CACHE_SIZE_MB}
            )
            async_enabled = True
            logging.debug("\u2705 Async I/O enabled for image processing")
        except Exception as e:
            logging.warning(f"\u26A0\uFE0F  Async I/O initialization failed, using standard I/O: {e}")
            image_handler = original_image_handler

    gpu_profiler = None
    if GPU_ACCELERATION_AVAILABLE and (
        config.USE_CUPY or config.USE_GPU_PREPROCESSING or config.ENABLE_GPU_LABEL_MAPPING
    ):
        GPUAcceleratedOperations()
        AdvancedGPUProcessor(max_gpu_memory_gb=config.GPU_MEMORY_LIMIT_GB)
        if config.ENABLE_GPU_MEMORY_PROFILING:
            gpu_profiler = GPUMemoryProfiler()
            gpu_profiler.log_memory_usage("Processing Start")
        logging.debug("\u2705 GPU acceleration components initialized")

    return image_handler, async_enabled, gpu_profiler


def _cleanup_processing_runtime(image_handler, async_enabled):
    """Restore patched image handlers and emit async performance stats."""
    if async_enabled:
        try:
            if hasattr(image_handler, '_async_processor'):
                async_stats = image_handler._async_processor.get_performance_stats()
                logging.info(
                    "\U0001F4C8 Async I/O Performance: %s reads, %.1f%% async, %.3fs avg",
                    async_stats['total_reads'],
                    async_stats['async_ratio'] * 100.0,
                    async_stats['average_read_time'],
                )

            restore_original_image_handler(image_handler)
            logging.info("\uD83E\uDDF9 Async I/O cleanup completed")
        except Exception as e:
            logging.warning(f"\u26A0\uFE0F  Async I/O cleanup warning: {e}")


# ================================================================
# BATCHED INFERENCE MANAGER FOR GPU OPTIMIZATION
# ================================================================

class BatchedInferenceManager:
    """Manages batched window processing for improved GPU utilization."""
    
    def __init__(self, predictor, batch_size=None):
        self.predictor = predictor
        self.batch_size = batch_size or config.BATCH_INFERENCE_SIZE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_queue = deque()
        self.results_queue = deque()
        self._queue_lock = threading.Lock()      # protects window_queue (fast)
        self._inference_lock = threading.Lock()   # serializes GPU inference (slow)
        
        logging.debug(f"\uD83D\uDE80 BatchedInferenceManager initialized with batch size: {self.batch_size}")
    
    def add_window_for_inference(self, window_data):
        """
        Add window data to batch queue.
        
        Args:
            window_data: Dict containing {'window': np.array, 'coords': (x, y), 'args': inference_args}
        
        Returns:
            List of results if batch is full, None otherwise
        """
        batch_to_process = None
        with self._queue_lock:
            self.window_queue.append(window_data)
            if len(self.window_queue) >= self.batch_size:
                batch_to_process = list(self.window_queue)
                self.window_queue.clear()
        
        if batch_to_process is not None:
            with self._inference_lock:
                return self._process_batch_data(batch_to_process)
        return None
    
    def process_remaining_batch(self):
        """Process any remaining windows in the queue."""
        batch_to_process = None
        with self._queue_lock:
            if self.window_queue:
                batch_to_process = list(self.window_queue)
                self.window_queue.clear()
        if batch_to_process is not None:
            with self._inference_lock:
                return self._process_batch_data(batch_to_process)
        return []
    
    def _process_batch_data(self, batch_data):
        """Process a batch of windows (called with _inference_lock held)."""
        if not batch_data:
            return []
        
        try:
            # Prepare batch for inference
            windows_batch = []
            raw_windows_batch = []
            coords_batch = []
            args_batch = []
            
            for data in batch_data:
                window = data['window']
                window_array, raw_window_array = _prepare_window_arrays(window)
                if window_array is not None:
                    windows_batch.append(window_array)
                    raw_windows_batch.append(raw_window_array)
                    coords_batch.append(data['coords'])
                    args_batch.append(data['args'])
            
            if not windows_batch:
                return []
            
            # Batch inference
            if config.ENABLE_PROFILING:
                batch_start = time.time()
            
            outputs = self.predictor(windows_batch)
            
            if config.ENABLE_PROFILING:
                batch_time = time.time() - batch_start
                logging.info(f"   \u26A1 Batch inference ({len(windows_batch)} windows): {batch_time:.3f}s")
            
            # Process results
            results = []
            for i, (output, coords, args, window_array, raw_window_array) in enumerate(
                zip(outputs, coords_batch, args_batch, windows_batch, raw_windows_batch)
            ):
                result = {
                    'output': output,
                    'coords': coords,
                    'args': args,
                    'window_idx': i,
                    'window_array': window_array,
                    'raw_window_array': raw_window_array,
                }
                results.append(result)
            
            torch.cuda.empty_cache()
            return results
            
        except Exception as e:
            logging.error(f"Batch inference error: {e}")
            # Fallback to individual processing
            results = []
            for data in batch_data:
                try:
                    # Process individually using original inference_worker logic
                    individual_result = self._process_individual_window(data)
                    if individual_result:
                        results.append(individual_result)
                except Exception as individual_e:
                    logging.error(f"Individual fallback error: {individual_e}")
            
            return results
    
    def _process_individual_window(self, window_data):
        """Fallback method for individual window processing."""
        window = window_data['window']
        coords = window_data['coords']
        args = window_data['args']
        
        if is_window_predominantly_black(window):
            return None

        try:
            window_array, raw_window_array = _prepare_window_arrays(window)
            if window_array is None:
                return None
            
            # Single inference
            outputs = self.predictor([window_array])
            output = outputs[0]
            
            return {
                'output': output,
                'coords': coords,
                'args': args,
                'window_idx': 0,
                'window_array': window_array,
                'raw_window_array': raw_window_array,
            }
            
        except Exception as e:
            logging.error(f"Individual window processing error: {e}")
            return None


# ================================================================
# UNION-FIND FUNCTIONS FOR MERGING OVERLAPPING LABELS
# ================================================================

def find(parent, x):
    """Find operation for union-find data structure with path compression."""
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent, parent[x])  # Path compression
    return parent[x]


def union(parent, label_a, label_b, overlap_ratio):
    """Union operation for merging overlapping labels."""
    if overlap_ratio > config.MERGE_IOU_THRESHOLD:
        root_a = find(parent, label_a)
        root_b = find(parent, label_b)
        if root_a != root_b:
            parent[root_b] = root_a


# ================================================================
# MEMORY-EFFICIENT LABEL MAPPING FUNCTIONS
# ================================================================

def apply_label_mapping_memory_efficient(mask, mapping_dict):
    """
    Apply label mapping without creating massive intermediate arrays.
    Uses vectorized numpy operations for maximum efficiency.
    
    For gigapixel images, uses chunked processing to avoid RAM overflow.
    """
    if not mapping_dict:
        return mask
    
    try:
        # Check if we should use chunked mode based on lookup table size OR mask size
        total_pixels = mask.shape[0] * mask.shape[1]
        max_label = max(max(mapping_dict.keys()), max(mapping_dict.values()))
        lookup_size_gb = (max_label + 1) * 4 / (1024 ** 3)  # uint32 = 4 bytes
        
        use_chunked = (
            lookup_size_gb > 1.0 or  # Lookup table would exceed 1GB
            (getattr(config, 'MEMORY_EFFICIENT_MODE', False) and 
             total_pixels > getattr(config, 'MAX_FULL_MASK_PIXELS', 500_000_000))
        )
        
        if use_chunked:
            logging.info(f"\uD83D\uDD04 Using chunked label mapping (max_label={max_label:,}, lookup would need {lookup_size_gb:.1f}GB)")
            return apply_label_mapping_chunked_safe(mask, mapping_dict)
        
        # Create vectorized lookup array - REVOLUTIONARY OPTIMIZATION
        # Instead of 6000+ boolean operations, use single vectorized lookup
        lookup = np.arange(max_label + 1, dtype=mask.dtype)
        
        # Build vectorized mapping in one operation
        for old_label, new_label in mapping_dict.items():
            lookup[old_label] = new_label
        
        # Apply mapping in single vectorized operation
        return lookup[mask]
    
    except (MemoryError, ValueError) as e:
        logging.error(f"Memory error in label mapping, falling back to chunked: {e}")
        # Fall back to chunked processing on memory error
        return apply_label_mapping_chunked_safe(mask, mapping_dict)


def apply_label_mapping_chunked(mask, mapping_dict):
    """
    Apply label mapping using chunked row processing with optional parallelization.
    Processes mask in chunks to avoid creating full-size intermediate arrays.
    Modifies mask in-place when INPLACE_LABEL_REMAPPING is enabled.
    
    When PARALLEL_CHUNK_PROCESSING is enabled, uses ThreadPoolExecutor to process
    multiple chunks simultaneously. This is safe because:
    - The lookup table is read-only after construction
    - Each chunk writes to non-overlapping rows of the mask
    - NumPy releases the GIL during array operations
    
    Args:
        mask: Full-resolution label mask (np.ndarray)
        mapping_dict: Dict mapping old labels to new labels
        
    Returns:
        Remapped mask (same array if in-place, new array otherwise)
    """
    if not mapping_dict:
        return mask
    
    # Check if global lookup table would be too large
    max_label = max(max(mapping_dict.keys()), max(mapping_dict.values()))
    lookup_size_gb = (max_label + 1) * 4 / (1024 ** 3)
    
    if lookup_size_gb > 1.0:
        # Use safe version with local lookup tables
        return apply_label_mapping_chunked_safe(mask, mapping_dict)
    
    chunk_size = getattr(config, 'MASK_CHUNK_SIZE', 4096)
    inplace = getattr(config, 'INPLACE_LABEL_REMAPPING', True)
    parallel = getattr(config, 'PARALLEL_CHUNK_PROCESSING', True)
    num_workers = getattr(config, 'CHUNK_WORKERS', 4)
    height = mask.shape[0]
    
    # Build lookup table once (this is small - just max_label+1 elements)
    lookup = np.arange(max_label + 1, dtype=mask.dtype)
    for old_label, new_label in mapping_dict.items():
        lookup[old_label] = new_label
    
    # Calculate chunk boundaries
    chunk_ranges = [(start, min(start + chunk_size, height)) 
                    for start in range(0, height, chunk_size)]
    num_chunks = len(chunk_ranges)
    
    logging.info(f"\uD83D\uDD04 Chunked label mapping: {height} rows in {num_chunks} chunks of {chunk_size}")
    
    if parallel and num_chunks > 1:
        from concurrent.futures import ThreadPoolExecutor
        
        logging.info(f"\u26A1 Using parallel processing with {num_workers} workers")
        
        if inplace:
            # Define worker function for in-place processing
            def process_chunk_inplace(chunk_range):
                start_row, end_row = chunk_range
                mask[start_row:end_row, :] = lookup[mask[start_row:end_row, :]]
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(executor.map(process_chunk_inplace, chunk_ranges))
            return mask
        else:
            # Create output array and process chunks in parallel
            result = np.empty_like(mask)
            
            def process_chunk_copy(chunk_range):
                start_row, end_row = chunk_range
                result[start_row:end_row, :] = lookup[mask[start_row:end_row, :]]
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(executor.map(process_chunk_copy, chunk_ranges))
            return result
    else:
        # Sequential processing (original implementation)
        if inplace:
            for start_row, end_row in chunk_ranges:
                mask[start_row:end_row, :] = lookup[mask[start_row:end_row, :]]
            return mask
        else:
            result = np.empty_like(mask)
            for start_row, end_row in chunk_ranges:
                result[start_row:end_row, :] = lookup[mask[start_row:end_row, :]]
            return result


def apply_label_mapping_chunked_safe(mask, mapping_dict):
    """
    Memory-safe chunked label mapping that uses local lookup tables per chunk.
    
    When label IDs are extremely large (billions), creating a global lookup table
    would require tens of GB of RAM. This function instead processes each chunk
    with a local lookup table sized only to that chunk's max label.
    
    Args:
        mask: Full-resolution label mask (modified in-place)
        mapping_dict: Dict mapping old labels to new labels
        
    Returns:
        Modified mask
    """
    if not mapping_dict:
        return mask
    
    chunk_size = getattr(config, 'MASK_CHUNK_SIZE', 2048)
    height = mask.shape[0]
    
    logging.info(f"\uD83D\uDD04 Safe chunked label mapping: {height} rows in chunks of {chunk_size}")
    
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        chunk = mask[start_row:end_row, :]
        
        # Get unique labels in this chunk
        chunk_unique = np.unique(chunk)
        chunk_max = chunk_unique.max() if len(chunk_unique) > 0 else 0
        
        if chunk_max > 0:
            # Build local lookup table sized to this chunk's max
            local_lookup = np.arange(chunk_max + 1, dtype=mask.dtype)
            
            # Apply only the mappings relevant to this chunk
            for old_label in chunk_unique:
                if old_label in mapping_dict:
                    local_lookup[old_label] = mapping_dict[old_label]
            
            # Apply mapping to chunk
            mask[start_row:end_row, :] = local_lookup[chunk]
    
    return mask


def apply_label_mapping_inplace_chunked(mask, mapping_dict, chunk_size=None):
    """
    Apply label mapping in-place using chunked processing.
    This is the most memory-efficient method for gigapixel masks.
    
    Args:
        mask: Full-resolution label mask (modified in-place)
        mapping_dict: Dict mapping old labels to new labels
        chunk_size: Number of rows per chunk (default from config)
    """
    if not mapping_dict:
        return
    
    if chunk_size is None:
        chunk_size = getattr(config, 'MASK_CHUNK_SIZE', 4096)
    
    height = mask.shape[0]
    
    # Check if global lookup table would be too large
    max_label = max(max(mapping_dict.keys()), max(mapping_dict.values()))
    lookup_size_gb = (max_label + 1) * 4 / (1024 ** 3)
    
    if lookup_size_gb > 1.0:
        # Use local lookup tables per chunk
        logging.info(f"\uD83D\uDD04 Safe inplace chunked mapping (max_label={max_label:,}, would need {lookup_size_gb:.1f}GB)")
        for start_row in range(0, height, chunk_size):
            end_row = min(start_row + chunk_size, height)
            chunk = mask[start_row:end_row, :]
            chunk_unique = np.unique(chunk)
            chunk_max = chunk_unique.max() if len(chunk_unique) > 0 else 0
            if chunk_max > 0:
                local_lookup = np.arange(chunk_max + 1, dtype=mask.dtype)
                for old_label in chunk_unique:
                    if old_label in mapping_dict:
                        local_lookup[old_label] = mapping_dict[old_label]
                mask[start_row:end_row, :] = local_lookup[chunk]
        return
    
    # Build lookup table once (small enough to fit in memory)
    lookup = np.arange(max_label + 1, dtype=mask.dtype)
    for old_label, new_label in mapping_dict.items():
        lookup[old_label] = new_label
    
    # Process chunks in-place
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        chunk = mask[start_row:end_row, :]
        mask[start_row:end_row, :] = lookup[chunk]


def remove_invalid_labels_chunked(mask, valid_labels, chunk_size=None):
    """
    Remove invalid labels using chunked processing.
    Sets all labels not in valid_labels to 0.
    
    Args:
        mask: Full-resolution label mask (modified in-place)
        valid_labels: Array/list of valid label IDs to keep
        chunk_size: Number of rows per chunk
    """
    if len(valid_labels) == 0:
        mask[:] = 0
        return
    
    if chunk_size is None:
        chunk_size = getattr(config, 'MASK_CHUNK_SIZE', 4096)
    
    height = mask.shape[0]
    max_label = mask.max()
    
    # Build validity lookup: valid_lookup[label] = label if valid, else 0
    valid_lookup = np.zeros(max_label + 1, dtype=mask.dtype)
    for label in valid_labels:
        if label <= max_label:
            valid_lookup[label] = label
    
    # Process chunks in-place
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        mask[start_row:end_row, :] = valid_lookup[mask[start_row:end_row, :]]


def relabel_consecutive_chunked(mask, chunk_size=None):
    """
    Relabel mask to consecutive IDs using chunked processing.
    
    For gigapixel masks, this avoids np.unique() on the full array
    by using the already-tracked labels from union-find.
    
    Args:
        mask: Full-resolution label mask (modified in-place)
        chunk_size: Number of rows per chunk
        
    Returns:
        Tuple of (number of unique labels, list of new consecutive IDs)
    """
    if chunk_size is None:
        chunk_size = getattr(config, 'MASK_CHUNK_SIZE', 4096)
    
    height = mask.shape[0]
    
    # First pass: collect unique labels from chunks (memory efficient)
    unique_set = set()
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        chunk_unique = np.unique(mask[start_row:end_row, :])
        unique_set.update(chunk_unique.tolist())
    
    unique_set.discard(0)  # Remove background
    unique_labels = sorted(unique_set)
    
    if len(unique_labels) == 0:
        return 0, []
    
    # Build consecutive relabeling lookup
    max_label = max(unique_labels)
    relabel_lookup = np.zeros(max_label + 1, dtype=mask.dtype)
    for new_id, old_id in enumerate(unique_labels, start=1):
        relabel_lookup[old_id] = new_id
    
    # Second pass: apply relabeling in chunks
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        mask[start_row:end_row, :] = relabel_lookup[mask[start_row:end_row, :]]
    
    return len(unique_labels), list(range(1, len(unique_labels) + 1))


def remove_invalid_labels_memory_efficient(mask, valid_labels):
    """
    Remove invalid labels using memory-efficient chunked processing.
    Avoids creating massive boolean arrays from np.isin.
    """
    if len(valid_labels) == 0:
        return np.zeros_like(mask)
    
    valid_label_set = set(valid_labels)
    chunk_size = getattr(config, 'MASK_CHUNK_SIZE', 2048)
    height = mask.shape[0]
    
    # Process in chunks to avoid huge boolean arrays
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        chunk = mask[start_row:end_row, :]
        
        # Use local lookup table instead of np.isin
        chunk_unique = np.unique(chunk)
        chunk_max = chunk_unique.max() if len(chunk_unique) > 0 else 0
        
        if chunk_max > 0:
            # Build local lookup: valid labels keep their value, invalid become 0
            local_lookup = np.zeros(chunk_max + 1, dtype=mask.dtype)
            for label in chunk_unique:
                if label in valid_label_set:
                    local_lookup[label] = label
            mask[start_row:end_row, :] = local_lookup[chunk]
    
    return mask


def relabel_consecutive_memory_efficient(mask):
    """
    Relabel mask to have consecutive IDs starting from 1.
    Uses vectorized numpy operations for maximum efficiency.
    """
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background
    
    if len(unique_labels) == 0:
        return mask
    
    # Create vectorized lookup array - much more efficient than loops
    max_label = unique_labels.max()
    lookup = np.zeros(max_label + 1, dtype=mask.dtype)
    
    # Build consecutive mapping
    for new_id, old_id in enumerate(unique_labels, start=1):
        lookup[old_id] = new_id
    
    # Apply mapping in one vectorized operation
    return lookup[mask]


def _save_debug_window_image(window_rgb, output_dir, subdir, x, y):
    """Save an RGB debug window image in BGR format for OpenCV output."""
    debug_dir = os.path.join(output_dir, subdir)
    os.makedirs(debug_dir, exist_ok=True)
    filename = f"window_x{x}_y{y}.png"
    cv2.imwrite(os.path.join(debug_dir, filename), cv2.cvtColor(window_rgb, cv2.COLOR_RGB2BGR))


def _create_debug_detection_overlay(window_rgb, boxes_local, masks):
    """Create a debug overlay (boxes + mask fill) on an RGB window image."""
    pred_img = window_rgb.copy()
    for i in range(len(boxes_local)):
        box = boxes_local[i].detach().cpu().numpy().astype(int)
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if masks is not None:
            mask = masks[i].detach().cpu().numpy().astype(np.uint8) * 255
            colored_mask = np.zeros_like(pred_img)
            colored_mask[mask > 0] = [0, 0, 255]
            pred_img = cv2.addWeighted(pred_img, 1.0, colored_mask, 0.3, 0)
    return pred_img


# ================================================================
# MAIN INFERENCE WORKER
# ================================================================

def inference_worker_batched(args, batch_manager):
    """
    Enhanced inference worker that uses batched processing for improved GPU utilization.
    
    Args:
        args: Tuple containing window processing arguments (including adipocyte_properties)
        batch_manager: BatchedInferenceManager instance
    """
    (
        image_handler,
        x,
        y,
        window_width,
        window_height,
        min_area_threshold_pixels,
        max_area_threshold_pixels,
        adipocyte_counter,
        parent,
        full_mask,
        output_dir,
        lock,
        adipocyte_properties,  # For incremental property collection
    ) = args

    try:
        # Read window
        window = image_handler.read_region((x, y), 0, (window_width, window_height))
        window = window.convert("RGB")

        # Prepare window data for batching
        window_data = {
            'window': window,
            'coords': (x, y),
            'args': args
        }
        
        # Add to batch manager and get results if batch is complete
        batch_results = batch_manager.add_window_for_inference(window_data)
        
        if batch_results:
            # Process all results in the batch
            for result_data in batch_results:
                _process_inference_result(result_data, parent, full_mask, 
                                        adipocyte_counter, lock, output_dir,
                                        min_area_threshold_pixels, max_area_threshold_pixels,
                                        adipocyte_properties)

    except Exception as e:
        # Check for memory-related errors
        error_message = str(e)
        if any(keyword in error_message.lower() for keyword in ['memory', 'out of memory', 'oom', 'cannot allocate', 'cannot read raw tile']):
            # Log detailed OOM error information
            current_memory = memory_manager.get_system_memory_usage()
            image_file = getattr(image_handler, 'file_path', 'unknown')
            memory_manager.log_oom_error(
                error_message=error_message,
                image_file=image_file,
                window_coords=(x, y),
                memory_usage=current_memory
            )
        
        logging.error(f"Error processing window at ({x}, {y}): {e}")


def _process_inference_result(result_data, parent, full_mask, adipocyte_counter, lock, output_dir,
                            min_area_threshold_pixels, max_area_threshold_pixels,
                            adipocyte_properties=None):
    """Process individual inference result from batch.
    
    Args:
        result_data: Dict with inference output and window arrays:
                    {'output', 'coords', 'window_array', 'raw_window_array'}
        parent: Union-find parent dict
        full_mask: Full resolution label mask
        adipocyte_counter: Mutable counter [current_id]
        lock: Threading lock
        output_dir: Output directory
        min_area_threshold_pixels: Min area filter
        max_area_threshold_pixels: Max area filter
        adipocyte_properties: Optional dict to collect properties incrementally
                             {adipocyte_id: {'area': int, 'centroid_x': float, 'centroid_y': float, 
                                            'bbox': (min_row, min_col, max_row, max_col)}}
    """
    output = result_data['output']
    x, y = result_data['coords']
    window_array = result_data.get('window_array')
    raw_window_array = result_data.get('raw_window_array')
    instances = output["instances"]

    # Save base debug windows for all non-black windows that reached inference.
    if config.DEBUG_MODE and window_array is not None:
        _save_debug_window_image(window_array, output_dir, "post_pros", x, y)
        if config.DEBUG_SAVE_UNPROCESSED_WINDOWS and raw_window_array is not None:
            _save_debug_window_image(raw_window_array, output_dir, "unprocessed", x, y)

    if len(instances) == 0:
        return  # No detections

    # Process detections (same logic as original inference_worker)
    boxes = instances.pred_boxes.tensor
    scores = instances.scores
    masks = instances.pred_masks

    # Apply NMS
    keep_indices = nms(boxes, scores, iou_threshold=config.IOU_THRESHOLD)
    boxes = boxes[keep_indices]
    masks = masks[keep_indices]
    scores = scores[keep_indices]

    if len(keep_indices) == 0:
        return  # All detections discarded after NMS

    # Keep local window coordinates for debug overlays.
    boxes_local = boxes

    # Save detection overlays only for windows with detections.
    if config.DEBUG_MODE and window_array is not None:
        pred_img = _create_debug_detection_overlay(window_array, boxes_local, masks)
        _save_debug_window_image(pred_img, output_dir, "predictions", x, y)
        if config.DEBUG_SAVE_UNPROCESSED_WINDOWS and raw_window_array is not None:
            raw_pred_img = _create_debug_detection_overlay(raw_window_array, boxes_local, masks)
            _save_debug_window_image(raw_pred_img, output_dir, "unprocessed_predictions", x, y)

    # Process each mask
    num_masks = masks.shape[0]
    for idx_mask in range(num_masks):
        mask = masks[idx_mask].to(dtype=torch.uint8).cpu().numpy()
        mask_height, mask_width = mask.shape
        x_offset = x
        y_offset = y

        # Calculate area of the mask
        area = int(mask.sum())

        # Check area thresholds
        if area < min_area_threshold_pixels or area > max_area_threshold_pixels:
            continue  # Discard this detection

        with lock:
            adipocyte_id = adipocyte_counter[0]
            adipocyte_counter[0] += 1

        # Get the region of full_mask corresponding to the current window
        full_mask_region = full_mask[y_offset:y_offset+mask_height, x_offset:x_offset+mask_width]

        # Check for overlapping labels
        overlapping_labels = np.unique(full_mask_region[mask > 0])
        for label in overlapping_labels:
            if label == 0:
                continue
            existing_mask = (full_mask_region == label)
            intersection = np.logical_and(mask > 0, existing_mask).sum()
            union_area = np.logical_or(mask > 0, existing_mask).sum()
            overlap_ratio = intersection / union_area
            # Update the union-find structure
            with lock:
                union(parent, adipocyte_id, label, overlap_ratio)

        # Update full_mask_region with the new label
        full_mask_region[mask > 0] = adipocyte_id

        # Update the full_mask
        full_mask[y_offset:y_offset+mask_height, x_offset:x_offset+mask_width] = full_mask_region
        
        # Collect properties incrementally if enabled
        if adipocyte_properties is not None:
            # Calculate centroid from mask
            mask_coords = np.where(mask > 0)
            if len(mask_coords[0]) > 0:
                centroid_y = float(y_offset + np.mean(mask_coords[0]))
                centroid_x = float(x_offset + np.mean(mask_coords[1]))
                # Calculate bounding box in full image coordinates
                bbox = (
                    int(y_offset + mask_coords[0].min()),  # min_row
                    int(x_offset + mask_coords[1].min()),  # min_col
                    int(y_offset + mask_coords[0].max()),  # max_row
                    int(x_offset + mask_coords[1].max())   # max_col
                )
                
                with lock:
                    adipocyte_properties[adipocyte_id] = {
                        'area': area,
                        'centroid_x': centroid_x,
                        'centroid_y': centroid_y,
                        'bbox': bbox
                    }


def inference_worker(args, predictor):
    """
    Main worker function for parallel window processing and adipocyte detection.
    
    Args:
        args: Tuple containing (image_handler, x, y, window_width, window_height,
              min_area_threshold_pixels, max_area_threshold_pixels, adipocyte_counter,
              parent, full_mask, output_dir, lock)
        predictor: Detectron2 predictor for adipocyte detection
    """
    if len(args) == 12:
        (
            image_handler,
            x,
            y,
            window_width,
            window_height,
            min_area_threshold_pixels,
            max_area_threshold_pixels,
            adipocyte_counter,
            parent,
            full_mask,
            output_dir,
            lock,
        ) = args
        adipocyte_properties = None
    else:
        (
            image_handler,
            x,
            y,
            window_width,
            window_height,
            min_area_threshold_pixels,
            max_area_threshold_pixels,
            adipocyte_counter,
            parent,
            full_mask,
            output_dir,
            lock,
            adipocyte_properties,
        ) = args

    try:
        # Read and preprocess the window
        window = image_handler.read_region((x, y), 0, (window_width, window_height))
        window = window.convert("RGB")

        window_array, raw_window_array = _prepare_window_arrays(window)
        if window_array is None:
            return  # Skip this window

        # Perform inference
        outputs = predictor([window_array])
        output = outputs[0]
        _process_inference_result(
            {
                'output': output,
                'coords': (x, y),
                'window_array': window_array,
                'raw_window_array': raw_window_array,
            },
            parent,
            full_mask,
            adipocyte_counter,
            lock,
            output_dir,
            min_area_threshold_pixels,
            max_area_threshold_pixels,
            adipocyte_properties,
        )

        torch.cuda.empty_cache()
        
        # Periodic GPU cleanup to prevent slowdown over large batches
        try:
            from gpu_kernels import periodic_gpu_cleanup
            periodic_gpu_cleanup(window_idx=adipocyte_counter[0])
        except ImportError:
            pass
        
        # Log GPU usage periodically (every 5th window to avoid spam)
        window_hash = hash((x, y)) % 5
        if window_hash == 0:  # Log every ~5th window
            monitor.log_gpu_status(f"Processing window ({x}, {y})")

    except Exception as e:
        # Check for memory-related errors
        error_message = str(e)
        if any(keyword in error_message.lower() for keyword in ['memory', 'out of memory', 'oom', 'cannot allocate', 'cannot read raw tile']):
            # Log detailed OOM error information
            current_memory = memory_manager.get_system_memory_usage()
            image_file = getattr(image_handler, 'file_path', 'unknown')
            memory_manager.log_oom_error(
                error_message=error_message,
                image_file=image_file,
                window_coords=(x, y),
                memory_usage=current_memory
            )
        
        logging.error(f"Error processing window at ({x}, {y}): {e}")
        # Log GPU usage on error for debugging
        monitor.log_gpu_status(f"Window ({x}, {y}) error")


# ================================================================
# COMMON FINALIZATION PATH
# ================================================================

def _finalize_processed_mask(full_mask, parent, adipocyte_properties, min_area_threshold_pixels,
                             max_area_threshold_pixels, height, width, image_handler,
                             async_enabled, gpu_profiler, mask_cleanup,
                             postprocessed_windows, processed_window_coords,
                             diagnostics=None):
    """Apply merged-label finalization and build final adipocyte outputs."""
    finalization_start = time.time()
    total_pixels = height * width
    use_memory_efficient = (
        getattr(config, 'MEMORY_EFFICIENT_MODE', False) and
        total_pixels > getattr(config, 'MAX_FULL_MASK_PIXELS', 500_000_000)
    )

    if diagnostics is not None:
        diagnostics['mask_backend'] = 'memmap' if isinstance(full_mask, np.memmap) else 'memory'
        diagnostics['mask_bytes'] = _estimate_mask_bytes(height, width)
        diagnostics['memory_efficient_finalization'] = bool(use_memory_efficient)

    if use_memory_efficient:
        logging.info(f"\uD83E\uDDE0 Using memory-efficient post-processing for {total_pixels:,} pixel image")
        all_assigned_labels = set(parent.keys())
        if adipocyte_properties is not None:
            all_assigned_labels.update(adipocyte_properties.keys())
        for label in list(all_assigned_labels):
            root = find(parent, label)
            all_assigned_labels.add(root)
        unique_labels = np.array(sorted(all_assigned_labels), dtype=np.uint32)
        unique_labels = unique_labels[unique_labels != 0]
        logging.info(f"   \uD83D\uDCCA Found {len(unique_labels)} unique labels from union-find/properties")
    else:
        unique_labels = np.unique(full_mask)
        unique_labels = unique_labels[unique_labels != 0]

    label_mapping_start = time.time()
    if len(unique_labels) > 0:
        lock_file = None
        try:
            logging.info("\uD83D\uDEA6 Requesting label mapping lock (memory-intensive operation)...")
            lock_file = memory_manager.acquire_label_mapping_lock(timeout=600)

            if (GPU_LABEL_MAPPING_AVAILABLE and
                config.ENABLE_GPU_LABEL_MAPPING and
                len(unique_labels) > config.GPU_LABEL_MAPPING_THRESHOLD):
                logging.info("\uD83D\uDE80 Building label mapping with GPU acceleration...")
                gpu_union_find = GPUUnionFind()
                label_mapping_cc = gpu_union_find.gpu_build_label_mapping(unique_labels, parent)
            else:
                logging.info("\uD83D\uDDA5\uFE0F  Building label mapping on CPU...")
                label_mapping_cc = {}
                for label in ColorChangingTqdm(unique_labels, desc="\U0001F517 Building Label Mapping (SYNCHRONIZED)"):
                    root_label = find(parent, label)
                    label_mapping_cc[label] = root_label
                print()

            logging.info("\uD83D\uDD27 Applying label mapping to full mask...")
            use_gpu = (
                GPU_LABEL_MAPPING_AVAILABLE
                and getattr(config, 'ENABLE_GPU_LABEL_MAPPING', False)
                and not getattr(config, 'FORCE_CPU_LABEL_MAPPING', False)
            )

            should_use_gpu = False
            if use_gpu:
                try:
                    gpu_mapper = GPULabelMapper(max_gpu_memory_gb=getattr(config, 'GPU_MEMORY_LIMIT_GB', 24))
                    total_required, mask_bytes, lut_bytes = gpu_mapper.estimate_memory_requirements(full_mask.shape, label_mapping_cc)
                    cap_bytes = int(gpu_mapper.max_gpu_memory_bytes)

                    fits_budget = (
                        int(total_required) <= cap_bytes
                        and int(lut_bytes) <= int(cap_bytes * getattr(config, 'GPU_LUT_FRACTION_MAX', 0.5))
                    )

                    cpu_bw_gbps = float(getattr(config, 'CPU_MEM_BW_GBPS', 40))
                    pcie_gbps = float(getattr(config, 'PCIE_EFFECTIVE_GBPS', 28))
                    margin = float(getattr(config, 'GPU_SPEED_MARGIN', 0.9))

                    t_cpu = (2.0 * mask_bytes) / (cpu_bw_gbps * (1024 ** 3))
                    t_gpu = ((2.0 * mask_bytes) + lut_bytes) / (pcie_gbps * (1024 ** 3))

                    should_use_gpu = fits_budget and (t_gpu < t_cpu * margin)

                    if getattr(config, 'LABEL_MAPPING_DEBUG', False):
                        logging.info(
                            "Label mapping decision :: mask=%d MB LUT=%d MB total=%d MB cap=%d MB | t_cpu=%.3fs t_gpu=%.3fs | fits=%s choose_gpu=%s",
                            mask_bytes // (1024 ** 2),
                            lut_bytes // (1024 ** 2),
                            total_required // (1024 ** 2),
                            cap_bytes // (1024 ** 2),
                            t_cpu,
                            t_gpu,
                            fits_budget,
                            should_use_gpu,
                        )
                except Exception as e:
                    logging.warning(f"\u26A0\uFE0F  GPU decision check failed ({e}); defaulting to CPU")
                    should_use_gpu = False

            if should_use_gpu:
                logging.info("\uD83D\uDE80 Using GPU-accelerated label mapping")
                print("LABEL MAPPING PATH: GPU")
                full_mask = apply_gpu_optimized_label_mapping(full_mask, label_mapping_cc, getattr(config, 'ENABLE_PROFILING', False))
            else:
                logging.info("\uD83D\uDDA5\uFE0F  Using CPU label mapping")
                print("LABEL MAPPING PATH: CPU")
                full_mask = apply_label_mapping_memory_efficient(full_mask, label_mapping_cc)

            logging.info("\u2705 Label mapping applied successfully")
            del label_mapping_cc
        finally:
            if lock_file:
                memory_manager.release_label_mapping_lock(lock_file)

            if gpu_profiler is not None:
                gpu_profiler.log_memory_usage("After Label Mapping")
                peak_usage = gpu_profiler.get_peak_usage()
                if peak_usage:
                    logging.info(
                        "\uD83C\uDFD4\uFE0F  Peak GPU memory usage: %.1fGB (%.1f%%) during '%s'",
                        peak_usage['used_gb'],
                        peak_usage['usage_percent'],
                        peak_usage['operation'],
                    )
    else:
        logging.info("No labels found for label mapping; leaving mask unchanged")

    label_mapping_seconds = time.time() - label_mapping_start

    if use_memory_efficient and adipocyte_properties is not None:
        logging.info("\uD83E\uDDE0 Using incrementally collected properties (memory-efficient)")

        label_to_root = {}
        for label in adipocyte_properties.keys():
            label_to_root[label] = find(parent, label)

        merged_properties = {}
        for orig_label, props in adipocyte_properties.items():
            root = label_to_root[orig_label]
            if root not in merged_properties:
                merged_properties[root] = {
                    'area': 0,
                    'centroid_x_sum': 0.0,
                    'centroid_y_sum': 0.0,
                    'count': 0,
                    'bbox_min_row': float('inf'),
                    'bbox_min_col': float('inf'),
                    'bbox_max_row': 0,
                    'bbox_max_col': 0,
                }
            mp = merged_properties[root]
            mp['area'] += props['area']
            mp['centroid_x_sum'] += props['centroid_x'] * props['area']
            mp['centroid_y_sum'] += props['centroid_y'] * props['area']
            mp['count'] += 1
            if props['bbox'][0] < mp['bbox_min_row']:
                mp['bbox_min_row'] = props['bbox'][0]
            if props['bbox'][1] < mp['bbox_min_col']:
                mp['bbox_min_col'] = props['bbox'][1]
            if props['bbox'][2] > mp['bbox_max_row']:
                mp['bbox_max_row'] = props['bbox'][2]
            if props['bbox'][3] > mp['bbox_max_col']:
                mp['bbox_max_col'] = props['bbox'][3]

        valid_roots = []
        for root, mp in merged_properties.items():
            if min_area_threshold_pixels <= mp['area'] <= max_area_threshold_pixels:
                valid_roots.append(root)
                if mp['area'] > 0:
                    mp['centroid_x'] = mp['centroid_x_sum'] / mp['area']
                    mp['centroid_y'] = mp['centroid_y_sum'] / mp['area']
                mp['bbox'] = (
                    mp['bbox_min_row'],
                    mp['bbox_min_col'],
                    mp['bbox_max_row'],
                    mp['bbox_max_col'],
                )

        logging.info(f"   \uD83D\uDCCA {len(valid_roots)} adipocytes pass area thresholds after merging")

        full_remap = {}
        for orig_label in adipocyte_properties.keys():
            root = label_to_root[orig_label]
            full_remap[orig_label] = root if root in valid_roots else 0

        logging.info("\uD83D\uDD27 Applying area filter to mask (chunked)...")
        apply_label_mapping_inplace_chunked(full_mask, full_remap)

        valid_roots_sorted = sorted(valid_roots)
        relabel_map = {0: 0}
        for new_id, old_root in enumerate(valid_roots_sorted, start=1):
            relabel_map[old_root] = new_id

        logging.info("\uD83D\uDD27 Relabeling to consecutive IDs (chunked)...")
        apply_label_mapping_inplace_chunked(full_mask, relabel_map)

        mask_areas = {}
        adipocyte_ids = []
        final_properties = {}

        for new_id, old_root in enumerate(valid_roots_sorted, start=1):
            mp = merged_properties[old_root]
            mask_areas[new_id] = mp['area']
            adipocyte_ids.append(new_id)
            final_properties[new_id] = {
                'area': mp['area'],
                'centroid_x': mp['centroid_x'],
                'centroid_y': mp['centroid_y'],
                'bbox': mp['bbox'],
            }

        logging.info(f"\u2705 Relabeled adipocytes: {len(adipocyte_ids)} adipocytes with consecutive IDs 1-{len(adipocyte_ids)}")
    else:
        props_table = regionprops_table(full_mask, properties=['label', 'area'])
        valid_mask = (
            (props_table['area'] >= min_area_threshold_pixels)
            & (props_table['area'] <= max_area_threshold_pixels)
        )
        valid_labels = props_table['label'][valid_mask]

        max_label_val = full_mask.max()
        mapping = np.zeros(max_label_val + 1, dtype=np.uint32)
        mapping[valid_labels] = valid_labels
        full_mask[:] = mapping[full_mask]

        unique_final_labels = np.unique(full_mask)
        unique_final_labels = unique_final_labels[unique_final_labels != 0]

        if len(unique_final_labels) > 0:
            max_label = full_mask.max()
            lookup_size_gb = (max_label + 1) * 4 / (1024 ** 3)
            relabel_dict = {old_id: new_id for new_id, old_id in enumerate(unique_final_labels, start=1)}

            if lookup_size_gb > 1.0:
                logging.info(f"\uD83D\uDD04 Using chunked relabeling (max_label={max_label:,}, would need {lookup_size_gb:.1f}GB lookup)")
                chunk_size = getattr(config, 'MASK_CHUNK_SIZE', 2048)
                for start_row in range(0, full_mask.shape[0], chunk_size):
                    end_row = min(start_row + chunk_size, full_mask.shape[0])
                    chunk = full_mask[start_row:end_row, :]
                    chunk_unique = np.unique(chunk)
                    chunk_max = chunk_unique.max() if len(chunk_unique) > 0 else 0
                    if chunk_max > 0:
                        local_lookup = np.zeros(chunk_max + 1, dtype=np.uint32)
                        for old_id in chunk_unique:
                            if old_id in relabel_dict:
                                local_lookup[old_id] = relabel_dict[old_id]
                        full_mask[start_row:end_row, :] = local_lookup[chunk]
            else:
                relabel_mapping = np.zeros(max_label + 1, dtype=np.uint32)
                for old_id, new_id in relabel_dict.items():
                    relabel_mapping[old_id] = new_id
                full_mask = relabel_mapping[full_mask]

            props_table = regionprops_table(full_mask, properties=['label', 'area', 'centroid', 'bbox'])
            logging.info(f"\u2705 Relabeled adipocytes: {len(unique_final_labels)} adipocytes now have consecutive IDs 1-{len(unique_final_labels)}")
        else:
            props_table = {
                'label': [],
                'area': [],
                'centroid-0': [],
                'centroid-1': [],
                'bbox-0': [],
                'bbox-1': [],
                'bbox-2': [],
                'bbox-3': [],
            }

        mask_areas = {}
        adipocyte_ids = []
        final_properties = {}

        for i, label in enumerate(props_table['label']):
            area = props_table['area'][i]
            mask_areas[label] = area
            adipocyte_ids.append(label)
            final_properties[label] = {
                'area': area,
                'centroid_x': props_table['centroid-1'][i],
                'centroid_y': props_table['centroid-0'][i],
                'bbox': (
                    props_table['bbox-0'][i],
                    props_table['bbox-1'][i],
                    props_table['bbox-2'][i],
                    props_table['bbox-3'][i],
                )
            }

        logging.info(f"\uD83D\uDCCA Built final_properties for {len(final_properties)} adipocytes")

    finalization_seconds = time.time() - finalization_start
    logging.info(f"\uD83D\uDCCA Processing complete: Initial detections: {len(unique_labels)}, Final adipocytes: {len(adipocyte_ids)}")
    log_resource_snapshot("after_finalization", diagnostics)

    if diagnostics is not None:
        diagnostics['initial_unique_labels'] = int(len(unique_labels))
        diagnostics['final_adipocytes'] = int(len(adipocyte_ids))
        diagnostics['label_mapping_seconds'] = label_mapping_seconds
        diagnostics['finalization_seconds'] = finalization_seconds

    _cleanup_processing_runtime(image_handler, async_enabled)

    logging.info("Finished processing all windows.")
    return full_mask, mask_areas, adipocyte_ids, postprocessed_windows, processed_window_coords, final_properties, mask_cleanup


# ================================================================
# MAIN WINDOW PROCESSING FUNCTION
# ================================================================

def process_all_windows(image_handler, predictor, window_size, stride, min_area_threshold_pixels, max_area_threshold_pixels, output_dir, window_coords=None):
    """
    Process all windows in an image using optimized batched inference workers.
    
    Args:
        image_handler: ImageHandler object for slide access
        predictor: Configured Detectron2 predictor for adipocyte detection
        window_size: (width, height) tuple for processing windows
        stride: (stride_x, stride_y) tuple for window overlap
        min_area_threshold_pixels: Minimum adipocyte area in pixels
        max_area_threshold_pixels: Maximum adipocyte area in pixels
        output_dir: Directory for saving results
        window_coords: Optional list of (x, y) window coordinates to process
        
    Returns:
        tuple: (full_mask, mask_areas, adipocyte_ids, postprocessed_windows, window_coords)
    """
    from image_processing import generate_sliding_windows
    
    logging.debug("\uD83D\uDE80 Processing all windows with batched inference...")
    width, height = image_handler.width, image_handler.height
    diagnostics = {
        'execution_path': 'generic_batched',
        'window_mode': 'full_image' if window_coords is None else 'selected_windows_generic',
        'requested_windows': 0,
        'batch_size': int(config.BATCH_INFERENCE_SIZE),
        'max_workers': int(config.MAX_IO_WORKERS),
        'async_io_enabled': bool(ASYNC_IO_AVAILABLE and config.ENABLE_ASYNC_IO),
        'window_execution_seconds': 0.0,
    }

    image_handler, async_enabled, gpu_profiler = _initialize_processing_runtime(image_handler, output_dir)

    # Prepare window coordinates
    if window_coords is None:
        window_coords = list(generate_sliding_windows(width, height, window_size, stride))
    else:
        window_coords = list(window_coords)
    diagnostics['requested_windows'] = int(len(window_coords))

    # Prepare window arguments
    window_args = [(image_handler, x, y, window_size[0], window_size[1])
                   for x, y in window_coords]
    total_windows = len(window_args)

    # Initialize full_mask (memory-mapped if low-memory mode enabled)
    log_resource_snapshot("before_mask_allocation", diagnostics)
    full_mask, mask_cleanup = create_mask_array(height, width, output_dir)
    diagnostics['mask_backend'] = 'memmap' if isinstance(full_mask, np.memmap) else 'memory'
    diagnostics['mask_bytes'] = _estimate_mask_bytes(height, width)
    log_resource_snapshot("after_mask_allocation", diagnostics)

    # Initialize union-find parent dictionary
    parent = {}
    # Adipocyte ID counter
    adipocyte_counter = [1]  # Use list to make it mutable in worker functions

    # Lock for thread-safe operations
    lock = threading.Lock()

    # --- Collect post-processed windows and their coordinates for stitching ---
    # Disabled: keeping these empty avoids holding every tile in RAM (unused downstream).
    postprocessed_windows = []
    window_coords = []
    processing_counter = [0]    # Shared counter for GPU monitoring
    
    # Initialize incremental property collection if enabled
    # This avoids multiple regionprops() scans on the gigapixel mask
    adipocyte_properties = {} if getattr(config, 'INCREMENTAL_PROPERTY_COLLECTION', True) else None

    # Initialize batched inference manager
    batch_manager = BatchedInferenceManager(predictor, config.BATCH_INFERENCE_SIZE)
    
    def inference_worker_collect_batched(args):
        # Unpack args
        (
            image_handler,
            x,
            y,
            w,
            h,
            min_area_threshold_pixels,
            max_area_threshold_pixels,
            adipocyte_counter,
            parent,
            full_mask,
            output_dir,
            lock,
            postprocessed_windows,
            window_coords,
            adipocyte_properties,  # Added for incremental property collection
        ) = args
        try:
            # Monitor GPU every 20 windows processed
            with lock:
                processing_counter[0] += 1
                if processing_counter[0] % 20 == 0:
                    monitor.log_gpu_status(f"Processed {processing_counter[0]} windows")

            # Use batched inference worker
            original_args = (
                image_handler,
                x,
                y,
                w,
                h,
                min_area_threshold_pixels,
                max_area_threshold_pixels,
                adipocyte_counter,
                parent,
                full_mask,
                output_dir,
                lock,
                adipocyte_properties,  # Pass through for property collection
            )
            inference_worker_batched(original_args, batch_manager)
            
        except Exception as e:
            # Check for memory-related errors
            error_message = str(e)
            if any(keyword in error_message.lower() for keyword in ['memory', 'out of memory', 'oom', 'cannot allocate', 'cannot read raw tile']):
                # Log detailed OOM error information
                current_memory = memory_manager.get_system_memory_usage()
                image_file = getattr(image_handler, 'file_path', 'unknown')
                memory_manager.log_oom_error(
                    error_message=error_message,
                    image_file=image_file,
                    window_coords=(x, y),
                    memory_usage=current_memory
                )
            
            logging.error(f"Error in inference_worker_collect_batched at ({x}, {y}): {e}")

    # Prepare arguments for worker function
    worker_args = []
    for args in window_args:
        _, x, y, w, h = args
        worker_args.append((
            image_handler,
            x,
            y,
            w,
            h,
            min_area_threshold_pixels,
            max_area_threshold_pixels,
            adipocyte_counter,
            parent,
            full_mask,
            output_dir,
            lock,
            postprocessed_windows,
            window_coords,
            adipocyte_properties,  # Include for incremental property collection
        ))

    # Use ThreadPoolExecutor to process windows in parallel
    max_workers = config.MAX_IO_WORKERS
    monitor.log_gpu_status("Starting batched window processing")
    processing_start = time.time()
    
    if config.ENABLE_PROFILING:
        logging.info(f"\uD83D\uDE80 Starting batched processing with {max_workers} workers, batch size {config.BATCH_INFERENCE_SIZE}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(ColorChangingTqdm(executor.map(inference_worker_collect_batched, worker_args), total=total_windows, desc="\U0001F9EC Processing Windows (Batched)"))
    print()  # Add line break after progress bar
    
    # Process any remaining windows in the batch
    remaining_results = batch_manager.process_remaining_batch()
    if remaining_results:
        for result_data in remaining_results:
            _process_inference_result(result_data, parent, full_mask, 
                                    adipocyte_counter, lock, output_dir,
                                    min_area_threshold_pixels, max_area_threshold_pixels,
                                    adipocyte_properties)
        if config.ENABLE_PROFILING:
            logging.info(f"   \uD83D\uDD04 Processed {len(remaining_results)} remaining windows from final batch")
    
    if config.ENABLE_PROFILING:
        processing_time = time.time() - processing_start
        logging.info(f"\u26A1 Batched processing completed: {processing_time:.3f}s")
        logging.info(f"   \uD83D\uDCCA Average time per window: {processing_time/total_windows:.4f}s")
        diagnostics['window_execution_seconds'] = processing_time
    else:
        diagnostics['window_execution_seconds'] = time.time() - processing_start if total_windows > 0 else 0.0
    diagnostics['mean_windows_per_second'] = (
        total_windows / diagnostics['window_execution_seconds']
        if diagnostics['window_execution_seconds'] > 0 else 0.0
    )
    
    monitor.log_gpu_status("Completed batched window processing")
    log_resource_snapshot("after_window_execution", diagnostics)
    
    # Log GPU memory usage after window processing
    if gpu_profiler is not None:
        gpu_profiler.log_memory_usage("After Window Processing")
    diagnostics['window_count_processed'] = int(total_windows)
    diagnostics['window_count_ready_for_inference'] = int(total_windows)
    results = _finalize_processed_mask(
        full_mask=full_mask,
        parent=parent,
        adipocyte_properties=adipocyte_properties,
        min_area_threshold_pixels=min_area_threshold_pixels,
        max_area_threshold_pixels=max_area_threshold_pixels,
        height=height,
        width=width,
        image_handler=image_handler,
        async_enabled=async_enabled,
        gpu_profiler=gpu_profiler,
        mask_cleanup=mask_cleanup,
        postprocessed_windows=postprocessed_windows,
        processed_window_coords=window_coords,
        diagnostics=diagnostics,
    )
    _set_execution_diagnostics(diagnostics)
    return results


def _prepare_selected_window_payload(args):
    """Read and preprocess a selected window for the pipelined executor."""
    image_handler, x, y, window_width, window_height = args
    prep_start = time.time()
    try:
        window = image_handler.read_region((x, y), 0, (window_width, window_height))
        window = window.convert("RGB")
        window_array, raw_window_array = _prepare_window_arrays(window)
        if window_array is None:
            return {
                'coords': (x, y),
                'skipped': True,
                'prep_time': time.time() - prep_start,
            }
        return {
            'coords': (x, y),
            'window_array': window_array,
            'raw_window_array': raw_window_array,
            'prep_time': time.time() - prep_start,
            'skipped': False,
        }
    except Exception as e:
        error_message = str(e)
        if any(keyword in error_message.lower() for keyword in ['memory', 'out of memory', 'oom', 'cannot allocate', 'cannot read raw tile']):
            current_memory = memory_manager.get_system_memory_usage()
            image_file = getattr(image_handler, 'file_path', 'unknown')
            memory_manager.log_oom_error(
                error_message=error_message,
                image_file=image_file,
                window_coords=(x, y),
                memory_usage=current_memory,
            )
        logging.error(f"Error preparing window at ({x}, {y}): {e}")
        return None


def process_selected_windows_threaded(image_handler, predictor, window_size, stride,
                                      min_area_threshold_pixels, max_area_threshold_pixels,
                                      output_dir, window_coords):
    """
    Process pre-filtered windows using direct threaded workers.
    Each worker performs read, preprocess, inference, and result application locally.
    """
    width, height = image_handler.width, image_handler.height
    selected_window_coords = list(window_coords or [])
    total_windows = len(selected_window_coords)
    max_workers = max(1, int(config.MAX_IO_WORKERS))
    sample_interval_windows = max(1, int(getattr(config, 'SELECTED_WINDOW_DIAGNOSTIC_INTERVAL', 100)))

    diagnostics = {
        'execution_path': 'selected_window_threaded',
        'window_mode': 'selected_windows_threaded',
        'requested_windows': int(total_windows),
        'max_workers': max_workers,
        'async_io_enabled': False,
        'window_execution_seconds': 0.0,
        'window_sample_interval': sample_interval_windows,
    }

    logging.info(
        "Selected-window threaded executor: %d windows | workers=%d | async_io=%s",
        total_windows,
        max_workers,
        False,
    )

    _, async_enabled, gpu_profiler = _initialize_processing_runtime(
        image_handler,
        output_dir,
        allow_async_patch=False,
    )

    log_resource_snapshot("before_mask_allocation", diagnostics)
    full_mask, mask_cleanup = create_mask_array(height, width, output_dir)
    diagnostics['mask_backend'] = 'memmap' if isinstance(full_mask, np.memmap) else 'memory'
    diagnostics['mask_bytes'] = _estimate_mask_bytes(height, width)
    log_resource_snapshot("after_mask_allocation", diagnostics)

    parent = {}
    adipocyte_counter = [1]
    lock = threading.Lock()
    stats_lock = threading.Lock()
    postprocessed_windows = []
    processed_window_coords = []
    adipocyte_properties = {} if getattr(config, 'INCREMENTAL_PROPERTY_COLLECTION', True) else None

    processing_start = time.time()
    processed_counter = [0]
    skipped_windows = [0]

    def log_threaded_sample(completed_windows):
        elapsed = time.time() - processing_start
        windows_per_second = (completed_windows / elapsed) if elapsed > 0 else 0.0
        adipocyte_total = max(0, adipocyte_counter[0] - 1)
        logging.info(
            "THREADED SAMPLE windows=%d/%d windows_per_sec=%.2f adipocytes=%d workers=%d",
            completed_windows,
            total_windows,
            windows_per_second,
            adipocyte_total,
            max_workers,
        )
        log_resource_snapshot(f"threaded_sample_{completed_windows}", diagnostics)
        monitor.log_gpu_status(f"Threaded selected windows={completed_windows}")

    def inference_worker_collect_threaded(worker_args):
        inference_worker(worker_args, predictor)
        with stats_lock:
            processed_counter[0] += 1
            completed_windows = processed_counter[0]
            if completed_windows == 1 or completed_windows % sample_interval_windows == 0:
                log_threaded_sample(completed_windows)

    worker_args = []
    for x, y in selected_window_coords:
        worker_args.append((
            image_handler,
            x,
            y,
            window_size[0],
            window_size[1],
            min_area_threshold_pixels,
            max_area_threshold_pixels,
            adipocyte_counter,
            parent,
            full_mask,
            output_dir,
            lock,
            adipocyte_properties,
        ))

    progress_bar = ColorChangingTqdm(total=total_windows, desc="\U0001F9EC Processing Windows (Threaded)")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(inference_worker_collect_threaded, args) for args in worker_args]
            for _ in concurrent.futures.as_completed(futures):
                progress_bar.update(1)
    finally:
        progress_bar.close()
        print()

    if processed_counter[0] > 0 and processed_counter[0] % sample_interval_windows != 0:
        log_threaded_sample(processed_counter[0])

    diagnostics['window_execution_seconds'] = time.time() - processing_start
    diagnostics['window_count_processed'] = int(processed_counter[0])
    diagnostics['window_count_ready_for_inference'] = int(processed_counter[0] - skipped_windows[0])
    diagnostics['skipped_windows'] = int(skipped_windows[0])
    diagnostics['mean_windows_per_second'] = (
        processed_counter[0] / diagnostics['window_execution_seconds']
        if diagnostics['window_execution_seconds'] > 0 else 0.0
    )

    log_resource_snapshot("after_window_execution", diagnostics)
    if gpu_profiler is not None:
        gpu_profiler.log_memory_usage("After Window Processing")

    results = _finalize_processed_mask(
        full_mask=full_mask,
        parent=parent,
        adipocyte_properties=adipocyte_properties,
        min_area_threshold_pixels=min_area_threshold_pixels,
        max_area_threshold_pixels=max_area_threshold_pixels,
        height=height,
        width=width,
        image_handler=image_handler,
        async_enabled=async_enabled,
        gpu_profiler=gpu_profiler,
        mask_cleanup=mask_cleanup,
        postprocessed_windows=postprocessed_windows,
        processed_window_coords=processed_window_coords,
        diagnostics=diagnostics,
    )
    _set_execution_diagnostics(diagnostics)
    return results


def process_selected_windows_pipelined(image_handler, predictor, window_size, stride,
                                       min_area_threshold_pixels, max_area_threshold_pixels,
                                       output_dir, window_coords):
    """Backward-compatible alias for the default selected-window threaded executor."""
    return process_selected_windows_threaded(
        image_handler=image_handler,
        predictor=predictor,
        window_size=window_size,
        stride=stride,
        min_area_threshold_pixels=min_area_threshold_pixels,
        max_area_threshold_pixels=max_area_threshold_pixels,
        output_dir=output_dir,
        window_coords=window_coords,
    )


# ================================================================
# MEMORY MANAGEMENT CLASS
# ================================================================

class MemoryManager:
    """Efficient memory management for large-scale processing (from original BIBLE)."""
    
    @staticmethod
    def flush_gpu_memory():
        """Clean up GPU memory from PyTorch and CuPy."""
        try:
            torch.cuda.empty_cache()
            if config.USE_CUPY:
                import cupy as cp
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
    
    @staticmethod
    def read_optimal_image(image_handler, width, height, scaling_factor, desired_level):
        """
        \uD83D\uDE80 OPTIMIZED image reading helper function (from original BIBLE).
        
        Consolidates duplicate image reading logic used across 6+ functions.
        Automatically selects the best pyramid level to minimize I/O and memory usage.
        
        Args:
            image_handler: OpenSlide image handler
            width, height: Original image dimensions  
            scaling_factor: Target scaling factor
            desired_level: Fallback level for non-slide images
            
        Returns:
            numpy.ndarray: RGB image array with shape (target_height, target_width, 3)
        """
        from PIL import Image
        
        target_width = int(width * scaling_factor)
        target_height = int(height * scaling_factor)
        
        # Find the best level that's closest to our target resolution
        if hasattr(image_handler, 'slide') and hasattr(image_handler.slide, 'level_downsamples'):
            downsamples = image_handler.slide.level_downsamples
            level_factors = [1 / ds for ds in downsamples]
            level_diffs = [abs(factor - scaling_factor) for factor in level_factors]
            best_level = level_diffs.index(min(level_diffs))
            level_dims = image_handler.slide.level_dimensions[best_level]
            
            # Read at optimal level (reduces I/O significantly)
            full_image_pil = image_handler.read_region((0, 0), best_level, level_dims)
            
            # Only resize if necessary
            if level_dims != (target_width, target_height):
                full_image_pil = full_image_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            # Fallback for non-slide images
            full_image_pil = image_handler.read_region((0, 0), desired_level, (width, height))
            full_image_pil = full_image_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        return np.array(full_image_pil, dtype=np.uint8)[:, :, :3]


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'find',
    'union',
    'apply_label_mapping_memory_efficient',
    'remove_invalid_labels_memory_efficient',
    'relabel_consecutive_memory_efficient',
    'inference_worker',
    'process_all_windows',
    'MemoryManager'
]
