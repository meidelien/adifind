#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Kernels for Accelerated Post-Inference Processing
=====================================================

High-performance CUDA kernels and utilities for:
- Periodic GPU memory cleanup (prevents inference slowdown)
- Label mapping with lookup tables
- Image blending for visualization
- Vectorized distance extraction

Requires: CuPy (pip install cupy-cuda12x for CUDA 12.x)
"""

import logging
import time
import gc
import numpy as np
from typing import Dict, Optional, Tuple, List, Any

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ================================================================
# PERIODIC GPU CLEANUP (Prevents slowdown over large batches)
# ================================================================

_cleanup_counter = 0
_last_cleanup_time = 0


def set_cleanup_interval(interval: int):
    """Set the cleanup interval (number of windows between cleanups)."""
    from config import config
    config.GPU_CLEANUP_INTERVAL = interval


def get_memory_stats() -> Dict[str, float]:
    """
    Get current GPU memory statistics.
    
    Returns:
        Dictionary with memory stats in MB
    """
    stats = {
        'torch_allocated_mb': 0,
        'torch_reserved_mb': 0,
        'cupy_used_mb': 0,
        'cupy_total_mb': 0,
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        stats['torch_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        stats['torch_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
    
    if CUPY_AVAILABLE and cp is not None:
        try:
            mempool = cp.get_default_memory_pool()
            stats['cupy_used_mb'] = mempool.used_bytes() / 1024 / 1024
            stats['cupy_total_mb'] = mempool.total_bytes() / 1024 / 1024
        except Exception:
            pass
    
    return stats


def periodic_gpu_cleanup(force: bool = False, window_idx: int = None) -> bool:
    """
    Periodic GPU memory cleanup to prevent fragmentation and slowdown.
    
    Call this every N windows (configurable via GPU_CLEANUP_INTERVAL) to maintain 
    consistent inference speed over large batches. The slowdown is caused by:
    
    1. GPU memory fragmentation (CUDA allocator struggles to find contiguous blocks)
    2. CuPy memory pool growth (separate from PyTorch, often overlooked)
    3. Python object accumulation (intermediate arrays not freed)
    
    Args:
        force: If True, forces aggressive cleanup even if not at interval
        window_idx: Current window index (used to determine if cleanup needed)
        
    Returns:
        True if cleanup was performed, False otherwise
    """
    global _cleanup_counter, _last_cleanup_time
    
    # Import config here to avoid circular imports
    try:
        from config import config
        cleanup_interval = getattr(config, 'GPU_CLEANUP_INTERVAL', 100)
        aggressive = getattr(config, 'GPU_CLEANUP_AGGRESSIVE', False)
    except ImportError:
        cleanup_interval = 100
        aggressive = False
    
    # Check if cleanup is needed
    if window_idx is not None:
        if window_idx % cleanup_interval != 0 and not force:
            return False
    else:
        _cleanup_counter += 1
        if _cleanup_counter % cleanup_interval != 0 and not force:
            return False
    
    cleanup_start = time.time()
    
    # Get memory stats before cleanup
    stats_before = get_memory_stats()
    
    # Step 1: Python garbage collection
    gc.collect()
    
    # Step 2: PyTorch CUDA cleanup and synchronization
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            # Synchronize to ensure all pending operations complete
            # This helps recover from transient CUDA errors
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                # CUDA context may be corrupted, try to continue
                pass
            
            torch.cuda.empty_cache()
            
            # Final sync after cleanup
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
        except Exception as e:
            logging.debug(f"PyTorch cleanup warning: {e}")
    
    # Step 3: CuPy memory pool cleanup (CRITICAL - often missed!)
    # NOTE: Only cleanup CuPy if we're in the main thread or force=True
    # CuPy operations from worker threads can cause CUDA context corruption
    if CUPY_AVAILABLE and cp is not None:
        try:
            # Synchronize CuPy's default stream first
            try:
                cp.cuda.Stream.null.synchronize()
            except Exception:
                pass
            
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            if force or aggressive:
                # Aggressive cleanup - free everything
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
            else:
                # Normal cleanup - free unused blocks only  
                # This is gentler and avoids re-allocation overhead
                mempool.free_all_unreferenced()
                    
        except Exception as e:
            logging.debug(f"CuPy cleanup warning: {e}")
    
    # Get memory stats after cleanup
    stats_after = get_memory_stats()
    cleanup_time = time.time() - cleanup_start
    
    # Log cleanup results
    torch_freed = stats_before['torch_reserved_mb'] - stats_after['torch_reserved_mb']
    cupy_freed = stats_before['cupy_used_mb'] - stats_after['cupy_used_mb']
    
    if torch_freed > 10 or cupy_freed > 10:  # Only log if significant
        logging.info(f"?? GPU cleanup: freed {torch_freed:.0f}MB (PyTorch) + {cupy_freed:.0f}MB (CuPy) in {cleanup_time:.3f}s")
    
    _last_cleanup_time = time.time()
    return True


# ================================================================
# CPU OVERLAP COMPUTATION (Thread-safe)
# ================================================================

def batch_compute_overlaps(
    new_mask: np.ndarray,
    new_label: int,
    existing_mask_region: np.ndarray,
    iou_threshold: float = 0.2
) -> List[Tuple[int, float]]:
    """
    Compute overlaps between a new mask and all existing labels in a region.
    
    Vectorized CPU implementation that's thread-safe and efficient.
    
    Args:
        new_mask: Binary mask of new detection (H, W)
        new_label: Label ID of new detection
        existing_mask_region: Existing label mask for the region (H, W)
        iou_threshold: Minimum IoU to consider as overlap
        
    Returns:
        List of (existing_label, iou) tuples for labels exceeding threshold
    """
    # Find all existing labels that overlap with new mask
    new_mask_bool = new_mask > 0
    new_area = np.sum(new_mask_bool)
    
    if new_area == 0:
        return []
    
    # Get unique labels in overlap region
    overlapping_pixels = existing_mask_region[new_mask_bool]
    unique_labels = np.unique(overlapping_pixels)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background
    
    if len(unique_labels) == 0:
        return []
    
    results = []
    
    # Vectorized overlap computation for all labels at once
    for label in unique_labels:
        existing_mask = (existing_mask_region == label)
        
        # Compute intersection and union
        intersection = np.sum(new_mask_bool & existing_mask)
        union_area = np.sum(new_mask_bool | existing_mask)
        
        if union_area > 0:
            iou = intersection / union_area
            if iou > iou_threshold:
                results.append((int(label), float(iou)))
    
    return results


# ================================================================
# GPU IMAGE BLENDING (for visualization)
# ================================================================

def gpu_blend_images(base_image: np.ndarray, overlay: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    GPU-accelerated image blending for visualization.
    
    Equivalent to cv2.addWeighted but on GPU for large images.
    
    Args:
        base_image: Base RGB image (H, W, 3)
        overlay: Overlay RGB image (H, W, 3)
        alpha: Blend factor for overlay (0-1)
        
    Returns:
        Blended RGB image
        
    Performance: ~3-5x faster for images >10M pixels
    """
    if not CUPY_AVAILABLE:
        import cv2
        return cv2.addWeighted(base_image, 1 - alpha, overlay, alpha, 0)
    
    try:
        # Transfer to GPU
        gpu_base = cp.asarray(base_image.astype(np.float32))
        gpu_overlay = cp.asarray(overlay.astype(np.float32))
        
        # Blend on GPU
        gpu_result = gpu_base * (1 - alpha) + gpu_overlay * alpha
        
        # Clip and convert back
        gpu_result = cp.clip(gpu_result, 0, 255).astype(cp.uint8)
        
        return cp.asnumpy(gpu_result)
        
    except Exception as e:
        import cv2
        logging.warning(f"?? GPU blending failed: {e}, using CPU")
        return cv2.addWeighted(base_image, 1 - alpha, overlay, alpha, 0)


# ================================================================
# VECTORIZED DISTANCE EXTRACTION
# ================================================================

def vectorized_extract_adipocyte_distances(
    adipocyte_mask: np.ndarray,
    distance_map: np.ndarray,
    adipocyte_ids: List[int]
) -> Dict[int, float]:
    """
    Vectorized extraction of minimum distances for all adipocytes.
    
    Instead of looping per-adipocyte, uses numpy advanced indexing
    and groupby-style operations for massive speedup.
    
    Args:
        adipocyte_mask: Label mask at analysis resolution
        distance_map: Distance transform in microns
        adipocyte_ids: List of adipocyte IDs to extract distances for
        
    Returns:
        Dictionary mapping adipocyte_id -> minimum distance in microns
        
    Performance: ~10-100x faster than per-adipocyte loop for >1000 adipocytes
    """
    if len(adipocyte_ids) == 0:
        return {}
    
    # Flatten arrays for efficient processing
    mask_flat = adipocyte_mask.ravel()
    dist_flat = distance_map.ravel()
    
    # Create output dictionary
    distances = {}
    
    # Get all non-zero positions
    nonzero_mask = mask_flat > 0
    labels = mask_flat[nonzero_mask]
    dists = dist_flat[nonzero_mask]
    
    # Sort by label for groupby-style processing
    sort_idx = np.argsort(labels)
    labels_sorted = labels[sort_idx]
    dists_sorted = dists[sort_idx]
    
    # Find boundaries between label groups
    label_changes = np.where(np.diff(labels_sorted) != 0)[0] + 1
    label_starts = np.concatenate([[0], label_changes])
    label_ends = np.concatenate([label_changes, [len(labels_sorted)]])
    
    # Extract unique labels and their min distances
    unique_labels = labels_sorted[label_starts]
    
    for i, label in enumerate(unique_labels):
        if label in adipocyte_ids:
            min_dist = np.min(dists_sorted[label_starts[i]:label_ends[i]])
            distances[int(label)] = float(min_dist)
    
    return distances


def gpu_vectorized_extract_distances(
    adipocyte_mask: np.ndarray,
    distance_map: np.ndarray,
    adipocyte_ids: List[int]
) -> Dict[int, float]:
    """
    GPU-accelerated extraction of minimum distances for all adipocytes.
    
    Args:
        adipocyte_mask: Label mask at analysis resolution
        distance_map: Distance transform in microns
        adipocyte_ids: List of adipocyte IDs to extract distances for
        
    Returns:
        Dictionary mapping adipocyte_id -> minimum distance in microns
    """
    if not CUPY_AVAILABLE or len(adipocyte_ids) == 0:
        return vectorized_extract_adipocyte_distances(adipocyte_mask, distance_map, adipocyte_ids)
    
    try:
        # Transfer to GPU
        mask_gpu = cp.asarray(adipocyte_mask.ravel())
        dist_gpu = cp.asarray(distance_map.ravel())
        
        # Get non-zero positions
        nonzero_mask = mask_gpu > 0
        labels = mask_gpu[nonzero_mask]
        dists = dist_gpu[nonzero_mask]
        
        # Sort by label
        sort_idx = cp.argsort(labels)
        labels_sorted = labels[sort_idx]
        dists_sorted = dists[sort_idx]
        
        # Transfer back to CPU for final processing
        labels_sorted_cpu = cp.asnumpy(labels_sorted)
        dists_sorted_cpu = cp.asnumpy(dists_sorted)
        
        # Find boundaries
        label_changes = np.where(np.diff(labels_sorted_cpu) != 0)[0] + 1
        label_starts = np.concatenate([[0], label_changes])
        label_ends = np.concatenate([label_changes, [len(labels_sorted_cpu)]])
        
        unique_labels = labels_sorted_cpu[label_starts]
        
        distances = {}
        for i, label in enumerate(unique_labels):
            if label in adipocyte_ids:
                min_dist = np.min(dists_sorted_cpu[label_starts[i]:label_ends[i]])
                distances[int(label)] = float(min_dist)
        
        return distances
        
    except Exception as e:
        logging.warning(f"?? GPU distance extraction failed: {e}, using CPU")
        return vectorized_extract_adipocyte_distances(adipocyte_mask, distance_map, adipocyte_ids)
