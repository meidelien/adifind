#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accelerated Visualization Module
================================

High-performance image saving optimizations for AdiFind annotated images.
Provides multiple acceleration strategies for faster image output.
"""

import os
import time
import logging
import numpy as np
import cv2
from pathlib import Path

# Import configuration
from config import config


def save_annotated_image_accelerated(annotated_image, output_dir, image_name, compression_mode="fast"):
    """
    Accelerated image saving with multiple optimization strategies.
    
    Args:
        annotated_image: Numpy array of the annotated image
        output_dir: Output directory
        image_name: Base image name
        compression_mode: "fast", "balanced", or "high_quality"
    
    Returns:
        str: Path to saved image file
        float: Time taken to save
    """
    save_start = time.time()
    
    if compression_mode == "fast":
        output_path = _save_fast_jpeg(annotated_image, output_dir, image_name)
    elif compression_mode == "balanced":
        output_path = _save_compressed_tiff(annotated_image, output_dir, image_name)
    elif compression_mode == "high_quality":
        output_path = _save_optimized_tiff(annotated_image, output_dir, image_name)
    else:
        supported_modes = "fast, balanced, high_quality"
        raise ValueError(
            f"Unsupported image save mode: {compression_mode!r}. "
            f"Supported modes: {supported_modes}"
        )
    
    save_time = time.time() - save_start
    logging.info(f"?? Image saved ({compression_mode}): {save_time:.3f}s -> {output_path}")
    
    return output_path, save_time


def _save_fast_jpeg(annotated_image, output_dir, image_name):
    """
    FASTEST: Save as high-quality JPEG (faster than TIFF).
    Good for quick previews and analysis review.
    """
    output_path = os.path.join(output_dir, f"{image_name}_adifind_annotated.jpg")
    
    # Ensure RGB format for JPEG
    if annotated_image.shape[2] == 3:
        # Convert BGR to RGB if needed
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    else:
        annotated_image_rgb = annotated_image
    
    # High quality JPEG (quality=95)
    success = cv2.imwrite(output_path, annotated_image_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    if not success:
        raise RuntimeError(f"Failed to save JPEG: {output_path}")
    
    return output_path


def _save_compressed_tiff(annotated_image, output_dir, image_name):
    """
    BALANCED: Save as LZW-compressed TIFF (3-5x faster than uncompressed).
    Lossless compression with good speed/size balance.
    """
    import tifffile
    
    output_path = os.path.join(output_dir, f"{image_name}_adifind_annotated.tiff")
    
    # Use LZW compression for good speed/size balance
    tifffile.imwrite(
        output_path, 
        annotated_image, 
        compression='lzw',  # Fast lossless compression
        bigtiff=True,
        photometric='rgb'
    )
    
    return output_path


def _save_optimized_tiff(annotated_image, output_dir, image_name):
    """
    HIGH QUALITY: Save as optimized TIFF with threading.
    Uses multiple threads for faster I/O.
    """
    import tifffile
    
    output_path = os.path.join(output_dir, f"{image_name}_adifind_annotated.tiff")
    
    # Optimized TIFF with threading
    tifffile.imwrite(
        output_path, 
        annotated_image,
        compression='lzw',
        bigtiff=True,
        photometric='rgb',
        predictor=2,  # Horizontal differencing for better compression
        tile=(512, 512),  # Tiled format for faster random access
        software='AdiFind-Accelerated'
    )
    
    return output_path


def benchmark_save_methods(annotated_image, output_dir, image_name):
    """
    Benchmark different saving methods to find the fastest.
    
    Args:
        annotated_image: Test image array
        output_dir: Output directory
        image_name: Base image name
    
    Returns:
        dict: Performance results for each method
    """
    methods = ["fast", "balanced", "high_quality"]
    results = {}
    
    for method in methods:
        try:
            # Create unique filename for this test
            test_name = f"{image_name}_benchmark_{method}"
            output_path, save_time = save_annotated_image_accelerated(
                annotated_image, output_dir, test_name, method
            )
            
            # Get file size
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            
            results[method] = {
                'save_time': save_time,
                'file_size_mb': file_size_mb,
                'path': output_path
            }
            
            # Clean up benchmark file
            os.remove(output_path)
            
        except Exception as e:
            results[method] = {'error': str(e)}
    
    # Log results
    logging.info("?? IMAGE SAVING BENCHMARK RESULTS:")
    logging.info("=" * 50)
    for method, result in results.items():
        if 'error' not in result:
            logging.info(f"{method:>15}: {result['save_time']:6.3f}s, {result['file_size_mb']:6.1f}MB")
        else:
            logging.info(f"{method:>15}: ERROR - {result['error']}")
    
    return results


def get_recommended_save_mode():
    """
    Get recommended save mode based on system capabilities and user preferences.
    
    Returns:
        str: Recommended compression mode
    """
    # For most users, "balanced" provides the best speed/quality trade-off
    # "fast" is good for quick analysis review
    # "high_quality" is good for publications
    
    return "balanced"  # Default recommendation


# ================================================================
# INTEGRATION FUNCTIONS
# ================================================================

def replace_current_save_function(annotated_image, output_dir, image_name=None, acceleration_mode=None):
    """
    Drop-in replacement for current annotated image saving.
    
    Args:
        annotated_image: Numpy array of annotated image
        output_dir: Output directory
        image_name: Image name (if None, uses old filename)
        acceleration_mode: Acceleration mode to use
    
    Returns:
        str: Path to saved file
    """
    if acceleration_mode is None:
        acceleration_mode = get_recommended_save_mode()
    
    if image_name is None:
        # Fallback to old naming for compatibility
        import tifffile
        output_path = os.path.join(output_dir, f"annotated_image.tiff")
        tifffile.imwrite(output_path, annotated_image, bigtiff=True)
        return output_path
    else:
        output_path, _ = save_annotated_image_accelerated(
            annotated_image, output_dir, image_name, acceleration_mode
        )
        return output_path


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'save_annotated_image_accelerated',
    'benchmark_save_methods',
    'get_recommended_save_mode',
    'replace_current_save_function'
]
