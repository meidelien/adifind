#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI argument parsing and validation for AdiFind."""

import os
import logging
import argparse
import re
from pathlib import Path


def _get_version():
    """Read version from __init__.py without importing heavyweight modules."""
    try:
        init_text = (Path(__file__).resolve().parent / "__init__.py").read_text(encoding="utf-8")
        match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.MULTILINE)
        if match:
            return match.group(1)
    except OSError:
        pass
    try:
        from importlib.metadata import version

        return version("adifind")
    except Exception:
        return "unknown"

VALID_IMAGE_EXTENSIONS = {
    '.svs', '.ndpi', '.tiff', '.tif', '.vms', '.vmu', '.scn',
    '.mrxs', '.svslide', '.bif', '.czi', '.png', '.jpg', '.jpeg'
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AdiFind: Advanced Adipocyte Detection in WSI')
    parser.add_argument('--version', action='version', version=f'%(prog)s {_get_version()}')
    
    parser.add_argument('image_path', type=str, nargs='?', help='Path to the whole slide image or directory containing images')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory (default: auto-generated)')
    parser.add_argument('--window_size', type=int, nargs=2, default=[2048, 2048],
                       help='Window size for processing [width height]')
    parser.add_argument('--stride', type=int, nargs=2, default=[1024, 1024],
                       help='Stride for sliding window [stride_x stride_y]')
    parser.add_argument('--min_area', type=float, default=None,
                       help='Minimum adipocyte area in \u00B5m\u00B2')
    parser.add_argument('--max_area', type=float, default=None,
                       help='Maximum adipocyte area in \u00B5m\u00B2')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to custom configuration file')
    parser.add_argument('--tissue_guidance', action='store_true',
                       help='Enable tissue guidance for targeted processing')
    parser.add_argument('--save_tissue_window_grid', action='store_true',
                       help='Save low-res thumbnail with tissue-guided window grid overlay')
    parser.add_argument('--tumor_segmentation', action='store_true',
                       help='Enable tumor segmentation and distance analysis')
    parser.add_argument('--save_distance_map', action='store_true',
                       help='Save distance-colored visualization (requires --tumor_segmentation)')
    parser.add_argument('--extended_properties', action='store_true',
                       help='Calculate additional morphological properties (eccentricity, solidity, extent, perimeter, equivalent_diameter)')
    parser.add_argument('--annotated_scale', type=float, default=0.3,
                       help='Scaling factor for annotated image output (default: 0.3)')
    parser.add_argument('--save_mode', type=str, default='balanced',
                       choices=['fast', 'balanced', 'high_quality'],
                       help='Image saving mode: fast (JPEG), balanced (compressed TIFF), high_quality (optimized TIFF)')
    annotated_output_group = parser.add_mutually_exclusive_group()
    annotated_output_group.add_argument(
        '--save_image_annotation',
        action='store_true',
        help='Force saving the base annotated TIFF output for this run (overrides config SAVE_ANNOTATED_IMAGE)'
    )
    annotated_output_group.add_argument(
        '--skip_image_annotation',
        action='store_true',
        help='Skip saving the base annotated TIFF output for this run (overrides config SAVE_ANNOTATED_IMAGE)'
    )
    qupath_output_group = parser.add_mutually_exclusive_group()
    qupath_output_group.add_argument(
        '--save_qupath_annotation',
        action='store_true',
        help='Force saving the QuPath GeoJSON annotation for this run (overrides config QuPath GeoJSON export settings)'
    )
    qupath_output_group.add_argument(
        '--skip_qupath_annotation',
        action='store_true',
        help='Skip saving the QuPath GeoJSON annotation for this run (overrides config QuPath GeoJSON export settings)'
    )
    parser.add_argument('--benchmark_saving', action='store_true',
                       help='Benchmark different image saving methods and show performance comparison')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed startup diagnostics, GPU status per window, and debug-level log messages')
    parser.add_argument('--profiling', action='store_true',
                       help='Enable detailed profiling information and timing reports')
    parser.add_argument(
        '--debug',
        nargs='?',
        const='processed',
        choices=['processed', 'unprocessed'],
        default=None,
        metavar='MODE',
        help='Enable debug mode. MODE: processed (default) or unprocessed (adds raw-window exports)'
    )
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use for processing')
    parser.add_argument('--disable_gpu_accel', action='store_true',
                       help='Disable GPU inference and GPU-accelerated ops; run in CPU-only mode')
    parser.add_argument('--disable_gpu_ops', action='store_true',
                       help='Disable CuPy ops, GPU preprocessing, and GPU label mapping while keeping GPU inference enabled')
    parser.add_argument('--disable_gpu_preprocessing', action='store_true',
                       help='Disable GPU Sobel/inversion preprocessing while keeping other GPU paths enabled')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Number of windows to batch for GPU inference (default: from config.py)')
    parser.add_argument('--resume_batch', type=str, default=None,
                       help='Resume batch processing from a previous state file')
    parser.add_argument('--resume_failed', action='store_true',
                       help='Retry processing of previously failed images (overrides config setting)')
    parser.add_argument('--list_resumable', action='store_true',
                       help='List all resumable batch jobs in the current directory')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be processed without actually processing images')
    parser.add_argument(
        '--gpu_probe_only',
        action='store_true',
        help=argparse.SUPPRESS,
    )

    # ROI selection options
    parser.add_argument('--roi_freehand', action='store_true',
                       help='Enable interactive freehand ROI selection on a downscaled thumbnail')
    parser.add_argument('--roi_polygon_file', type=str, default=None,
                       help='Path to a previously saved ROI polygon JSON file (skips interactive GUI)')
    parser.add_argument('--roi_max_dim', type=int, default=2048,
                       help='Max dimension (pixels) for ROI thumbnail (default: 2048)')
    parser.add_argument('--roi_min_coverage', type=float, default=0.2,
                       help='Minimum ROI coverage (0-1) required for a window to be processed (default: 0.2)')
    
    # Memory optimization options
    parser.add_argument('--low_memory', action='store_true',
                       help='Enable low-memory mode for systems with 64GB RAM or less (uses disk-backed mask storage)')
    parser.add_argument('--memmap_mask', action='store_true',
                       help='Use memory-mapped file for mask storage (reduces RAM but slower I/O)')
    
    # Adipocyte ID visualization toggle
    adipocyte_id_group = parser.add_mutually_exclusive_group()
    adipocyte_id_group.add_argument('--show_adipocyte_ids', action='store_true',
                                    help='Show adipocyte ID numbers on annotated images')
    adipocyte_id_group.add_argument('--hide_adipocyte_ids', action='store_true',
                                    help='Hide adipocyte ID numbers on annotated images (default: hidden)')
    
    # Analysis grid visualization toggle
    grid_group = parser.add_mutually_exclusive_group()
    grid_group.add_argument('--show_grid', action='store_true',
                           help='Show analysis grid overlay on annotated images')
    grid_group.add_argument('--hide_grid', action='store_true',
                           help='Hide analysis grid overlay on annotated images (default: shown)')
    
    return parser.parse_args()


def get_image_files(image_path):
    """
    Get list of image files to process.
    
    Args:
        image_path: Path to image file or directory
        
    Returns:
        List of image file paths to process
    """
    if os.path.isdir(image_path):
        image_files = [
            os.path.join(image_path, f)
            for f in os.listdir(image_path)
            if os.path.isfile(os.path.join(image_path, f))
            and Path(f).suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        return sorted(image_files)  # Sort for consistent processing order
    else:
        # Process a single image file
        return [image_path]


def validate_inputs(args):
    """Validate input arguments and files."""
    # Check if image file or directory exists
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file or directory not found: {args.image_path}")
    
    # If it's a directory, check for valid image files
    if os.path.isdir(args.image_path):
        image_files = [
            f for f in os.listdir(args.image_path)
            if os.path.isfile(os.path.join(args.image_path, f))
            and Path(f).suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        if not image_files:
            raise FileNotFoundError(f"No valid image files found in directory: {args.image_path}")
        logging.info(f"Found {len(image_files)} image files to process")
    else:
        # Validate single image format
        image_ext = Path(args.image_path).suffix.lower()
        if image_ext not in VALID_IMAGE_EXTENSIONS:
            logging.warning("Image format %s may not be supported", image_ext)
    
    # Validate window size and stride
    if args.window_size[0] <= 0 or args.window_size[1] <= 0:
        raise ValueError("Window size must be positive")
    
    if args.stride[0] <= 0 or args.stride[1] <= 0:
        raise ValueError("Stride must be positive")
    
    # Validate area thresholds
    if args.min_area is not None and args.min_area <= 0:
        raise ValueError("Minimum area must be positive")
    
    if args.max_area is not None and args.max_area <= 0:
        raise ValueError("Maximum area must be positive")
    
    if args.min_area is not None and args.max_area is not None:
        if args.min_area >= args.max_area:
            raise ValueError("Minimum area must be less than maximum area")
    
    # Validate annotated scale
    if args.annotated_scale <= 0:
        raise ValueError("Annotated scale must be positive")
    
    if args.annotated_scale > 2.0:
        logging.warning("Annotated scale %.2f is greater than 2.0 - this may use significant memory", args.annotated_scale)

    # ROI validation
    if hasattr(args, 'roi_max_dim') and args.roi_max_dim is not None:
        if args.roi_max_dim <= 0:
            raise ValueError("ROI max dim must be positive")
    if hasattr(args, 'roi_min_coverage') and args.roi_min_coverage is not None:
        if args.roi_min_coverage < 0.0 or args.roi_min_coverage > 1.0:
            raise ValueError("ROI min coverage must be between 0.0 and 1.0")
    
    logging.info("? Input validation passed")


__all__ = [
    'parse_arguments',
    'validate_inputs',
    'get_image_files',
    'VALID_IMAGE_EXTENSIONS',
]
