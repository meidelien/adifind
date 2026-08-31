#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Manager Module
============================

Handles configuration updates from command line arguments and files.

IMPORTANT REMINDER FOR FUTURE TASKS:
=====================================
When running terminal commands in this workspace, always use:
Set-Location "D:\Github\adifind\code"; & "C:\ProgramData\anaconda3\envs\adifind\python.exe" main.py [arguments]

This ensures:
1. Correct working directory (code folder)
2. Proper Python environment (adifind conda env)
3. PowerShell compatibility on Windows
"""

import logging
from config import config


VALID_ANNOTATED_IMAGE_SAVE_MODES = {'fast', 'balanced', 'high_quality'}


def update_config_from_args(args):
    """
    Update configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
    """
    # Verbose / quiet logging
    config.VERBOSE_LOGGING = getattr(args, 'verbose', False)

    # Update configuration from file if provided
    if args.config_file:
        # TODO: Implement config file loading if needed
        logging.info("\uD83D\uDCDD Configuration file specified: %s", args.config_file)
        logging.warning("\u26A0\uFE0F Configuration file loading not yet implemented")
    
    # Update configuration from command line arguments
    config.ENABLE_TISSUE_GUIDANCE = bool(args.tissue_guidance)
    if config.ENABLE_TISSUE_GUIDANCE:
        logging.info("\uD83E\uDDEC Tissue guidance enabled")
    else:
        logging.debug("\u2139\uFE0F Tissue guidance disabled")
    
    if hasattr(args, 'save_tissue_window_grid') and args.save_tissue_window_grid:
        config.SAVE_TISSUE_WINDOW_GRID_THUMBNAIL = True
        logging.info("\uD83E\uDE9F Tissue window grid thumbnail will be saved (low-res preview)")

    if args.tumor_segmentation:
        config.ENABLE_TUMOR_SEGMENTATION = True
        logging.info("\uD83D\uDD2C Tumor segmentation enabled")
    else:
        config.ENABLE_TUMOR_SEGMENTATION = False
        logging.debug("\u2139\uFE0F Tumor segmentation disabled")
    
    if args.save_distance_map:
        if args.tumor_segmentation:
            config.SAVE_DISTANCE_COLORED_IMAGE = True
            logging.info("\uD83C\uDF08 Distance-colored visualization enabled")
        else:
            logging.warning("\u26A0\uFE0F --save_distance_map requires --tumor_segmentation. Distance map disabled.")
            config.SAVE_DISTANCE_COLORED_IMAGE = False
    else:
        config.SAVE_DISTANCE_COLORED_IMAGE = False
        logging.debug("\u2139\uFE0F Distance-colored visualization disabled (use --save_distance_map to enable)")
    
    if args.extended_properties:
        config.CALCULATE_EXTENDED_PROPERTIES = True
        logging.info("\uD83D\uDCCA Extended morphological properties enabled (eccentricity, solidity, extent, perimeter, equivalent_diameter)")
    else:
        config.CALCULATE_EXTENDED_PROPERTIES = False
        logging.debug("\u2139\uFE0F Using BIBLE default properties only (use --extended_properties for additional morphological properties)")
    
    # Debug mode supports optional values:
    #   --debug / --debug processed     -> processed debug outputs
    #   --debug unprocessed             -> processed + raw window debug outputs
    # Backward compatibility: boolean debug values from older snapshots are still accepted.
    debug_arg = getattr(args, 'debug', None)
    debug_selection = None
    if isinstance(debug_arg, bool):
        debug_selection = 'processed' if debug_arg else None
    elif debug_arg in ('processed', 'unprocessed'):
        debug_selection = debug_arg

    config.DEBUG_MODE = debug_selection is not None
    config.DEBUG_SAVE_UNPROCESSED_WINDOWS = (debug_selection == 'unprocessed')

    if config.DEBUG_MODE:
        if config.DEBUG_SAVE_UNPROCESSED_WINDOWS:
            logging.info("\uD83D\uDC1B Debug mode enabled (processed + unprocessed window exports)")
        else:
            logging.info("\uD83D\uDC1B Debug mode enabled (processed window exports)")
    else:
        logging.debug("\u2139\uFE0F Debug mode disabled")
    
    if hasattr(args, 'disable_gpu_accel') and args.disable_gpu_accel:
        config.USE_GPU_INFERENCE = False
        config.USE_CUPY = False
        config.USE_GPU_PREPROCESSING = False
        config.ENABLE_GPU_LABEL_MAPPING = False
        logging.info("\u00F0\u0178\u2013\u00A5\u00EF\u00B8\u008F GPU inference and GPU-accelerated ops disabled via CLI (CPU-only mode)")

    if hasattr(args, 'disable_gpu_ops') and args.disable_gpu_ops:
        config.USE_CUPY = False
        config.USE_GPU_PREPROCESSING = False
        config.ENABLE_GPU_LABEL_MAPPING = False
        logging.info("\u00F0\u0178\u2013\u00A5\u00EF\u00B8\u008F CuPy, GPU preprocessing, and GPU label mapping disabled (GPU inference remains enabled)")

    if hasattr(args, 'disable_gpu_preprocessing') and args.disable_gpu_preprocessing:
        config.USE_GPU_PREPROCESSING = False
        logging.info("\u00F0\u0178\u2013\u00A5\u00EF\u00B8\u008F GPU preprocessing disabled (GPU inference and other GPU ops remain unchanged)")

    if args.min_area is not None:
        config.MIN_ADIPOCYTE_AREA_MICRONS = args.min_area
        logging.info("\uD83D\uDCD0 Minimum area set to: %.1f \u00B5m\u00B2", args.min_area)
    
    if args.max_area is not None:
        config.MAX_ADIPOCYTE_AREA_MICRONS = args.max_area
        logging.info("\uD83D\uDCD0 Maximum area set to: %.1f \u00B5m\u00B2", args.max_area)
    
    if args.batch_size is not None:
        config.BATCH_INFERENCE_SIZE = args.batch_size
        logging.info("\uD83D\uDD22 GPU batch inference size set to: %d windows", args.batch_size)
    
    if args.annotated_scale != 0.3:  # Only log if different from default
        config.ANNOTATED_IMAGE_SCALE = args.annotated_scale
        logging.info("\uD83D\uDDBC\uFE0F Annotated image scale set to: %.2f", args.annotated_scale)
    else:
        config.ANNOTATED_IMAGE_SCALE = args.annotated_scale
    
    # ROI guidance configuration
    if hasattr(args, 'roi_freehand') and args.roi_freehand:
        config.ENABLE_ROI_GUIDANCE = True
        logging.info("\u00F0\u0178\u00A7\u00AC ROI guidance enabled (freehand selection)")
    else:
        config.ENABLE_ROI_GUIDANCE = False

    if hasattr(args, 'roi_max_dim') and args.roi_max_dim is not None:
        if args.roi_max_dim > 0:
            config.ROI_THUMBNAIL_MAX_DIM = args.roi_max_dim
            logging.info("\u00F0\u0178\u2013\u00BC\u00EF\u00B8\u008F ROI thumbnail max dim set to: %d", args.roi_max_dim)
        else:
            logging.warning("\u00E2\u0161\u00A0\u00EF\u00B8\u008F Invalid roi_max_dim; using default: %d", config.ROI_THUMBNAIL_MAX_DIM)

    if hasattr(args, 'roi_polygon_file') and args.roi_polygon_file:
        config.ROI_POLYGON_FILE = args.roi_polygon_file
        config.ENABLE_ROI_GUIDANCE = True
        logging.info("\U0001F4C2 ROI polygon file: %s", args.roi_polygon_file)

    if hasattr(args, 'roi_min_coverage') and args.roi_min_coverage is not None:
        config.ROI_MIN_COVERAGE = float(args.roi_min_coverage)
        logging.info("\u00F0\u0178\u201C\u0090 ROI min coverage set to: %.2f", config.ROI_MIN_COVERAGE)

    # Image saving configuration
    save_mode = getattr(args, 'save_mode', None) or 'balanced'
    if save_mode not in VALID_ANNOTATED_IMAGE_SAVE_MODES:
        supported_modes = ', '.join(sorted(VALID_ANNOTATED_IMAGE_SAVE_MODES))
        raise ValueError(
            f"Unsupported image save mode: {save_mode!r}. Supported modes: {supported_modes}"
        )
    config.ANNOTATED_IMAGE_SAVE_MODE = save_mode
    if save_mode != 'balanced':
        logging.info("\uD83D\uDCBE Image save mode set to: %s", save_mode)

    if getattr(args, 'save_image_annotation', False):
        config.SAVE_ANNOTATED_IMAGE = True
        logging.info("\uD83D\uDDBC\uFE0F Annotated image saving enabled via CLI override")
    elif getattr(args, 'skip_image_annotation', False):
        config.SAVE_ANNOTATED_IMAGE = False
        logging.info("\uD83D\uDDBC\uFE0F Annotated image saving disabled via CLI override")
    else:
        annotated_state = "enabled" if bool(getattr(config, 'SAVE_ANNOTATED_IMAGE', True)) else "disabled"
        logging.info("\uD83D\uDDBC\uFE0F Annotated image saving %s (config default)", annotated_state)

    if getattr(args, 'save_qupath_annotation', False):
        config.ENABLE_QUPATH_EXPORT = True
        config.SAVE_QUPATH_GEOJSON = True
        logging.info("\U0001F5FA\uFE0F QuPath GeoJSON export enabled via CLI override")
    elif getattr(args, 'skip_qupath_annotation', False):
        config.ENABLE_QUPATH_EXPORT = False
        logging.info("\U0001F5FA\uFE0F QuPath GeoJSON export disabled via CLI override")
    else:
        qupath_state = (
            "enabled"
            if bool(getattr(config, 'ENABLE_QUPATH_EXPORT', False) and getattr(config, 'SAVE_QUPATH_GEOJSON', False))
            else "disabled"
        )
        logging.info("\U0001F5FA\uFE0F QuPath GeoJSON export %s (config default)", qupath_state)

    if hasattr(args, 'benchmark_saving') and args.benchmark_saving:
        config.BENCHMARK_IMAGE_SAVING = True
        logging.info("\u23F1\uFE0F Image saving benchmark enabled - will test all save methods")
    
    if args.profiling:
        config.ENABLE_PROFILING = True
        logging.info("\u23F1\uFE0F Detailed profiling enabled - timing reports will be displayed")
    else:
        config.ENABLE_PROFILING = False
    
    # Adipocyte ID visibility
    if args.show_adipocyte_ids:
        config.SHOW_ADIPOCYTE_IDS = True
        logging.info("\uD83D\uDD22 Adipocyte ID numbers will be shown on annotated images")
    elif args.hide_adipocyte_ids:
        config.SHOW_ADIPOCYTE_IDS = False
        logging.info("\uD83D\uDD22 Adipocyte ID numbers will be hidden on annotated images")
    # If neither flag is set, use the default from config.py
    
    # Analysis grid visibility
    if args.show_grid:
        config.SHOW_GRID = True
        logging.info("\uD83D\uDCCA Analysis grid overlay will be shown on annotated images")
    elif args.hide_grid:
        config.SHOW_GRID = False
        logging.info("\uD83D\uDCCA Analysis grid overlay will be hidden on annotated images")
    # If neither flag is set, use the default from config.py
    
    # Low-memory mode configuration
    if hasattr(args, 'low_memory') and args.low_memory:
        _enable_low_memory_mode()
        logging.info("\uD83D\uDCBE Low-memory mode enabled (for 64GB RAM systems)")
    elif hasattr(args, 'memmap_mask') and args.memmap_mask:
        config.USE_MEMMAP_MASK = True
        logging.info("\uD83D\uDCBE Memory-mapped mask storage enabled")
    else:
        # Auto-detect low memory systems
        _auto_detect_low_memory()


def _enable_low_memory_mode():
    """Enable all low-memory optimizations for systems with 64GB RAM or less."""
    config.LOW_MEMORY_MODE = True
    config.USE_MEMMAP_MASK = True
    
    # Reduce parallel workers
    config.MAX_IO_WORKERS = min(config.MAX_IO_WORKERS, 16)
    config.CHUNK_WORKERS = min(config.CHUNK_WORKERS, 8)
    
    # Reduce batch sizes
    config.BATCH_INFERENCE_SIZE = min(config.BATCH_INFERENCE_SIZE, 4)
    config.MASK_CHUNK_SIZE = min(config.MASK_CHUNK_SIZE, 2048)
    
    # Reduce I/O cache
    config.ASYNC_CACHE_SIZE_MB = min(config.ASYNC_CACHE_SIZE_MB, 256)
    config.ASYNC_PREFETCH_SIZE = min(config.ASYNC_PREFETCH_SIZE, 10)
    
    # Disable memory-heavy outputs
    config.SAVE_POSTPROCESSED_IMAGE = False
    config.SAVE_UNPROCESSED_IMAGE = False
    
    # Reduce annotated image size
    if config.ANNOTATED_IMAGE_SCALE > 0.25:
        config.ANNOTATED_IMAGE_SCALE = 0.25
    
    logging.info("   \u2022 MAX_IO_WORKERS: %d", config.MAX_IO_WORKERS)
    logging.info("   \u2022 BATCH_INFERENCE_SIZE: %d", config.BATCH_INFERENCE_SIZE)
    logging.info("   \u2022 MASK_CHUNK_SIZE: %d", config.MASK_CHUNK_SIZE)
    logging.info("   \u2022 USE_MEMMAP_MASK: %s", config.USE_MEMMAP_MASK)
    logging.info("   \u2022 ANNOTATED_IMAGE_SCALE: %.2f", config.ANNOTATED_IMAGE_SCALE)


def _auto_detect_low_memory():
    """Auto-detect system RAM and enable low-memory mode if needed."""
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        threshold_gb = getattr(config, 'LOW_MEMORY_THRESHOLD_GB', 96)
        
        if total_ram_gb < threshold_gb:
            logging.info(f"\uD83D\uDD0D Detected {total_ram_gb:.1f}GB RAM (threshold: {threshold_gb}GB)")
            logging.info("\uD83D\uDCBE Auto-enabling low-memory mode...")
            _enable_low_memory_mode()
        else:
            logging.info(f"\u2705 System has {total_ram_gb:.1f}GB RAM - using standard memory settings")
    except ImportError:
        logging.debug("psutil not available - skipping RAM auto-detection")


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'update_config_from_args'
]
