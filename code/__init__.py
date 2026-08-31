#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AdiFind: adipocyte detection and analysis for whole slide images.

Usage:
    python -m adifind image.svs --output_dir results/
    from adifind import process_image
"""

__version__ = "16.0.0"
__author__ = "Martin Eide Lien"
__description__ = "Advanced Adipocyte Detection in Whole Slide Images"

# Configuration
from config import config, paths, Config, Paths

# Image handling
from image_processing import (
    ImageHandler,
    get_mpp,
    generate_sliding_windows,
    is_window_predominantly_black,
)

# Core processing
from core_processing import (
    process_all_windows,
    inference_worker,
    apply_label_mapping_memory_efficient,
)

# Visualization and export
from visualization import (
    annotate_image_with_adipocytes,
    export_qupath_annotations,
    export_results_csv,
)

# Models
from models import (
    configure_adipocyte_model,
    configure_tumor_model,
    CustomBatchPredictor,
)

# System utilities
from system_utils import (
    monitor,
    memory_manager,
    SystemMonitor,
    MemoryManager,
)

# Main entry point
from main import main, adifind_main, run_adipocyte_detection


def process_image(image_path, output_dir=None, **kwargs):
    """Process a single image by forwarding arguments to the CLI entrypoint."""
    import sys
    from pathlib import Path

    args = [image_path]
    if output_dir:
        args.extend(['--output_dir', str(output_dir)])

    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                args.append(f'--{key}')
        else:
            args.extend([f'--{key}', str(value)])

    original_argv = sys.argv
    sys.argv = ['adifind'] + args

    try:
        main()
    finally:
        sys.argv = original_argv


def get_version():
    """Return the current AdiFind version string."""
    return __version__


__all__ = [
    '__version__',
    '__author__',
    'get_version',
    'main',
    'adifind_main',
    'run_adipocyte_detection',
    'process_image',
    'config',
    'paths',
    'Config',
    'Paths',
    'ImageHandler',
    'get_mpp',
    'generate_sliding_windows',
    'process_all_windows',
    'inference_worker',
    'apply_label_mapping_memory_efficient',
    'annotate_image_with_adipocytes',
    'export_qupath_annotations',
    'export_results_csv',
    'configure_adipocyte_model',
    'configure_tumor_model',
    'CustomBatchPredictor',
    'monitor',
    'memory_manager',
    'SystemMonitor',
    'MemoryManager',
]
