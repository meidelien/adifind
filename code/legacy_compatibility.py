#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy Compatibility Module
===========================

Provides backward compatibility functions for AdiFind.
"""

import sys


def adifind_main():
    """Legacy main function name."""
    from main import main
    return main()


def run_adipocyte_detection(image_path, output_dir=None, **kwargs):
    """
    Legacy function for running adipocyte detection programmatically.
    
    Args:
        image_path: Path to whole slide image
        output_dir: Output directory (optional)
        **kwargs: Additional configuration parameters
    """
    from main import main
    
    # Build arguments
    sys.argv = ['adifind', image_path]
    
    if output_dir:
        sys.argv.extend(['--output_dir', output_dir])
    
    # Add other parameters
    for key, value in kwargs.items():
        if key == 'window_size' and isinstance(value, (list, tuple)):
            sys.argv.extend(['--window_size'] + [str(v) for v in value])
        elif key == 'stride' and isinstance(value, (list, tuple)):
            sys.argv.extend(['--stride'] + [str(v) for v in value])
        elif key == 'tissue_guidance' and value:
            sys.argv.append('--tissue_guidance')
        elif key == 'debug':
            if isinstance(value, str):
                debug_mode = value.strip().lower()
                if debug_mode in ('processed', 'unprocessed'):
                    sys.argv.extend(['--debug', debug_mode])
                elif debug_mode:
                    # Backward-compatible fallback for any truthy string
                    sys.argv.append('--debug')
            elif value:
                sys.argv.append('--debug')
        elif key in ['min_area', 'max_area', 'gpu_id']:
            sys.argv.extend([f'--{key}', str(value)])
    
    return main()


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'adifind_main',
    'run_adipocyte_detection'
]
