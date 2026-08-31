#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistics Utilities Module
===========================

Statistical calculation utilities for AdiFind analysis results.
"""

import logging


def calculate_area_statistics(mask_areas, mpp):
    """
    Calculate area statistics for detected adipocytes.
    
    Args:
        mask_areas: Dictionary of adipocyte areas in pixels
        mpp: Microns per pixel value
        
    Returns:
        tuple: (areas_um2, avg_area, total_area)
    """
    if not mask_areas or len(mask_areas) == 0:
        logging.warning("??  No adipocytes found for statistics calculation")
        return [], 0.0, 0.0
    
    areas_um2 = [area * (mpp ** 2) for area in mask_areas.values()]
    avg_area = sum(areas_um2) / len(areas_um2) if areas_um2 else 0
    total_area = sum(areas_um2)
    
    logging.info("?? Average adipocyte area: %.1f \u00B5m\u00B2", avg_area)
    logging.info("?? Total adipocyte area: %.1f \u00B5m\u00B2", total_area)
    
    return areas_um2, avg_area, total_area


def calculate_distance_statistics(adipocyte_distances):
    """
    Calculate distance statistics for adipocytes to tumor boundaries.
    
    Args:
        adipocyte_distances: Dictionary of adipocyte distances to tumor
        
    Returns:
        tuple: (valid_distances, avg_distance, min_distance, max_distance)
    """
    if not adipocyte_distances:
        logging.warning("??  No adipocyte distances provided for statistics calculation")
        return [], 0.0, 0.0, 0.0
    
    valid_distances = [d for d in adipocyte_distances.values() if d >= 0]
    
    if valid_distances and len(valid_distances) > 0:
        avg_distance = sum(valid_distances) / len(valid_distances)
        min_distance = min(valid_distances)
        max_distance = max(valid_distances)
        
        logging.info("?? Distance statistics: avg=%.1f?m, min=%.1f?m, max=%.1f?m", 
                    avg_distance, min_distance, max_distance)
        
        return valid_distances, avg_distance, min_distance, max_distance
    else:
        logging.info("?? No valid distances computed (no tumor found)")
        return [], 0, 0, 0


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'calculate_area_statistics',
    'calculate_distance_statistics'
]
