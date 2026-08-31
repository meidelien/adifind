#!/usr/bin/env python3
"""
Tissue Guidance Caching System
==============================

Implements intelligent caching of tissue detection results to avoid
redundant tissue segmentation for the same slides.
"""

import os
import json
import hashlib
import pickle
import logging
import numpy as np
from pathlib import Path

class TissueGuidanceCache:
    """Caches tissue detection results for faster reprocessing."""
    
    def __init__(self, cache_dir="tissue_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_slide_hash(self, image_path, tissue_model_path):
        """Generate unique hash for slide + model combination."""
        slide_stat = os.stat(image_path)
        model_stat = os.stat(tissue_model_path) if os.path.exists(tissue_model_path) else None
        
        hash_data = f"{image_path}_{slide_stat.st_size}_{slide_stat.st_mtime}"
        if model_stat:
            hash_data += f"_{model_stat.st_mtime}"
        
        return hashlib.md5(hash_data.encode()).hexdigest()
    
    def cache_tissue_results(self, slide_hash, tissue_regions, tissue_mask):
        """Cache tissue detection results."""
        cache_file = self.cache_dir / f"{slide_hash}_tissue.pkl"
        
        cache_data = {
            'tissue_regions': tissue_regions,
            'tissue_mask': tissue_mask,
            'timestamp': os.path.getctime(cache_file) if cache_file.exists() else None
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logging.info(f"? Cached tissue results: {cache_file}")
    
    def load_cached_tissue_results(self, slide_hash):
        """Load cached tissue detection results if available."""
        cache_file = self.cache_dir / f"{slide_hash}_tissue.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            logging.info(f"? Loaded cached tissue results: {cache_file}")
            return cache_data['tissue_regions'], cache_data['tissue_mask']
            
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}")
            return None

def implement_adaptive_window_sizing():
    """
    Adaptive window sizing based on tissue density.
    Dense tissue areas use larger windows, sparse areas use smaller windows.
    """
    def calculate_tissue_density(tissue_mask, region):
        """Calculate tissue density in a given region."""
        x, y, w, h = region
        region_mask = tissue_mask[y:y+h, x:x+w]
        return np.sum(region_mask > 0) / (w * h)
    
    def adaptive_window_generator(tissue_mask, base_window_size, stride):
        """Generate windows with adaptive sizing based on tissue density."""
        height, width = tissue_mask.shape
        base_w, base_h = base_window_size
        
        # Analyze tissue density in grid
        grid_size = 512  # Analyze in 512x512 grid cells
        for y in range(0, height - base_h, stride[1]):
            for x in range(0, width - base_w, stride[0]):
                # Calculate local tissue density
                density = calculate_tissue_density(tissue_mask, (x, y, grid_size, grid_size))
                
                # Adapt window size based on density
                if density > 0.8:  # High density - use larger windows
                    window_w, window_h = int(base_w * 1.5), int(base_h * 1.5)
                elif density < 0.2:  # Low density - use smaller windows
                    window_w, window_h = int(base_w * 0.7), int(base_h * 0.7)
                else:  # Medium density - use base size
                    window_w, window_h = base_w, base_h
                
                yield (x, y, window_w, window_h)
