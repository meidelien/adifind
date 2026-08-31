"""
Async I/O Integration for AdiFind
=================================

Integration layer to add async I/O capabilities to the main processing pipeline.
"""

import logging
import time
import numpy as np
import config as config
from typing import Tuple, List, Optional, Dict, Any

from .async_io import (
    AsyncImageLoader,
    FastImageIO,
    LightweightTileCache,
    AsyncBenchmarker,
    get_optimized_read_function,
    clear_all_caches
)


class AsyncImageProcessor:
    """
    Wrapper for image processing with async I/O capabilities.
    Drop-in replacement for standard image processing methods.
    """
    
    def __init__(self, image_handler, enable_async=True, cache_size_mb=64):
        self.image_handler = image_handler
        self.original_read_region = image_handler.read_region  # Store original method
        self.enable_async = enable_async
        self.async_loader = None
        self.optimized_read = None
        self.stats = {
            'total_reads': 0,
            'async_reads': 0,
            'cache_hits': 0,
            'total_time': 0.0
        }
        
        if enable_async:
            # Create async loader with proper original method
            temp_handler = type('TempHandler', (), {})()
            temp_handler.read_region = self.original_read_region
            
            self.async_loader = AsyncImageLoader(
                temp_handler, 
                prefetch_size=2, 
                max_workers=2
            )
            self.optimized_read = get_optimized_read_function(image_handler)
            logging.info(f"?? Async I/O enabled with {cache_size_mb}MB cache")
        else:
            logging.info("?? Using standard I/O (async disabled)")
    
    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]):
        """
        Read image region with optional async optimization.
        
        Args:
            location: (x, y) coordinates
            level: Pyramid level
            size: (width, height) of region
            
        Returns:
            Image region as numpy array or PIL Image
        """
        start_time = time.time()
        self.stats['total_reads'] += 1
        
        try:
            if self.enable_async and self.optimized_read:
                # Use optimized async read
                result = self.optimized_read(location, level, size)
                self.stats['async_reads'] += 1
            else:
                # Fall back to original method (not current handler method)
                result = self.original_read_region(location, level, size)
            
            self.stats['total_time'] += time.time() - start_time
            return result
            
        except Exception as e:
            logging.warning(f"Async read failed, falling back to standard: {e}")
            # Always fall back to original method
            result = self.original_read_region(location, level, size)
            self.stats['total_time'] += time.time() - start_time
            return result
    
    def read_multiple_regions(self, regions_list: List[Tuple], max_workers: int = 2):
        """
        Read multiple regions with parallel I/O.
        
        Args:
            regions_list: List of (location, level, size) tuples
            max_workers: Number of parallel workers
            
        Returns:
            List of loaded regions
        """
        if self.enable_async:
            return FastImageIO.batch_read_regions(
                self.image_handler, 
                regions_list, 
                max_workers=max_workers,
                original_read_method=self.original_read_region
            )
        else:
            # Sequential reads for fallback using original method
            results = []
            for location, level, size in regions_list:
                result = self.original_read_region(location, level, size)
                results.append(result)
            return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_reads = self.stats['total_reads']
        async_ratio = self.stats['async_reads'] / total_reads if total_reads > 0 else 0
        avg_time = self.stats['total_time'] / total_reads if total_reads > 0 else 0
        
        stats = {
            'async_enabled': self.enable_async,
            'total_reads': total_reads,
            'async_ratio': async_ratio,
            'average_read_time': avg_time,
            'total_time': self.stats['total_time']
        }
        
        if self.async_loader:
            stats['cache_stats'] = self.async_loader.get_cache_stats()
        
        return stats
    
    def clear_caches(self):
        """Clear all caches to free memory."""
        if self.async_loader:
            self.async_loader.clear_cache()
        clear_all_caches()
        logging.info("?? Async processor caches cleared")


def integrate_async_io(image_handler, config=None):
    """
    Create an async-enabled image processor for the given handler.
    
    Args:
        image_handler: Original image handler object
        config: Optional configuration dictionary
        
    Returns:
        AsyncImageProcessor instance
    """
    # Default configuration
    default_config = {
        'enable_async': True,
        'cache_size_mb': 64,
    }
    
    if config:
        default_config.update(config)
    
    processor = AsyncImageProcessor(
        image_handler=image_handler,
        enable_async=default_config['enable_async'],
        cache_size_mb=default_config['cache_size_mb']
    )
    
    return processor


def benchmark_async_performance(image_handler, num_test_regions=20):
    """
    Benchmark async I/O performance for the given image handler.
    
    Args:
        image_handler: Image handler to benchmark
        num_test_regions: Number of regions to test
        
    Returns:
        Benchmark results dictionary
    """
    # Generate test regions
    try:
        # Get image dimensions
        dimensions = image_handler.dimensions
        level_count = image_handler.level_count
        
        # Create test regions
        test_regions = []
        for i in range(num_test_regions):
            # Random location within image bounds
            x = int((i * 123) % (dimensions[0] - 512))  # Simple pseudo-random
            y = int((i * 456) % (dimensions[1] - 512))
            location = (x, y)
            level = min(i % level_count, level_count - 1)
            size = (512, 512)  # Standard tile size
            test_regions.append((location, level, size))
        
        # Run benchmark
        benchmarker = AsyncBenchmarker()
        results = benchmarker.benchmark_read_operations(image_handler, test_regions)
        benchmarker.print_benchmark_results()
        
        return results
        
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        return None


def patch_image_handler_with_async(image_handler, config=None):
    """
    Monkey-patch an image handler to use async I/O.
    
    This modifies the original object to use async capabilities
    while maintaining the same interface.
    
    Args:
        image_handler: Original image handler
        config: Optional async configuration
        
    Returns:
        Modified image handler with async capabilities
    """
    # Store original method
    original_read_region = image_handler.read_region
    
    # Create async processor
    async_processor = integrate_async_io(image_handler, config)
    
    # Replace read_region method
    def async_read_region(location, level, size):
        return async_processor.read_region(location, level, size)
    
    # Monkey patch the method
    image_handler.read_region = async_read_region
    image_handler._async_processor = async_processor  # Store reference
    image_handler._original_read_region = original_read_region  # Keep original
    
    logging.info("?? Image handler patched with async I/O capabilities")
    return image_handler


def restore_original_image_handler(image_handler):
    """
    Restore image handler to its original state (remove async patch).
    
    Args:
        image_handler: Patched image handler
        
    Returns:
        Restored image handler
    """
    if hasattr(image_handler, '_original_read_region'):
        image_handler.read_region = image_handler._original_read_region
        if hasattr(image_handler, '_async_processor'):
            image_handler._async_processor.clear_caches()
            delattr(image_handler, '_async_processor')
        delattr(image_handler, '_original_read_region')
        logging.info("?? Image handler restored to original state")
    
    return image_handler
