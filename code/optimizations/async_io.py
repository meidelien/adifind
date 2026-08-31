"""
Async I/O Optimization for AdiFind
==================================

Focused async I/O implementation for better performance without memory overhead.
Only includes safe optimizations that don't consume excessive GPU memory.
"""

import numpy as np
import threading
import queue
import config as config
from concurrent.futures import ThreadPoolExecutor
import gc
import time
import logging


class AsyncImageLoader:
    """
    Lightweight asynchronous image loading with prefetching.
    Optimized for better I/O performance without excessive memory usage.
    """
    
    def __init__(self, image_handler, prefetch_size=getattr(config, 'ASYNC_PREFETCH_SIZE', 24), max_workers=2):
        self.image_handler = image_handler
        self.original_read_region = image_handler.read_region  # Store original method
        self.prefetch_queue = queue.Queue(maxsize=prefetch_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = {}
        self.cache_size_limit = 5  # Small cache to avoid memory issues
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0
        }
        
    def prefetch_region(self, location, level, size):
        """Prefetch an image region asynchronously."""
        def load_region():
            try:
                region = self.original_read_region(location, level, size)
                return (location, level, size), region
            except Exception as e:
                logging.warning(f"Async prefetch failed for {location}: {e}")
                return (location, level, size), None
        
        future = self.executor.submit(load_region)
        return future
    
    def get_region_cached(self, location, level, size):
        """Get region with lightweight caching support."""
        cache_key = (location, level, size)
        self.stats['total_requests'] += 1
        
        # Check cache first
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Load from file using original method
        region = self.original_read_region(location, level, size)
        
        # Add to cache if there's space
        if len(self.cache) < self.cache_size_limit:
            self.cache[cache_key] = region
        elif len(self.cache) >= self.cache_size_limit:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.cache[cache_key] = region
        
        return region
    
    def clear_cache(self):
        """Clear the image cache to free memory."""
        self.cache.clear()
        gc.collect()
        
    def get_cache_stats(self):
        """Get cache performance statistics."""
        if self.stats['total_requests'] > 0:
            hit_rate = self.stats['cache_hits'] / self.stats['total_requests']
            return {
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                **self.stats
            }
        return self.stats


class FastImageIO:
    """
    Optimized image I/O operations focused on speed.
    """
    
    @staticmethod
    def fast_array_conversion(pil_image):
        """
        Faster PIL to numpy conversion.
        Avoids unnecessary copies and ensures optimal memory layout.
        """
        # Use more efficient conversion method
        if pil_image.mode == 'RGBA':
            # Convert RGBA to RGB (remove alpha channel)
            np_array = np.array(pil_image)[:, :, :3]
        elif pil_image.mode == 'RGB':
            np_array = np.array(pil_image)
        else:
            # Convert other modes to RGB first
            pil_rgb = pil_image.convert('RGB')
            np_array = np.array(pil_rgb)
        
        return np_array
    
    @staticmethod
    def batch_read_regions(image_handler, regions_list, max_workers=2, original_read_method=None):
        """
        Read multiple regions in parallel for better I/O performance.
        Conservative worker count to avoid memory pressure.
        
        Args:
            image_handler: Image handler object
            regions_list: List of (location, level, size) tuples
            max_workers: Number of parallel workers (conservative default)
            original_read_method: Original read method to use (avoids recursion)
            
        Returns:
            List of loaded regions
        """
        # Use provided original method or default to handler method
        read_method = original_read_method or image_handler.read_region
        
        def load_single_region(region_params):
            location, level, size = region_params
            try:
                return read_method(location, level, size)
            except Exception as e:
                logging.warning(f"Error loading region {location}: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            regions = list(executor.map(load_single_region, regions_list))
        
        return regions
    
    @staticmethod
    def optimize_image_format(image_array):
        """
        Optimize image format for processing speed.
        Convert to optimal dtype and ensure contiguous memory.
        """
        # Ensure contiguous memory layout
        if not image_array.flags['C_CONTIGUOUS']:
            image_array = np.ascontiguousarray(image_array)
        
        # Use uint8 for images to save memory and increase speed
        if image_array.dtype != np.uint8 and image_array.max() <= 255:
            image_array = image_array.astype(np.uint8)
        
        return image_array


class LightweightTileCache:
    """
    Lightweight LRU cache for image tiles with memory monitoring.
    """
    
    def __init__(self, max_size_mb=64):  # Much smaller default cache
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.access_order = []
        self.current_size = 0
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def _estimate_size(self, image_array):
        """Estimate memory size of image array."""
        if hasattr(image_array, 'nbytes'):
            return image_array.nbytes
        elif hasattr(image_array, '__len__'):
            # Rough estimate for PIL images
            return len(image_array.tobytes()) if hasattr(image_array, 'tobytes') else 1024
        return 1024  # Default estimate
    
    def get(self, key):
        """Get item from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            self.stats['hits'] += 1
            return self.cache[key]
        
        self.stats['misses'] += 1
        return None
    
    def put(self, key, value):
        """Add item to cache."""
        if key in self.cache:
            return
        
        # Calculate size
        item_size = self._estimate_size(value)
        
        # Remove old items if necessary
        while (self.current_size + item_size > self.max_size_bytes and 
               self.access_order):
            oldest_key = self.access_order.pop(0)
            oldest_value = self.cache.pop(oldest_key)
            self.current_size -= self._estimate_size(oldest_value)
            self.stats['evictions'] += 1
        
        # Add new item
        self.cache[key] = value
        self.access_order.append(key)
        self.current_size += item_size
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
        self.current_size = 0
    
    def get_stats(self):
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'cache_size_mb': self.current_size / (1024 * 1024),
            'num_items': len(self.cache),
            **self.stats
        }


# Global cache instance (small size for safety)
global_tile_cache = LightweightTileCache(max_size_mb=getattr(config, 'ASYNC_CACHE_SIZE_MB', 512))


def get_optimized_read_function(image_handler):
    """
    Get an optimized read function for the image handler.
    Returns a wrapper that includes lightweight caching and optimization.
    """
    # Store reference to original method to avoid recursion
    original_read_region = image_handler.read_region
    
    def optimized_read_region(location, level, size):
        # Create cache key
        cache_key = (location, level, size)
        
        # Try cache first
        cached_result = global_tile_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Read from file using ORIGINAL method (not the patched one)
        region = original_read_region(location, level, size)
        
        # Keep PIL images as PIL images to maintain compatibility
        # Only cache the original result without conversion for now
        global_tile_cache.put(cache_key, region)
        return region
    
    return optimized_read_region


class AsyncBenchmarker:
    """
    Benchmark async I/O performance against standard I/O.
    """
    
    def __init__(self):
        self.results = {}
    
    def benchmark_read_operations(self, image_handler, test_regions):
        """
        Benchmark standard vs async read operations.
        
        Args:
            image_handler: Image handler to test
            test_regions: List of (location, level, size) tuples to test
            
        Returns:
            Dictionary with benchmark results
        """
        logging.info("?? Starting I/O benchmark...")
        
        # Benchmark 1: Standard sequential reads
        start_time = time.time()
        standard_results = []
        for location, level, size in test_regions:
            try:
                region = image_handler.read_region(location, level, size)
                standard_results.append(region)
            except Exception as e:
                logging.warning(f"Standard read failed for {location}: {e}")
        
        standard_time = time.time() - start_time
        
        # Benchmark 2: Async I/O
        async_loader = AsyncImageLoader(image_handler, prefetch_size=getattr(config, 'ASYNC_PREFETCH_SIZE', 24), max_workers=getattr(config, 'ASYNC_MAX_WORKERS', 10))
        start_time = time.time()
        async_results = []
        for location, level, size in test_regions:
            try:
                region = async_loader.get_region_cached(location, level, size)
                async_results.append(region)
            except Exception as e:
                logging.warning(f"Async read failed for {location}: {e}")
        
        async_time = time.time() - start_time
        
        # Benchmark 3: Optimized read function
        optimized_read = get_optimized_read_function(image_handler)
        start_time = time.time()
        optimized_results = []
        for location, level, size in test_regions:
            try:
                region = optimized_read(location, level, size)
                optimized_results.append(region)
            except Exception as e:
                logging.warning(f"Optimized read failed for {location}: {e}")
        
        optimized_time = time.time() - start_time
        
        # Calculate results
        speedup_async = standard_time / async_time if async_time > 0 else 1.0
        speedup_optimized = standard_time / optimized_time if optimized_time > 0 else 1.0
        
        results = {
            'standard_time': standard_time,
            'async_time': async_time,
            'optimized_time': optimized_time,
            'speedup_async': speedup_async,
            'speedup_optimized': speedup_optimized,
            'regions_tested': len(test_regions),
            'async_cache_stats': async_loader.get_cache_stats(),
            'global_cache_stats': global_tile_cache.get_stats()
        }
        
        self.results = results
        return results
    
    def print_benchmark_results(self):
        """Print formatted benchmark results."""
        if not self.results:
            print("? No benchmark results available")
            return
        
        r = self.results
        
        print("\n?? **Async I/O Benchmark Results**")
        print("=" * 50)
        print(f"?? Regions tested: {r['regions_tested']}")
        print(f"??  Standard I/O time: {r['standard_time']:.2f}s")
        print(f"? Async I/O time: {r['async_time']:.2f}s")
        print(f"?? Optimized I/O time: {r['optimized_time']:.2f}s")
        print(f"?? Async speedup: {r['speedup_async']:.2f}x")
        print(f"?? Optimized speedup: {r['speedup_optimized']:.2f}x")
        
        # Cache statistics
        if 'async_cache_stats' in r:
            cache_stats = r['async_cache_stats']
            print(f"?? Async cache hit rate: {cache_stats.get('hit_rate', 0):.1%}")
        
        if 'global_cache_stats' in r:
            global_stats = r['global_cache_stats']
            print(f"?? Global cache hit rate: {global_stats.get('hit_rate', 0):.1%}")
            print(f"?? Cache size: {global_stats.get('cache_size_mb', 0):.1f}MB")
        
        print("=" * 50)


def clear_all_caches():
    """Clear all caches to free memory."""
    global_tile_cache.clear()
    gc.collect()
    logging.info("?? All async I/O caches cleared")
