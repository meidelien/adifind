#!/usr/bin/env python3
"""
Memory Streaming Optimization
=============================

Implements streaming processing to handle large slides without
loading entire masks into memory simultaneously.
"""

import numpy as np
import logging
import gc
from typing import Iterator, Tuple
import h5py
import tempfile

class StreamingMaskProcessor:
    """Processes masks in chunks to reduce memory footprint."""
    
    def __init__(self, image_shape, chunk_size=(4096, 4096)):
        self.image_shape = image_shape
        self.chunk_size = chunk_size
        self.temp_file = None
        self.h5_file = None
        
        # Create temporary HDF5 file for large mask storage
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        self.h5_file = h5py.File(self.temp_file.name, 'w')
        
        # Create dataset for streaming mask
        self.mask_dataset = self.h5_file.create_dataset(
            'mask', 
            shape=image_shape, 
            dtype=np.uint32,
            chunks=self.chunk_size,
            compression='gzip'
        )
        
        logging.info(f"???  Created streaming mask: {image_shape} with chunks {chunk_size}")
    
    def get_chunk_iterator(self) -> Iterator[Tuple[slice, slice]]:
        """Generate chunk slices for streaming processing."""
        height, width = self.image_shape
        chunk_h, chunk_w = self.chunk_size
        
        for y in range(0, height, chunk_h):
            for x in range(0, width, chunk_w):
                y_end = min(y + chunk_h, height)
                x_end = min(x + chunk_w, width)
                yield slice(y, y_end), slice(x, x_end)
    
    def update_chunk(self, y_slice, x_slice, chunk_data):
        """Update a specific chunk of the mask."""
        self.mask_dataset[y_slice, x_slice] = chunk_data
    
    def get_chunk(self, y_slice, x_slice):
        """Retrieve a specific chunk of the mask."""
        return self.mask_dataset[y_slice, x_slice]
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.h5_file:
            self.h5_file.close()
        if self.temp_file:
            import os
            os.unlink(self.temp_file.name)

class MemoryEfficientUnionFind:
    """Union-Find implementation optimized for large-scale streaming processing."""
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find_with_compression(self, x):
        """Find with path compression - optimized for large datasets."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        
        # Path compression with iterative approach (stack-safe)
        path = []
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        
        # Compress path
        for node in path:
            self.parent[node] = x
        
        return x
    
    def union_by_rank(self, x, y):
        """Union by rank for balanced trees."""
        root_x = self.find_with_compression(x)
        root_y = self.find_with_compression(y)
        
        if root_x == root_y:
            return
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

def implement_streaming_label_mapping():
    """
    Implement label mapping that works with streaming chunks
    to avoid loading entire mask into memory.
    """
    def stream_process_chunks(streaming_processor, union_find_manager):
        """Process mask chunks in streaming fashion."""
        total_processed = 0
        
        for y_slice, x_slice in streaming_processor.get_chunk_iterator():
            # Load chunk
            chunk = streaming_processor.get_chunk(y_slice, x_slice)
            
            # Process unique labels in chunk
            unique_labels = np.unique(chunk)
            unique_labels = unique_labels[unique_labels != 0]
            
            # Build label mapping for this chunk
            chunk_mapping = {}
            for label in unique_labels:
                root_label = union_find_manager.find_with_compression(label)
                chunk_mapping[label] = root_label
            
            # Apply mapping to chunk
            if chunk_mapping:
                # Vectorized label mapping
                max_label = max(chunk_mapping.keys())
                lookup = np.arange(max_label + 1, dtype=chunk.dtype)
                for old_label, new_label in chunk_mapping.items():
                    lookup[old_label] = new_label
                
                # Apply and save chunk
                chunk_mapped = lookup[chunk]
                streaming_processor.update_chunk(y_slice, x_slice, chunk_mapped)
            
            total_processed += 1
            if total_processed % 10 == 0:
                logging.info(f"?? Processed {total_processed} chunks")
                gc.collect()  # Periodic cleanup

class OptimizedImageRegionCache:
    """Cache frequently accessed image regions to reduce I/O."""
    
    def __init__(self, max_cache_size_mb=512):
        self.cache = {}
        self.access_count = {}
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.current_cache_size = 0
    
    def get_region_key(self, x, y, width, height, level):
        """Generate cache key for image region."""
        return f"{x}_{y}_{width}_{height}_{level}"
    
    def cache_region(self, key, region_data):
        """Cache an image region with LRU eviction."""
        region_size = region_data.nbytes
        
        # Evict if necessary
        while self.current_cache_size + region_size > self.max_cache_size and self.cache:
            # Find least recently used item
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            evicted_size = self.cache[lru_key].nbytes
            
            del self.cache[lru_key]
            del self.access_count[lru_key]
            self.current_cache_size -= evicted_size
        
        # Cache new region
        self.cache[key] = region_data.copy()
        self.access_count[key] = 0
        self.current_cache_size += region_size
    
    def get_cached_region(self, key):
        """Retrieve cached region if available."""
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
