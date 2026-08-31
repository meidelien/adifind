#!/usr/bin/env python3
"""
GPU-Accelerated Label Mapping
"""

import logging
import time
import numpy as np
import math
from typing import Dict, Tuple


def _safe_int_prod(shape):
    p = 1
    for x in shape:
        p *= int(x)
    return int(p)

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Import config
try:
    from config import config
except ImportError:
    # Fallback for testing
    class MockConfig:
        ENABLE_PROFILING = False
    config = MockConfig()

class GPULabelMapper:
    """GPU-accelerated label mapping with intelligent memory management."""
    
    def __init__(self, max_gpu_memory_gb=30):
        """
        Initialize GPU label mapper.
        
        Args:
            max_gpu_memory_gb: Maximum GPU memory to use (leave 2GB free for other operations)
        """
        self.gpu_available = CUPY_AVAILABLE and cp.cuda.is_available()
        self.max_gpu_memory_bytes = max_gpu_memory_gb * 1024 * 1024 * 1024
        
        if self.gpu_available:
            # Get actual GPU memory
            total_memory = cp.cuda.Device().mem_info[1]  # Total memory
            self.max_gpu_memory_bytes = min(self.max_gpu_memory_bytes, total_memory * 0.9)  # Use 90% max
            logging.info(f"?? GPULabelMapper initialized - using up to {self.max_gpu_memory_bytes/1024/1024/1024:.1f}GB GPU memory")
        else:
            logging.warning("???  GPU not available for label mapping - using CPU fallback")
    
    def estimate_memory_requirements(self, mask_shape, mapping_size):
        """Estimate GPU memory requirements for mask and lookup table."""
        # Mask memory (uint32)
        mask_memory = np.prod(mask_shape) * 4  # 4 bytes per uint32
        
        # Lookup table memory (uint32)
        max_label = max(mapping_size.keys()) if mapping_size else 0
        lookup_memory = (max_label + 1) * 4  # 4 bytes per uint32
        
        # Working memory (conservative estimate)
        working_memory = mask_memory * 0.5  # 50% overhead
        
        total_required = mask_memory + lookup_memory + working_memory
        return total_required, mask_memory, lookup_memory
    
    def calculate_optimal_chunks(self, mask_shape, mapping_dict):
        """Calculate optimal chunk size based on available GPU memory."""
        total_required, mask_memory, lookup_memory = self.estimate_memory_requirements(mask_shape, mapping_dict)
        
        if total_required <= self.max_gpu_memory_bytes:
            # Can process entire mask at once
            return [(0, mask_shape[0])], 1
        
        # Calculate how many chunks we need
        available_for_chunks = self.max_gpu_memory_bytes - lookup_memory
        chunk_memory = available_for_chunks * 0.8  # Conservative
        
        # Prevent division by zero
        if mask_shape[1] == 0:
            return [(0, mask_shape[0])], 1
        
        rows_per_chunk = int((chunk_memory / 4) / mask_shape[1])  # 4 bytes per pixel
        rows_per_chunk = max(100, rows_per_chunk)  # Minimum 100 rows per chunk
        
        # Create chunk boundaries
        chunks = []
        for start_row in range(0, mask_shape[0], rows_per_chunk):
            end_row = min(start_row + rows_per_chunk, mask_shape[0])
            chunks.append((start_row, end_row))
        
        logging.info(f"?? GPU memory analysis:")
        logging.info(f"   ???  Total required: {total_required/1024/1024/1024:.2f}GB")
        logging.info(f"   ?? Processing in {len(chunks)} chunks of ~{rows_per_chunk} rows each")
        
        return chunks, len(chunks)
    
    def apply_gpu_label_mapping_chunked(self, mask, mapping_dict):
        """
        Apply label mapping using GPU with automatic chunking for memory management.
        
        Args:
            mask: Input mask (numpy array)
            mapping_dict: Dictionary mapping old labels to new labels
            
        Returns:
            numpy array: Label-mapped mask
        """
        if not self.gpu_available or not mapping_dict:
            return self._cpu_fallback(mask, mapping_dict)
        
        try:
            # Calculate optimal chunking strategy
            chunks, num_chunks = self.calculate_optimal_chunks(mask.shape, mapping_dict)
            
            # Create lookup table on GPU (stays resident for all chunks)
            max_label = max(max(mapping_dict.keys()), max(mapping_dict.values()))
            lookup_cpu = np.arange(max_label + 1, dtype=mask.dtype)
            
            # Build mapping
            for old_label, new_label in mapping_dict.items():
                lookup_cpu[old_label] = new_label
            
            # Transfer lookup table to GPU once
            lookup_gpu = cp.asarray(lookup_cpu)
            
            if config.ENABLE_PROFILING:
                total_start = time.time()
                logging.info(f"?? Starting GPU label mapping with {num_chunks} chunks...")
            
            # Process mask in chunks
            result_mask = mask.copy()
            
            for chunk_idx, (start_row, end_row) in enumerate(chunks):
                if config.ENABLE_PROFILING:
                    chunk_start = time.time()
                
                # Extract chunk
                mask_chunk = mask[start_row:end_row, :]
                
                # Transfer chunk to GPU
                mask_chunk_gpu = cp.asarray(mask_chunk)
                
                # Apply mapping on GPU (vectorized operation)
                mapped_chunk_gpu = lookup_gpu[mask_chunk_gpu]
                
                # Transfer result back to CPU
                result_mask[start_row:end_row, :] = cp.asnumpy(mapped_chunk_gpu)
                
                # Cleanup GPU memory for this chunk
                del mask_chunk_gpu, mapped_chunk_gpu
                
                if config.ENABLE_PROFILING:
                    chunk_time = time.time() - chunk_start
                    logging.info(f"   ? Chunk {chunk_idx+1}/{num_chunks} processed: {chunk_time:.3f}s")
                    
                    # Log memory usage periodically
                    if chunk_idx % 5 == 0:
                        free_memory, total_memory = cp.cuda.Device().mem_info
                        used_memory = total_memory - free_memory
                        logging.info(f"   ?? GPU memory: {used_memory/1024/1024/1024:.1f}GB/{total_memory/1024/1024/1024:.1f}GB used")
            
            # Final GPU cleanup
            del lookup_gpu
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
            if config.ENABLE_PROFILING:
                total_time = time.time() - total_start
                logging.info(f"? GPU label mapping completed: {total_time:.3f}s")
                logging.info(f"   ?? Processed {len(mapping_dict)} mappings on {mask.size:,} pixels")
                # Prevent division by zero
                if total_time > 0:
                    performance = mask.size / total_time / 1_000_000  # Millions of pixels per second
                    logging.info(f"   ? Performance: {performance:.1f} million pixels/second")
                else:
                    logging.info(f"   ? Performance: instant processing")
            
            return result_mask
            
        except Exception as e:
            logging.warning(f"GPU label mapping failed, falling back to CPU: {e}")
            return self._cpu_fallback(mask, mapping_dict)
    
    def _cpu_fallback(self, mask, mapping_dict):
        """CPU fallback for label mapping."""
        if not mapping_dict:
            return mask
        
        # Use the existing memory-efficient CPU implementation
        max_label = max(max(mapping_dict.keys()), max(mapping_dict.values()))
        lookup = np.arange(max_label + 1, dtype=mask.dtype)
        
        for old_label, new_label in mapping_dict.items():
            lookup[old_label] = new_label
        
        return lookup[mask]

class GPUUnionFind:
    """GPU-accelerated Union-Find operations for massive speedup."""
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.gpu_available = CUPY_AVAILABLE and cp.cuda.is_available()
    
    def gpu_build_label_mapping(self, unique_labels, parent_dict):
        """
        Build label mapping on GPU for massive arrays.
        
        Args:
            unique_labels: Array of unique labels to process
            parent_dict: Parent dictionary from union-find
            
        Returns:
            dict: Label mapping dictionary
        """
        # Safety checks
        if len(unique_labels) == 0:
            logging.info("?? No labels to process - returning empty mapping")
            return {}
        
        if not parent_dict:
            logging.info("?? No parent dictionary - creating identity mapping")
            return {label: label for label in unique_labels}
        
        if not self.gpu_available or len(unique_labels) < 1000:
            # Use CPU for small datasets
            return self._cpu_build_mapping(unique_labels, parent_dict)
        
        try:
            if config.ENABLE_PROFILING:
                mapping_start = time.time()
                logging.info(f"?? Building label mapping on GPU for {len(unique_labels)} labels...")
            
            # Convert unique_labels to GPU
            labels_gpu = cp.asarray(unique_labels)
            
            # Create result array on GPU
            mapping_gpu = cp.zeros_like(labels_gpu, dtype=cp.uint32)
            
            # Process labels in batches to find roots
            batch_size = 100000  # Process 100k labels at a time
            label_mapping = {}
            
            for i in range(0, len(unique_labels), batch_size):
                end_idx = min(i + batch_size, len(unique_labels))
                batch_labels = unique_labels[i:end_idx]
                
                # Find roots for this batch (CPU operation - union-find path compression)
                batch_roots = []
                for label in batch_labels:
                    root = self._find_root_cpu(label, parent_dict)
                    batch_roots.append(root)
                    label_mapping[label] = root
                
                if config.ENABLE_PROFILING and i % (batch_size * 10) == 0:
                    logging.info(f"   ?? Processed {end_idx}/{len(unique_labels)} labels...")
            
            if config.ENABLE_PROFILING:
                mapping_time = time.time() - mapping_start
                logging.info(f"? GPU label mapping built: {mapping_time:.3f}s")
                # Prevent division by zero
                if mapping_time > 0:
                    performance = len(unique_labels) / mapping_time
                    logging.info(f"   ?? Performance: {performance:.0f} labels/second")
                else:
                    logging.info(f"   ?? Performance: instant processing")
            
            return label_mapping
            
        except Exception as e:
            logging.warning(f"GPU label mapping build failed, using CPU: {e}")
            return self._cpu_build_mapping(unique_labels, parent_dict)
    
    def _find_root_cpu(self, label, parent_dict):
        """CPU-based root finding with path compression."""
        if label not in parent_dict:
            return label
        
        # Path compression
        path = []
        current = label
        while current in parent_dict and parent_dict[current] != current:
            path.append(current)
            current = parent_dict[current]
        
        # Compress path
        for node in path:
            parent_dict[node] = current
        
        return current
    
    def _cpu_build_mapping(self, unique_labels, parent_dict):
        """CPU fallback for building label mapping."""
        if len(unique_labels) == 0:
            return {}
        
        if not parent_dict:
            return {label: label for label in unique_labels}
            
        label_mapping = {}
        for label in unique_labels:
            root = self._find_root_cpu(label, parent_dict)
            label_mapping[label] = root
        return label_mapping

def estimate_gpu_memory_for_mask(mask_shape):
    """Estimate GPU memory requirements for a mask."""
    # uint32 mask: 4 bytes per pixel
    mask_memory_gb = (np.prod(mask_shape) * 4) / (1024**3)
    return mask_memory_gb

def can_fit_on_gpu(mask_shape, available_memory_gb=30):
    """Check if mask can fit on GPU with available memory."""
    required_gb = estimate_gpu_memory_for_mask(mask_shape)
    return required_gb <= available_memory_gb

# Integration function for existing codebase
def apply_gpu_optimized_label_mapping(mask, mapping_dict, enable_profiling=False):
    import logging, numpy as np
    # Try to read config for budget and debug flag
    try:
        from . import config as _cfg
    except Exception:
        class _cfg: pass
        setattr(_cfg, 'GPU_MEMORY_LIMIT_GB', 24)
        setattr(_cfg, 'LABEL_MAPPING_DEBUG', False)

    budget_gb = getattr(_cfg, 'GPU_MEMORY_LIMIT_GB', 24)
    gpu_mapper = GPULabelMapper(max_gpu_memory_gb=budget_gb)

    # Estimate and optionally log details
    try:
        total_required, mask_bytes, lut_bytes = gpu_mapper.estimate_memory_requirements(mask.shape, mapping_dict)
        if getattr(_cfg, 'LABEL_MAPPING_DEBUG', False):
            def _fmt_bytes(b):
                b = float(b)
                for u in ['B','KB','MB','GB','TB']:
                    if b < 1024.0: return f"{b:.1f} {u}"
                    b /= 1024.0
                return f"{b:.1f} PB"
            pixels = _safe_int_prod(mask.shape)
            logging.info(
                "GPU budget check :: pixels=%d mask=%s LUT=%s total=%s cap=%s",
                int(pixels), _fmt_bytes(mask_bytes), _fmt_bytes(lut_bytes),
                _fmt_bytes(total_required), _fmt_bytes(gpu_mapper.max_gpu_memory_bytes)
            )
        if int(total_required) > int(gpu_mapper.max_gpu_memory_bytes):
            logging.info("???  Label mapping exceeds GPU budget; using CPU fallback")
            return _cpu_label_mapping_fallback(mask, mapping_dict)
    except Exception as e:
        logging.warning(f"??  GPU budget check failed ({e}); defaulting to CPU")
        return _cpu_label_mapping_fallback(mask, mapping_dict)

    # Try single-pass GPU mapping first; if it fails, try chunked; else CPU fallback
    try:
        result = _apply_gpu_label_mapping_single(mask, mapping_dict, profile=enable_profiling)
        return result
    except Exception as e_single:
        logging.warning(f"GPU single-pass failed ({e_single}); trying chunked GPU mapping")
        try:
            chunked = _gpu_chunked_label_mapping(mask, mapping_dict, gpu_mapper, enable_profiling)
            if chunked is not None:
                return chunked
        except Exception as e_chunk:
            logging.warning(f"GPU chunked path failed ({e_chunk}); falling back to CPU")
        return _cpu_label_mapping_fallback(mask, mapping_dict)
def _cpu_label_mapping_fallback(mask, mapping_dict):
    """CPU fallback - identical to existing apply_label_mapping_memory_efficient."""
    max_label = max(max(mapping_dict.keys()), max(mapping_dict.values()))
    lookup = np.arange(max_label + 1, dtype=mask.dtype)
    
    for old_label, new_label in mapping_dict.items():
        lookup[old_label] = new_label
    
    return lookup[mask]

# ================================================================
# MEMORY ESTIMATION UTILITIES
# ================================================================

def estimate_chunk_size_for_gpu_memory(mask_shape, available_memory_gb, mapping_size):
    """
    Calculate optimal chunk size for GPU processing.
    
    Args:
        mask_shape: Shape of the full mask
        available_memory_gb: Available GPU memory in GB
        mapping_size: Number of mappings
        
    Returns:
        int: Optimal number of rows per chunk
    """
    height, width = mask_shape
    available_bytes = available_memory_gb * 1024 * 1024 * 1024
    
    # Memory for lookup table (constant across chunks)
    max_label = mapping_size if isinstance(mapping_size, int) else len(mapping_size)
    lookup_memory = max_label * 4  # uint32
    
    # Available memory for chunks
    available_for_chunks = available_bytes - lookup_memory
    available_for_chunks *= 0.7  # Safety margin
    
    # Memory per row
    memory_per_row = width * 4 * 2  # Input + output mask rows
    
    # Calculate rows per chunk
    rows_per_chunk = int(available_for_chunks / memory_per_row)
    rows_per_chunk = max(100, min(rows_per_chunk, height))  # Min 100 rows, max full height
    
    return rows_per_chunk

def log_memory_savings(original_time, gpu_time, mask_size, mapping_size):
    """Log performance improvements from GPU acceleration."""
    if original_time > 0 and gpu_time > 0:
        speedup = original_time / gpu_time
        throughput = mask_size / gpu_time / 1_000_000  # Million pixels per second
        
        logging.info(f"?? GPU Label Mapping Performance:")
        logging.info(f"   ? Speedup: {speedup:.1f}x faster than CPU")
        logging.info(f"   ?? Throughput: {throughput:.1f} million pixels/second")
        logging.info(f"   ?? Processed: {mapping_size:,} mappings on {mask_size:,} pixels")
    else:
        logging.info(f"?? GPU Label Mapping completed with {mapping_size:,} mappings on {mask_size:,} pixels")

# ================================================================
# ADVANCED GPU OPTIMIZATIONS
# ================================================================

class AdvancedGPUOperations:
    """Advanced GPU operations for ultimate performance with RTX 5090."""
    
    def __init__(self):
        self.gpu_available = CUPY_AVAILABLE and cp.cuda.is_available()
    
    def gpu_parallel_chunk_processing(self, mask, mapping_dict, num_streams=4):
        """
        Use CUDA streams for parallel chunk processing on high-end GPUs.
        Perfect for RTX 5090 with massive parallel processing capability.
        """
        if not self.gpu_available:
            return _cpu_label_mapping_fallback(mask, mapping_dict)
        
        try:
            # Create multiple CUDA streams for parallel processing
            streams = [cp.cuda.Stream() for _ in range(num_streams)]
            
            # Calculate chunks - prevent division by zero
            height = mask.shape[0]
            if num_streams <= 0:
                num_streams = 1
            chunk_size = height // num_streams
            if chunk_size <= 0:
                chunk_size = height
            
            # Create lookup table once
            max_label = max(max(mapping_dict.keys()), max(mapping_dict.values()))
            lookup_cpu = np.arange(max_label + 1, dtype=mask.dtype)
            for old_label, new_label in mapping_dict.items():
                lookup_cpu[old_label] = new_label
            
            lookup_gpu = cp.asarray(lookup_cpu)
            
            # Process chunks in parallel streams
            result_chunks = []
            for stream_idx, stream in enumerate(streams):
                with stream:
                    start_row = stream_idx * chunk_size
                    end_row = (stream_idx + 1) * chunk_size if stream_idx < num_streams - 1 else height
                    
                    # Process chunk
                    chunk = mask[start_row:end_row, :]
                    chunk_gpu = cp.asarray(chunk)
                    mapped_gpu = lookup_gpu[chunk_gpu]
                    result_chunk = cp.asnumpy(mapped_gpu)
                    
                    result_chunks.append((start_row, end_row, result_chunk))
            
            # Synchronize all streams
            for stream in streams:
                stream.synchronize()
            
            # Reconstruct result
            result_mask = mask.copy()
            for start_row, end_row, chunk_result in result_chunks:
                result_mask[start_row:end_row, :] = chunk_result
            
            # Cleanup
            del lookup_gpu
            for stream in streams:
                del stream
            cp.get_default_memory_pool().free_all_blocks()
            
            logging.info(f"? Parallel GPU processing with {num_streams} streams completed")
            return result_mask
            
        except Exception as e:
            logging.warning(f"Parallel GPU processing failed: {e}")
            return _cpu_label_mapping_fallback(mask, mapping_dict)


def estimate_memory_requirements(self, mask_shape, mapping_dict):
    import numpy as np
    pixels = _safe_int_prod(mask_shape)
    itemsize = np.dtype(np.uint32).itemsize
    mask_bytes = int(pixels) * int(itemsize)
    if mapping_dict:
        try:
            max_key = max(int(k) for k in mapping_dict.keys())
            max_val = max(int(v) for v in mapping_dict.values())
            max_label = max(max_key, max_val)
        except Exception:
            max_label = 0
    else:
        max_label = 0
    lut_elems = int(max_label) + 1
    lut_bytes = lut_elems * itemsize
    working = int((mask_bytes + lut_bytes) * 0.5)
    total = int(mask_bytes + lut_bytes + working)
    return total, mask_bytes, lut_bytes






def _gpu_chunked_label_mapping(mask, mapping_dict, gpu_mapper, enable_profiling=False):
    """Stream the label mapping in GPU-sized chunks to avoid OOM.
    Requires that the LUT fits in the budget; otherwise return None to signal CPU fallback.
    """
    import numpy as np
    try:
        import cupy as cp
    except Exception:
        return None

    # LUT sizing
    if mapping_dict:
        max_key = max(int(k) for k in mapping_dict.keys())
        max_val = max(int(v) for v in mapping_dict.values())
        max_label = max(max_key, max_val)
    else:
        max_label = 0
    lut_elems = int(max_label) + 1
    if lut_elems <= 0:
        return mask

    lut_bytes = lut_elems * np.dtype(np.uint32).itemsize
    if lut_bytes >= gpu_mapper.max_gpu_memory_bytes // 2:
        # LUT too large; better to fall back to CPU
        return None

    # Diagnostics
    try:
        from . import config as _cfg
    except Exception:
        class _cfg: pass
        setattr(_cfg, 'LABEL_MAPPING_DEBUG', False)

    if getattr(_cfg, 'LABEL_MAPPING_DEBUG', False):
        import logging
        logging.info(
            "GPU chunked :: lut_elems=%d lut_bytes=%d budget_bytes=%d",
            lut_elems, lut_bytes, gpu_mapper.max_gpu_memory_bytes
        )
        try:
            free_b, total_b = cp.cuda.Device().mem_info
            logging.info("GPU mem info :: free=%d total=%d", free_b, total_b)
        except Exception:
            pass

    # Build LUT on GPU
    cp.cuda.Device().synchronize()
    d_lut = cp.arange(lut_elems, dtype=cp.uint32)
    if mapping_dict:
        keys = np.fromiter((int(k) for k in mapping_dict.keys()), dtype=np.int64, count=len(mapping_dict))
        vals = np.fromiter((int(v) for v in mapping_dict.values()), dtype=np.int64, count=len(mapping_dict))
        d_lut[keys.astype(cp.int64)] = vals.astype(cp.uint32)

    flat = mask.reshape(-1)
    out = np.empty_like(flat)
    itemsize = flat.dtype.itemsize

    # Leave room for LUT and overhead; use at most ~1/3 of budget for the chunk.
    chunk_budget = max(64 * 1024 * 1024, int(gpu_mapper.max_gpu_memory_bytes - lut_bytes) // 3)
    elems_per_chunk = max(1, chunk_budget // itemsize)

    for start in range(0, flat.size, elems_per_chunk):
        end = min(flat.size, start + elems_per_chunk)
        h_chunk = flat[start:end]
        d_chunk = cp.asarray(h_chunk, order='C')
        d_mapped = d_lut[d_chunk]
        out[start:end] = cp.asnumpy(d_mapped)

        if getattr(_cfg, 'LABEL_MAPPING_DEBUG', False):
            import logging
            logging.debug("GPU chunk [%d:%d] elems=%d", start, end, end - start)

        del d_chunk, d_mapped
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize()

    return out.reshape(mask.shape)
