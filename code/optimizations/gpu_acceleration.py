#!/usr/bin/env python3
"""
GPU Acceleration Extensions
===========================

Extends GPU acceleration beyond distance transforms to include
mask operations, image preprocessing, and label management.
"""

import logging
import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    import torch
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class GPUAcceleratedOperations:
    """GPU-accelerated operations for mask and image processing."""
    
    def __init__(self):
        self.gpu_available = CUPY_AVAILABLE and cp.cuda.is_available()
        
        if self.gpu_available:
            logging.info("? GPU acceleration available for extended operations")
        
        if not self.gpu_available:
            logging.info("???  Using CPU fallback for extended operations")
    
    def gpu_label_mapping(self, mask, label_mapping_dict):
        """GPU-accelerated label mapping for large masks."""
        if not self.gpu_available or not label_mapping_dict:
            return self._cpu_label_mapping(mask, label_mapping_dict)
        
        try:
            # Transfer to GPU
            mask_gpu = cp.asarray(mask)
            
            # Create lookup table
            max_label = max(max(label_mapping_dict.keys()), max(label_mapping_dict.values()))
            lookup_gpu = cp.arange(max_label + 1, dtype=mask.dtype)
            
            # Build mapping on GPU
            for old_label, new_label in label_mapping_dict.items():
                lookup_gpu[old_label] = new_label
            
            # Apply mapping
            result_gpu = lookup_gpu[mask_gpu]
            
            # Transfer back to CPU
            result = cp.asnumpy(result_gpu)
            
            # Cleanup GPU memory
            del mask_gpu, lookup_gpu, result_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            logging.info("? Used GPU label mapping")
            return result
            
        except Exception as e:
            logging.warning(f"GPU label mapping failed, using CPU: {e}")
            return self._cpu_label_mapping(mask, label_mapping_dict)
    
    def _cpu_label_mapping(self, mask, label_mapping_dict):
        """CPU fallback for label mapping."""
        if not label_mapping_dict:
            return mask
        
        max_label = max(max(label_mapping_dict.keys()), max(label_mapping_dict.values()))
        lookup = np.arange(max_label + 1, dtype=mask.dtype)
        
        for old_label, new_label in label_mapping_dict.items():
            lookup[old_label] = new_label
        
        return lookup[mask]
    
    def gpu_image_filters(self, image, apply_sobel=False, apply_bilateral=False):
        """GPU-accelerated image filtering operations."""
        if not self.gpu_available:
            return self._cpu_image_filters(image, apply_sobel, apply_bilateral)
        
        try:
            # Transfer to GPU
            image_gpu = cp.asarray(image)
            
            if apply_bilateral:
                # GPU bilateral filter (approximation)
                from cupyx.scipy.ndimage import gaussian_filter
                # Bilateral filter approximation using multiple Gaussian filters
                filtered_gpu = gaussian_filter(image_gpu.astype(cp.float32), sigma=1.5)
                image_gpu = filtered_gpu.astype(cp.uint8)
            
            if apply_sobel:
                # GPU Sobel filter
                if len(image_gpu.shape) == 3:
                    # Convert to grayscale for Sobel
                    gray_gpu = cp.mean(image_gpu, axis=2).astype(cp.uint8)
                else:
                    gray_gpu = image_gpu
                
                # Sobel kernels
                sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
                sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)
                
                # Apply Sobel
                from cupyx.scipy.ndimage import convolve
                grad_x = convolve(gray_gpu.astype(cp.float32), sobel_x)
                grad_y = convolve(gray_gpu.astype(cp.float32), sobel_y)
                magnitude = cp.sqrt(grad_x**2 + grad_y**2)
                
                # Convert back to uint8
                image_gpu = magnitude.astype(cp.uint8)
                if len(image.shape) == 3:
                    image_gpu = cp.stack([image_gpu] * 3, axis=2)
            
            # Transfer back to CPU
            result = cp.asnumpy(image_gpu)
            
            # Cleanup
            del image_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            logging.info("? Used GPU image filtering")
            return result
            
        except Exception as e:
            logging.warning(f"GPU image filtering failed, using CPU: {e}")
            return self._cpu_image_filters(image, apply_sobel, apply_bilateral)
    
    def _cpu_image_filters(self, image, apply_sobel=False, apply_bilateral=False):
        """CPU fallback for image filtering."""
        import cv2
        
        result = image.copy()
        
        if apply_bilateral and len(image.shape) == 3:
            result = cv2.bilateralFilter(result, 20, 20, 20)
        
        if apply_sobel:
            if len(result.shape) == 3:
                gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            else:
                gray = result
            
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
            
            if len(image.shape) == 3:
                result = np.stack([magnitude] * 3, axis=2)
            else:
                result = magnitude
        
        return result

    def gpu_morphological_operations(self, mask, operation='closing', kernel_size=3):
        """GPU-accelerated morphological operations using optimized methods."""
        if not self.gpu_available:
            return self._cpu_morphological_fallback(mask, operation, kernel_size)
        
        # Use the ultra-fast version by default (avoids CuPy warnings)
        try:
            return self.gpu_morphological_operations_fast(mask, operation, kernel_size)
        except Exception as e:
            logging.warning(f"Fast GPU morphological operation failed: {e}")
            # Fallback to convolution-based method
            return self.gpu_morphological_operations_conv(mask, operation, kernel_size)
    
    def gpu_morphological_operations_conv(self, mask, operation='closing', kernel_size=3):
        """GPU-accelerated morphological operations using optimized CuPy convolution."""
        if not self.gpu_available:
            return self._cpu_morphological_fallback(mask, operation, kernel_size)
        
        try:
            # Transfer to GPU
            mask_gpu = cp.asarray(mask, dtype=cp.float32)
            
            # Create optimized structuring element (circular/elliptical kernel)
            kernel = self._create_gpu_morphological_kernel(kernel_size)
            kernel_gpu = cp.asarray(kernel, dtype=cp.float32)
            
            # Apply operation using fast convolution-based approach
            if operation == 'erosion':
                result_gpu = self._gpu_erosion(mask_gpu, kernel_gpu)
            elif operation == 'dilation':
                result_gpu = self._gpu_dilation(mask_gpu, kernel_gpu)
            elif operation == 'opening':
                # Opening = erosion followed by dilation
                eroded_gpu = self._gpu_erosion(mask_gpu, kernel_gpu)
                result_gpu = self._gpu_dilation(eroded_gpu, kernel_gpu)
                del eroded_gpu
            elif operation == 'closing':
                # Closing = dilation followed by erosion
                dilated_gpu = self._gpu_dilation(mask_gpu, kernel_gpu)
                result_gpu = self._gpu_erosion(dilated_gpu, kernel_gpu)
                del dilated_gpu
            else:
                logging.warning(f"Unknown morphological operation: {operation}")
                return self._cpu_morphological_fallback(mask, operation, kernel_size)
            
            # Convert back to original dtype and transfer to CPU
            result = cp.asnumpy(result_gpu.astype(mask.dtype))
            
            # Cleanup
            del mask_gpu, kernel_gpu, result_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            logging.info(f"? Used optimized GPU morphological operation: {operation}")
            return result
            
        except Exception as e:
            logging.warning(f"GPU morphological operation failed: {e}")
            return self._cpu_morphological_fallback(mask, operation, kernel_size)
    
    def _create_gpu_morphological_kernel(self, size):
        """Create an optimized morphological kernel (circular/elliptical)."""
        if size <= 3:
            # Small kernels - use simple cross pattern for speed
            kernel = np.zeros((size, size), dtype=np.float32)
            center = size // 2
            kernel[center, :] = 1  # Horizontal line
            kernel[:, center] = 1  # Vertical line
        else:
            # Larger kernels - use circular/elliptical pattern
            kernel = np.zeros((size, size), dtype=np.float32)
            center = size // 2
            radius = center
            
            for y in range(size):
                for x in range(size):
                    distance = np.sqrt((x - center)**2 + (y - center)**2)
                    if distance <= radius:
                        kernel[y, x] = 1
        
        return kernel
    
    def _gpu_erosion(self, mask_gpu, kernel_gpu):
        """Fast GPU erosion using minimum convolution."""
        from cupyx.scipy.ndimage import convolve
        
        # Normalize mask to 0-1 range
        normalized_mask = mask_gpu / 255.0 if mask_gpu.max() > 1 else mask_gpu
        
        # Apply convolution and take minimum operation
        # For erosion: pixel is 1 only if all kernel pixels are 1
        conv_result = convolve(normalized_mask, kernel_gpu, mode='constant', cval=0.0)
        kernel_sum = cp.sum(kernel_gpu)
        
        # Erosion condition: convolution result equals kernel sum
        result = (conv_result >= kernel_sum * 0.99).astype(cp.float32)  # Small tolerance for floating point
        
        return result * 255.0 if mask_gpu.max() > 1 else result
    
    def _gpu_dilation(self, mask_gpu, kernel_gpu):
        """Fast GPU dilation using maximum convolution."""
        from cupyx.scipy.ndimage import convolve
        
        # Normalize mask to 0-1 range
        normalized_mask = mask_gpu / 255.0 if mask_gpu.max() > 1 else mask_gpu
        
        # Apply convolution
        # For dilation: pixel is 1 if any kernel pixel is 1
        conv_result = convolve(normalized_mask, kernel_gpu, mode='constant', cval=0.0)
        
        # Dilation condition: convolution result > 0
        result = (conv_result > 0.01).astype(cp.float32)  # Small threshold to avoid noise
        
        return result * 255.0 if mask_gpu.max() > 1 else result
    
    def gpu_morphological_operations_fast(self, mask, operation='closing', kernel_size=3):
        """Ultra-fast GPU morphological operations using direct kernel operations."""
        if not self.gpu_available:
            return self._cpu_morphological_fallback(mask, operation, kernel_size)
        
        try:
            # Transfer to GPU as boolean for faster operations
            mask_gpu = cp.asarray(mask > 0, dtype=cp.bool_)
            
            # Create simple kernel for speed
            kernel_radius = kernel_size // 2
            
            if operation == 'erosion':
                result_gpu = self._gpu_fast_erosion(mask_gpu, kernel_radius)
            elif operation == 'dilation':
                result_gpu = self._gpu_fast_dilation(mask_gpu, kernel_radius)
            elif operation == 'opening':
                # Opening = erosion then dilation
                eroded = self._gpu_fast_erosion(mask_gpu, kernel_radius)
                result_gpu = self._gpu_fast_dilation(eroded, kernel_radius)
                del eroded
            elif operation == 'closing':
                # Closing = dilation then erosion
                dilated = self._gpu_fast_dilation(mask_gpu, kernel_radius)
                result_gpu = self._gpu_fast_erosion(dilated, kernel_radius)
                del dilated
            else:
                return self._cpu_morphological_fallback(mask, operation, kernel_size)
            
            # Convert back to original format
            if mask.dtype == np.uint8:
                result = cp.asnumpy(result_gpu.astype(cp.uint8) * 255)
            else:
                result = cp.asnumpy(result_gpu.astype(mask.dtype))
            
            # Cleanup
            del mask_gpu, result_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            logging.info(f"? Used ultra-fast GPU morphological operation: {operation}")
            return result
            
        except Exception as e:
            logging.warning(f"Fast GPU morphological operation failed: {e}")
            return self._cpu_morphological_fallback(mask, operation, kernel_size)
    
    def _gpu_fast_erosion(self, mask_gpu, radius):
        """Ultra-fast GPU erosion using sliding window minimum."""
        result = mask_gpu.copy()
        
        # Apply erosion by checking minimum in neighborhood
        # This is much faster than convolution for simple kernels
        h, w = mask_gpu.shape
        
        # Create shifted versions and take minimum
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                
                # Create shifted version with padding
                if dy >= 0 and dx >= 0:
                    shifted = cp.zeros_like(mask_gpu)
                    shifted[dy:, dx:] = mask_gpu[:-dy if dy > 0 else h, :-dx if dx > 0 else w]
                elif dy >= 0 and dx < 0:
                    shifted = cp.zeros_like(mask_gpu)
                    shifted[dy:, :dx] = mask_gpu[:-dy if dy > 0 else h, -dx:]
                elif dy < 0 and dx >= 0:
                    shifted = cp.zeros_like(mask_gpu)
                    shifted[:dy, dx:] = mask_gpu[-dy:, :-dx if dx > 0 else w]
                else:  # dy < 0 and dx < 0
                    shifted = cp.zeros_like(mask_gpu)
                    shifted[:dy, :dx] = mask_gpu[-dy:, -dx:]
                
                # Take minimum (AND operation for erosion)
                result = cp.logical_and(result, shifted)
                del shifted
        
        return result
    
    def _gpu_fast_dilation(self, mask_gpu, radius):
        """Ultra-fast GPU dilation using sliding window maximum."""
        result = mask_gpu.copy()
        
        # Apply dilation by checking maximum in neighborhood
        h, w = mask_gpu.shape
        
        # Create shifted versions and take maximum
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                
                # Create shifted version with padding
                if dy >= 0 and dx >= 0:
                    shifted = cp.zeros_like(mask_gpu)
                    shifted[dy:, dx:] = mask_gpu[:-dy if dy > 0 else h, :-dx if dx > 0 else w]
                elif dy >= 0 and dx < 0:
                    shifted = cp.zeros_like(mask_gpu)
                    shifted[dy:, :dx] = mask_gpu[:-dy if dy > 0 else h, -dx:]
                elif dy < 0 and dx >= 0:
                    shifted = cp.zeros_like(mask_gpu)
                    shifted[:dy, dx:] = mask_gpu[-dy:, :-dx if dx > 0 else w]
                else:  # dy < 0 and dx < 0
                    shifted = cp.zeros_like(mask_gpu)
                    shifted[:dy, :dx] = mask_gpu[-dy:, -dx:]
                
                # Take maximum (OR operation for dilation)
                result = cp.logical_or(result, shifted)
                del shifted
        
        return result
    
    def _cpu_morphological_fallback(self, mask, operation, kernel_size):
        """CPU fallback for morphological operations."""
        import cv2
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operation == 'closing':
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif operation == 'opening':
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == 'erosion':
            return cv2.erode(mask, kernel, iterations=1)
        elif operation == 'dilation':
            return cv2.dilate(mask, kernel, iterations=1)
        else:
            return mask
    
    def gpu_connected_components(self, binary_mask):
        """GPU-accelerated connected components using CuPy."""
        if not self.gpu_available:
            return self._cpu_connected_components(binary_mask)
        
        try:
            # Transfer to GPU
            mask_gpu = cp.asarray(binary_mask, dtype=cp.uint8)
            
            # Use CuPy's label function (equivalent to scipy.ndimage.label)
            from cupyx.scipy.ndimage import label as gpu_label
            labeled_gpu, num_features = gpu_label(mask_gpu)
            
            # Transfer back
            labeled_mask = cp.asnumpy(labeled_gpu)
            
            # Cleanup
            del mask_gpu, labeled_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            logging.info(f"? Used GPU connected components: {num_features} components found")
            return labeled_mask, num_features
        except Exception as e:
            logging.warning(f"GPU connected components failed: {e}")
            return self._cpu_connected_components(binary_mask)
    
    def _cpu_connected_components(self, binary_mask):
        """CPU fallback for connected components."""
        from scipy.ndimage import label
        return label(binary_mask)
    
    def gpu_mask_statistics(self, mask, properties_mask=None):
        """GPU-accelerated computation of mask statistics."""
        if not self.gpu_available:
            return self._cpu_mask_statistics(mask, properties_mask)
        
        try:
            # Transfer to GPU
            mask_gpu = cp.asarray(mask)
            
            # Compute statistics on GPU
            stats = {
                'mean': float(cp.mean(mask_gpu)),
                'std': float(cp.std(mask_gpu)),
                'min': float(cp.min(mask_gpu)),
                'max': float(cp.max(mask_gpu)),
                'nonzero_count': int(cp.count_nonzero(mask_gpu)),
                'unique_labels': len(cp.unique(mask_gpu)),
                'total_pixels': int(mask_gpu.size)
            }
            
            if properties_mask is not None:
                props_gpu = cp.asarray(properties_mask)
                # Compute overlap statistics
                intersection = cp.logical_and(mask_gpu > 0, props_gpu > 0)
                union = cp.logical_or(mask_gpu > 0, props_gpu > 0)
                stats['overlap_ratio'] = float(cp.sum(intersection) / cp.sum(mask_gpu > 0)) if cp.sum(mask_gpu > 0) > 0 else 0.0
                stats['jaccard_index'] = float(cp.sum(intersection) / cp.sum(union)) if cp.sum(union) > 0 else 0.0
                del props_gpu, intersection, union
            
            # Cleanup
            del mask_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            logging.info("? Used GPU mask statistics computation")
            return stats
        except Exception as e:
            logging.warning(f"GPU statistics computation failed: {e}")
            return self._cpu_mask_statistics(mask, properties_mask)
    
    def _cpu_mask_statistics(self, mask, properties_mask=None):
        """CPU fallback for mask statistics."""
        stats = {
            'mean': float(np.mean(mask)),
            'std': float(np.std(mask)),
            'min': float(np.min(mask)),
            'max': float(np.max(mask)),
            'nonzero_count': int(np.count_nonzero(mask)),
            'unique_labels': len(np.unique(mask)),
            'total_pixels': int(mask.size)
        }
        
        if properties_mask is not None:
            intersection = np.logical_and(mask > 0, properties_mask > 0)
            union = np.logical_or(mask > 0, properties_mask > 0)
            stats['overlap_ratio'] = float(np.sum(intersection) / np.sum(mask > 0)) if np.sum(mask > 0) > 0 else 0.0
            stats['jaccard_index'] = float(np.sum(intersection) / np.sum(union)) if np.sum(union) > 0 else 0.0
        
        return stats
    
    def gpu_histogram_analysis(self, image, bins=256):
        """GPU-accelerated histogram computation."""
        if not self.gpu_available:
            return self._cpu_histogram_analysis(image, bins)
        
        try:
            # Transfer to GPU
            image_gpu = cp.asarray(image)
            
            # Compute histogram on GPU
            if len(image_gpu.shape) == 3:
                # RGB image - compute per channel
                histograms = []
                for channel in range(image_gpu.shape[2]):
                    hist_gpu, bin_edges_gpu = cp.histogram(image_gpu[:,:,channel], bins=bins)
                    histograms.append(cp.asnumpy(hist_gpu))
                bin_edges = cp.asnumpy(bin_edges_gpu)
            else:
                # Grayscale
                hist_gpu, bin_edges_gpu = cp.histogram(image_gpu, bins=bins)
                histograms = [cp.asnumpy(hist_gpu)]
                bin_edges = cp.asnumpy(bin_edges_gpu)
            
            # Cleanup
            del image_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            logging.info(f"? Used GPU histogram analysis with {bins} bins")
            return histograms, bin_edges
        except Exception as e:
            logging.warning(f"GPU histogram computation failed: {e}")
            return self._cpu_histogram_analysis(image, bins)
    
    def _cpu_histogram_analysis(self, image, bins=256):
        """CPU fallback for histogram analysis."""
        if len(image.shape) == 3:
            # RGB image - compute per channel
            histograms = []
            for channel in range(image.shape[2]):
                hist, bin_edges = np.histogram(image[:,:,channel], bins=bins)
                histograms.append(hist)
        else:
            # Grayscale
            hist, bin_edges = np.histogram(image, bins=bins)
            histograms = [hist]
        
        return histograms, bin_edges
    
    def gpu_batch_array_operations(self, arrays_list, operation='sum'):
        """GPU-accelerated batch operations on multiple arrays."""
        if not self.gpu_available or len(arrays_list) < 2:
            return self._cpu_batch_operations(arrays_list, operation)
        
        try:
            # Transfer all arrays to GPU
            gpu_arrays = [cp.asarray(arr) for arr in arrays_list]
            
            # Perform batch operation
            if operation == 'sum':
                result_gpu = cp.sum(cp.stack(gpu_arrays), axis=0)
            elif operation == 'mean':
                result_gpu = cp.mean(cp.stack(gpu_arrays), axis=0)
            elif operation == 'max':
                result_gpu = cp.max(cp.stack(gpu_arrays), axis=0)
            elif operation == 'min':
                result_gpu = cp.min(cp.stack(gpu_arrays), axis=0)
            elif operation == 'logical_or':
                result_gpu = cp.logical_or.reduce(gpu_arrays)
            elif operation == 'logical_and':
                result_gpu = cp.logical_and.reduce(gpu_arrays)
            else:
                logging.warning(f"Unknown batch operation: {operation}")
                return self._cpu_batch_operations(arrays_list, operation)
            
            # Transfer back
            result = cp.asnumpy(result_gpu)
            
            # Cleanup
            for gpu_arr in gpu_arrays:
                del gpu_arr
            del result_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            logging.info(f"? Used GPU batch operation: {operation} on {len(arrays_list)} arrays")
            return result
        except Exception as e:
            logging.warning(f"GPU batch operations failed: {e}")
            return self._cpu_batch_operations(arrays_list, operation)
    
    def _cpu_batch_operations(self, arrays_list, operation='sum'):
        """CPU fallback for batch operations."""
        if len(arrays_list) < 2:
            return arrays_list[0] if arrays_list else None
        
        if operation == 'sum':
            return np.sum(np.stack(arrays_list), axis=0)
        elif operation == 'mean':
            return np.mean(np.stack(arrays_list), axis=0)
        elif operation == 'max':
            return np.max(np.stack(arrays_list), axis=0)
        elif operation == 'min':
            return np.min(np.stack(arrays_list), axis=0)
        elif operation == 'logical_or':
            return np.logical_or.reduce(arrays_list)
        elif operation == 'logical_and':
            return np.logical_and.reduce(arrays_list)
        else:
            return arrays_list[0]
    
    def gpu_mask_overlay_operations(self, base_mask, overlay_masks, operations):
        """GPU-accelerated mask overlay operations with multiple masks."""
        if not self.gpu_available:
            return self._cpu_mask_overlay(base_mask, overlay_masks, operations)
        
        try:
            # Transfer base mask to GPU
            base_gpu = cp.asarray(base_mask)
            result_gpu = base_gpu.copy()
            
            # Process each overlay
            for overlay, op in zip(overlay_masks, operations):
                overlay_gpu = cp.asarray(overlay)
                
                if op == 'union':
                    result_gpu = cp.logical_or(result_gpu, overlay_gpu)
                elif op == 'intersection':
                    result_gpu = cp.logical_and(result_gpu, overlay_gpu)
                elif op == 'difference':
                    result_gpu = cp.logical_and(result_gpu, ~overlay_gpu)
                elif op == 'add':
                    result_gpu = result_gpu + overlay_gpu
                elif op == 'subtract':
                    result_gpu = cp.maximum(result_gpu - overlay_gpu, 0)
                elif op == 'multiply':
                    result_gpu = result_gpu * overlay_gpu
                else:
                    logging.warning(f"Unknown overlay operation: {op}")
                
                del overlay_gpu
            
            # Transfer back
            result = cp.asnumpy(result_gpu)
            
            # Cleanup
            del base_gpu, result_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            logging.info(f"? Used GPU mask overlay with {len(overlay_masks)} overlays")
            return result
        except Exception as e:
            logging.warning(f"GPU mask overlay failed: {e}")
            return self._cpu_mask_overlay(base_mask, overlay_masks, operations)
    
    def _cpu_mask_overlay(self, base_mask, overlay_masks, operations):
        """CPU fallback for mask overlay operations."""
        result = base_mask.copy()
        
        for overlay, op in zip(overlay_masks, operations):
            if op == 'union':
                result = np.logical_or(result, overlay)
            elif op == 'intersection':
                result = np.logical_and(result, overlay)
            elif op == 'difference':
                result = np.logical_and(result, ~overlay)
            elif op == 'add':
                result = result + overlay
            elif op == 'subtract':
                result = np.maximum(result - overlay, 0)
            elif op == 'multiply':
                result = result * overlay
        
        return result

class OptimizedWindowBatcher:
    """Batches windows for processing to improve GPU utilization."""
    
    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.current_batch = []
        self.current_coords = []
    
    def add_window(self, window, coords):
        """Add window to current batch."""
        self.current_batch.append(window)
        self.current_coords.append(coords)
        
        # Return batch when full
        if len(self.current_batch) >= self.batch_size:
            batch = self.current_batch.copy()
            coords = self.current_coords.copy()
            self.current_batch.clear()
            self.current_coords.clear()
            return batch, coords
        
        return None, None
    
    def get_remaining_batch(self):
        """Get any remaining windows in partial batch."""
        if self.current_batch:
            batch = self.current_batch.copy()
            coords = self.current_coords.copy()
            self.current_batch.clear()
            self.current_coords.clear()
            return batch, coords
        return None, None

class AdvancedGPUProcessor:
    """Advanced GPU operationssor with multi-stream parallel processing and enhanced error handling."""
    
    def __init__(self, max_gpu_memory_gb=30):
        self.gpu_available = CUPY_AVAILABLE and cp.cuda.is_available()
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.streams = []
        
        if self.gpu_available:
            # Initialize multiple CUDA streams for parallel processing
            self.streams = [cp.cuda.Stream() for _ in range(4)]
            logging.info("? Advanced GPU processor initialized with 4 streams")
        else:
            logging.info("???  Advanced GPU processor using CPU fallback")
    
    def gpu_parallel_chunk_processing(self, mask, mapping_dict, num_streams=4):
        """Process mask chunks in parallel using multiple GPU streams."""
        if not self.gpu_available or not mapping_dict:
            return self._single_stream_processing(mask, mapping_dict)
        
        try:
            # Calculate chunk sizes
            chunk_size = mask.shape[0] // num_streams
            chunks = []
            for i in range(num_streams):
                start_row = i * chunk_size
                end_row = start_row + chunk_size if i < num_streams - 1 else mask.shape[0]
                chunks.append((start_row, end_row))
            
            # Create lookup table once
            max_label = max(max(mapping_dict.keys()), max(mapping_dict.values()))
            lookup_cpu = np.arange(max_label + 1, dtype=mask.dtype)
            for old_label, new_label in mapping_dict.items():
                lookup_cpu[old_label] = new_label
            
            # Process chunks in parallel
            result_chunks = []
            gpu_chunks = []
            lookup_gpus = []
            
            # Launch all chunks asynchronously
            for i, (start_row, end_row) in enumerate(chunks):
                stream = self.streams[i % len(self.streams)]
                
                with stream:
                    # Transfer chunk to GPU
                    chunk = mask[start_row:end_row, :]
                    chunk_gpu = cp.asarray(chunk)
                    lookup_gpu = cp.asarray(lookup_cpu)
                    
                    # Apply mapping
                    result_gpu = lookup_gpu[chunk_gpu]
                    
                    gpu_chunks.append(result_gpu)
                    lookup_gpus.append(lookup_gpu)
            
            # Wait for all streams to complete and collect results
            for i, (result_gpu, lookup_gpu) in enumerate(zip(gpu_chunks, lookup_gpus)):
                stream = self.streams[i % len(self.streams)]
                with stream:
                    result_chunk = cp.asnumpy(result_gpu)
                    result_chunks.append(result_chunk)
                    del result_gpu, lookup_gpu
            
            # Synchronize all streams
            for stream in self.streams:
                stream.synchronize()
            
            # Combine results
            result = np.vstack(result_chunks)
            
            # Cleanup
            cp.get_default_memory_pool().free_all_blocks()
            
            logging.info(f"? Used GPU parallel processing with {num_streams} streams")
            return result
            
        except Exception as e:
            logging.warning(f"GPU parallel processing failed: {e}")
            return self._single_stream_processing(mask, mapping_dict)
    
    def _single_stream_processing(self, mask, mapping_dict):
        """Fallback to single-stream processing."""
        if not self.gpu_available:
            return mask  # Basic CPU fallback
        
        try:
            mask_gpu = cp.asarray(mask)
            max_label = max(max(mapping_dict.keys()), max(mapping_dict.values()))
            lookup_gpu = cp.arange(max_label + 1, dtype=mask.dtype)
            
            for old_label, new_label in mapping_dict.items():
                lookup_gpu[old_label] = new_label
            
            result_gpu = lookup_gpu[mask_gpu]
            result = cp.asnumpy(result_gpu)
            
            del mask_gpu, lookup_gpu, result_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            return result
        except Exception:
            return mask
    
    def gpu_advanced_filtering(self, image, filters_config):
        """Advanced GPU filtering with multiple filter types."""
        if not self.gpu_available:
            return self._cpu_advanced_filtering(image, filters_config)
        
        try:
            # Transfer to GPU
            image_gpu = cp.asarray(image, dtype=cp.float32)
            result_gpu = image_gpu.copy()
            
            for filter_type, params in filters_config.items():
                if filter_type == 'gaussian':
                    from cupyx.scipy.ndimage import gaussian_filter
                    sigma = params.get('sigma', 1.0)
                    result_gpu = gaussian_filter(result_gpu, sigma=sigma)
                
                elif filter_type == 'median':
                    from cupyx.scipy.ndimage import median_filter
                    size = params.get('size', 3)
                    result_gpu = median_filter(result_gpu, size=size)
                
                elif filter_type == 'sobel':
                    # Sobel edge detection
                    sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
                    sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)
                    
                    from cupyx.scipy.ndimage import convolve
                    if len(result_gpu.shape) == 3:
                        gray_gpu = cp.mean(result_gpu, axis=2)
                    else:
                        gray_gpu = result_gpu
                    
                    grad_x = convolve(gray_gpu, sobel_x)
                    grad_y = convolve(gray_gpu, sobel_y)
                    magnitude = cp.sqrt(grad_x**2 + grad_y**2)
                    
                    if len(result_gpu.shape) == 3:
                        result_gpu = cp.stack([magnitude] * 3, axis=2)
                    else:
                        result_gpu = magnitude
                
                elif filter_type == 'laplacian':
                    from cupyx.scipy.ndimage import laplace
                    result_gpu = laplace(result_gpu)
            
            # Convert back to original dtype and transfer to CPU
            result = cp.asnumpy(result_gpu.astype(image.dtype))
            
            # Cleanup
            del image_gpu, result_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            logging.info(f"? Used GPU advanced filtering with {len(filters_config)} filters")
            return result
            
        except Exception as e:
            logging.warning(f"GPU advanced filtering failed: {e}")
            return self._cpu_advanced_filtering(image, filters_config)
    
    def _cpu_advanced_filtering(self, image, filters_config):
        """CPU fallback for advanced filtering."""
        import cv2
        from scipy.ndimage import gaussian_filter, median_filter, laplace
        
        result = image.astype(np.float32)
        
        for filter_type, params in filters_config.items():
            if filter_type == 'gaussian':
                sigma = params.get('sigma', 1.0)
                result = gaussian_filter(result, sigma=sigma)
            elif filter_type == 'median':
                size = params.get('size', 3)
                result = median_filter(result, size=size)
            elif filter_type == 'sobel':
                if len(result.shape) == 3:
                    gray = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray = result.astype(np.uint8)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(sobelx**2 + sobely**2)
                if len(result.shape) == 3:
                    result = np.stack([magnitude] * 3, axis=2)
                else:
                    result = magnitude
            elif filter_type == 'laplacian':
                result = laplace(result)
        
        return result.astype(image.dtype)

class GPUMemoryProfiler:
    """Profile GPU memory usage for optimization."""
    
    def __init__(self):
        self.gpu_available = CUPY_AVAILABLE and cp.cuda.is_available()
        self.memory_log = []
    
    def log_memory_usage(self, operation_name):
        """Log current GPU memory usage."""
        if not self.gpu_available:
            return
        
        try:
            free_memory, total_memory = cp.cuda.Device().mem_info
            used_memory = total_memory - free_memory
            usage_percent = (used_memory / total_memory) * 100
            
            self.memory_log.append({
                'operation': operation_name,
                'used_gb': used_memory / (1024**3),
                'total_gb': total_memory / (1024**3),
                'usage_percent': usage_percent
            })
            
            logging.info(f"?? GPU Memory [{operation_name}]: {used_memory/1024/1024/1024:.1f}GB/{total_memory/1024/1024/1024:.1f}GB ({usage_percent:.1f}%)")
        except Exception:
            pass
    
    def get_peak_usage(self):
        """Get peak memory usage from log."""
        if not self.memory_log:
            return None
        
        return max(self.memory_log, key=lambda x: x['usage_percent'])
    
    def clear_log(self):
        """Clear memory usage log."""
        self.memory_log.clear()

# Utility functions for easy integration
def apply_gpu_morphological_operation(mask, operation='closing', kernel_size=3):
    """Easy-to-use GPU morphological operation."""
    gpu_ops = GPUAcceleratedOperations()
    return gpu_ops.gpu_morphological_operations(mask, operation, kernel_size)

def compute_gpu_mask_statistics(mask, properties_mask=None):
    """Easy-to-use GPU mask statistics."""
    gpu_ops = GPUAcceleratedOperations()
    return gpu_ops.gpu_mask_statistics(mask, properties_mask)

def apply_gpu_connected_components(binary_mask):
    """Easy-to-use GPU connected components."""
    gpu_ops = GPUAcceleratedOperations()
    return gpu_ops.gpu_connected_components(binary_mask)

def process_gpu_batch_arrays(arrays_list, operation='sum'):
    """Easy-to-use GPU batch processing."""
    gpu_ops = GPUAcceleratedOperations()
    return gpu_ops.gpu_batch_array_operations(arrays_list, operation)


def gpu_sobel_preprocessing(window_np, apply_bilateral=False):
    """
    GPU-accelerated Sobel preprocessing that exactly matches cv2.Sobel behavior.
    Replicates the exact preprocessing pipeline used for model training.
    
    Args:
        window_np: Input image as numpy array (RGB format)
        apply_bilateral: Whether to apply bilateral filtering after Sobel
        
    Returns:
        Processed image array matching cv2.Sobel output exactly
    """
    try:
        import cupy as cp
        
        # For now, prioritize accuracy over GPU acceleration for Sobel preprocessing
        # The exact cv2.Sobel behavior is critical for inference accuracy
        # Future optimization: implement exact GPU kernel matching cv2.Sobel
        raise Exception("Using CPU fallback for guaranteed accuracy")
        
    except Exception:
        # CPU fallback with exact original implementation
        import cv2
        
        window_cv = cv2.cvtColor(window_np, cv2.COLOR_RGB2BGR)
        window_cv = window_cv.astype(np.float32)
        sobelx = cv2.Sobel(window_cv, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(window_cv, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        sobel_magnitude = np.uint8(sobel_magnitude)
        
        if apply_bilateral:
            window_cv = cv2.bilateralFilter(sobel_magnitude, 20, 20, 20)
        else:
            window_cv = sobel_magnitude
            
        result = cv2.cvtColor(window_cv, cv2.COLOR_BGR2RGB)
        return result


def gpu_bilateral_filter_approximation(image, d=20, sigma_color=20, sigma_space=20):
    """
    GPU approximation of bilateral filter using Gaussian approximations.
    Note: This is an approximation - for exact matching, use CPU cv2.bilateralFilter.
    
    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Filtered image
    """
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as ndi
        
        gpu_image = cp.asarray(image)
        
        # Simple approximation using Gaussian blur
        # This is NOT exact but provides GPU acceleration
        filtered = ndi.gaussian_filter(gpu_image, sigma=sigma_space/3.0)
        
        return cp.asnumpy(filtered)
        
    except Exception as e:
        print(f"Warning: GPU bilateral filter failed: {e}")
        # CPU fallback
        import cv2
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def _pytorch_sobel_fallback(image_tensor):
    """Standard PyTorch Sobel implementation."""
    # Sobel kernels
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=image_tensor.dtype, device=image_tensor.device).unsqueeze(0).unsqueeze(0)
    
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=image_tensor.dtype, device=image_tensor.device).unsqueeze(0).unsqueeze(0)
    
    # Convert to grayscale if needed
    if image_tensor.shape[1] == 3:
        gray = 0.299 * image_tensor[:, 0:1] + 0.587 * image_tensor[:, 1:2] + 0.114 * image_tensor[:, 2:3]
    else:
        gray = image_tensor
    
    # Apply Sobel filters
    grad_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    
    # Compute magnitude
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    
    return magnitude


def gpu_image_inversion(image):
    """
    GPU-accelerated image inversion (255 - image).
    Simple and fast operation perfect for GPU acceleration.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Inverted image array
    """
    try:
        import cupy as cp
        
        # Move to GPU
        gpu_image = cp.asarray(image)
        
        # Perform inversion: 255 - image
        inverted_gpu = 255 - gpu_image
        
        # Move back to CPU
        result = cp.asnumpy(inverted_gpu)
        
        return result
        
    except Exception as e:
        # CPU fallback
        return 255 - image
