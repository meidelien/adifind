"""
Comprehensive performance optimization integration for AdiFind.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all optimization modules
from gpu_acceleration import (
    AdvancedGPUProcessor, 
    gpu_memory_pool_manager,
    gpu_batch_resize,
    gpu_array_operations,
    gpu_mask_operations
)
from io_acceleration import (
    AsyncImageLoader,
    MemoryManager, 
    FastImageIO,
    TileCache,
    get_optimized_read_function
)
import config


class PerformanceOptimizer:
    """
    Master performance optimization coordinator.
    Automatically configures and manages all performance enhancements.
    """
    
    def __init__(self, image_handler=None):
        self.image_handler = image_handler
        self.gpu_processor = None
        self.memory_manager = None
        self.async_loader = None
        self.tile_cache = None
        self.optimized_read_func = None
        
        # Performance monitoring
        self.optimization_status = {}
        
        # Initialize all optimizations
        self._initialize_optimizations()
    
    def _initialize_optimizations(self):
        """Initialize all available optimizations."""
        
        # 1. GPU Optimizations (Conservative mode)
        try:
            # Disable advanced GPU processor to avoid OOM
            # self.gpu_processor = AdvancedGPUProcessor()
            self.gpu_processor = None
            
            # Disable GPU memory pool for RTX 5090 to avoid OOM
            # if config.Config.ENABLE_MEMORY_POOLING:
            #     gpu_memory_pool_manager()
            
            self.optimization_status['gpu_acceleration'] = False  # Disabled for stability
            print("?? GPU acceleration disabled to prevent OOM")
            
        except Exception as e:
            self.optimization_status['gpu_acceleration'] = False
            print(f"?? GPU acceleration failed: {e}")
        
        # 2. Memory Management (Conservative)
        try:
            # Use conservative memory target to avoid OOM
            target_memory = 12  # Conservative 12GB instead of 16-24GB
            self.memory_manager = MemoryManager(target_memory_gb=target_memory)
            self.optimization_status['memory_management'] = True
            print("? Conservative memory management initialized")
            
        except Exception as e:
            self.optimization_status['memory_management'] = False
            print(f"?? Memory management failed: {e}")
        
        # 3. Async I/O
        if self.image_handler and config.Config.ENABLE_ASYNC_IO:
            try:
                # Larger prefetch for RTX 5090
                prefetch_size = 8 if config.Config.GPU_MEMORY_LIMIT_GB > 20 else 4
                self.async_loader = AsyncImageLoader(
                    self.image_handler, 
                    prefetch_size=prefetch_size,
                    max_workers=4
                )
                self.optimization_status['async_io'] = True
                print("? Asynchronous I/O initialized")
                
            except Exception as e:
                self.optimization_status['async_io'] = False
                print(f"?? Async I/O failed: {e}")
        
        # 4. Smart Caching (Disabled to save memory)
        if config.Config.ENABLE_SMART_CACHING:
            try:
                # Smaller cache to avoid memory issues
                cache_size = 128  # Much smaller cache (was 512-1024)
                self.tile_cache = TileCache(max_size_mb=cache_size)
                
                if self.image_handler:
                    self.optimized_read_func = get_optimized_read_function(self.image_handler)
                
                self.optimization_status['smart_caching'] = True
                print("? Conservative caching initialized")
                
            except Exception as e:
                self.optimization_status['smart_caching'] = False
                print(f"?? Smart caching failed: {e}")
        else:
            self.optimization_status['smart_caching'] = False
            print("?? Smart caching disabled to save memory")
        
        # 5. Configuration Optimizations
        self._apply_config_optimizations()
    
    def _apply_config_optimizations(self):
        """Apply automatic configuration optimizations."""
        
        # Detect RTX 5090 and apply optimal settings
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if "RTX 5090" in gpu_name or gpu_memory_gb > 28:
                    print(f"?? RTX 5090 detected! Applying optimal settings...")
                    
                    # Apply RTX 5090 specific optimizations
                    config.Config.BATCH_INFERENCE_SIZE = 32  # Even larger batches
                    config.Config.MAX_IO_WORKERS = 96        # More parallel workers
                    config.Config.GPU_STREAM_COUNT = 12      # More CUDA streams
                    
                    self.optimization_status['rtx5090_optimizations'] = True
                    
                elif gpu_memory_gb > 16:
                    print(f"?? High-end GPU detected ({gpu_memory_gb:.1f}GB)! Applying enhanced settings...")
                    config.Config.BATCH_INFERENCE_SIZE = min(24, config.Config.BATCH_INFERENCE_SIZE)
                    self.optimization_status['high_end_gpu'] = True
                    
        except Exception as e:
            print(f"?? GPU detection failed: {e}")
    
    def get_optimized_read_region(self):
        """Get the optimized read_region function."""
        if self.optimized_read_func:
            return self.optimized_read_func
        elif self.async_loader:
            return self.async_loader.get_region_cached
        else:
            return self.image_handler.read_region if self.image_handler else None
    
    def process_batch_gpu(self, images, operation='preprocess'):
        """Process a batch of images using GPU acceleration."""
        if not self.gpu_processor:
            return images
        
        try:
            if operation == 'resize':
                return gpu_batch_resize(images)
            elif operation == 'array_ops':
                return [gpu_array_operations(img) for img in images]
            elif operation == 'mask_ops':
                return [gpu_mask_operations(img) for img in images]
            else:
                return images
                
        except Exception as e:
            print(f"?? GPU batch processing failed: {e}")
            return images
    
    def adaptive_batch_size(self, base_size, image_size):
        """Calculate optimal batch size based on current system state."""
        if not self.memory_manager:
            return base_size
        
        try:
            return self.memory_manager.adaptive_batch_size(base_size, image_size)
        except:
            return base_size
    
    def cleanup_memory(self):
        """Perform comprehensive memory cleanup."""
        try:
            # Clear tile cache
            if self.tile_cache:
                self.tile_cache.clear()
            
            # Clear async loader cache
            if self.async_loader:
                self.async_loader.clear_cache()
            
            # GPU memory cleanup
            if self.gpu_processor:
                import torch
                torch.cuda.empty_cache()
                
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
            
            # System memory cleanup
            if self.memory_manager:
                self.memory_manager.cleanup_temp_arrays()
            
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"?? Memory cleanup failed: {e}")
    
    def print_optimization_status(self):
        """Print the status of all optimizations."""
        print("\n?? Performance Optimization Status:")
        print("=" * 40)
        
        for optimization, status in self.optimization_status.items():
            status_icon = "?" if status else "?"
            opt_name = optimization.replace('_', ' ').title()
            print(f"{status_icon} {opt_name}")
        
        # Performance tips
        active_optimizations = sum(self.optimization_status.values())
        total_optimizations = len(self.optimization_status)
        
        print(f"\n?? Active: {active_optimizations}/{total_optimizations} optimizations")
        
        if active_optimizations == total_optimizations:
            print("?? All optimizations active! Maximum performance enabled.")
        elif active_optimizations > total_optimizations // 2:
            print("? Most optimizations active. Good performance expected.")
        else:
            print("?? Limited optimizations active. Consider checking dependencies.")


def create_performance_optimizer(image_handler=None):
    """
    Factory function to create and initialize a performance optimizer.
    
    Args:
        image_handler: Optional image handler for I/O optimizations
        
    Returns:
        Configured PerformanceOptimizer instance
    """
    optimizer = PerformanceOptimizer(image_handler)
    return optimizer


def apply_global_optimizations():
    """
    Apply global performance optimizations that don't require specific instances.
    Call this at the start of your application.
    """
    print("?? Applying global performance optimizations...")
    
    # Set optimal thread counts for NumPy/OpenCV
    try:
        import os
        import cv2
        
        # Set optimal thread counts based on CPU
        cpu_count = os.cpu_count()
        optimal_threads = min(cpu_count, 16)  # Cap at 16 for most workloads
        
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        cv2.setNumThreads(optimal_threads)
        
        print(f"? Set {optimal_threads} threads for OpenCV/NumPy")
        
    except Exception as e:
        print(f"?? Thread optimization failed: {e}")
    
    # GPU optimization flags
    try:
        import torch
        if torch.cuda.is_available():
            # Enable optimized GPU operations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            print("? CUDA optimizations enabled")
            
    except Exception as e:
        print(f"?? CUDA optimization failed: {e}")


# Auto-apply global optimizations when imported
if __name__ != "__main__":
    apply_global_optimizations()
