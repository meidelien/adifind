#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AdiFind configuration defaults and paths."""

from model_registry import MODEL_FILENAMES


class Config:
    """Configuration defaults for AdiFind WSI analysis."""

    def __getattribute__(self, name):
        if name == "USE_GPU_ACCELERATION":
            return object.__getattribute__(self, "USE_GPU_INFERENCE")
        if name == "ENABLE_GPU_ACCELERATION":
            return any((
                object.__getattribute__(self, "USE_CUPY"),
                object.__getattribute__(self, "USE_GPU_PREPROCESSING"),
                object.__getattribute__(self, "ENABLE_GPU_LABEL_MAPPING"),
            ))
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name == "USE_GPU_ACCELERATION":
            enabled = bool(value)
            object.__setattr__(self, "USE_GPU_INFERENCE", enabled)
            object.__setattr__(self, "USE_CUPY", enabled)
            object.__setattr__(self, "USE_GPU_PREPROCESSING", enabled)
            object.__setattr__(self, "ENABLE_GPU_LABEL_MAPPING", enabled)
            return
        if name == "ENABLE_GPU_ACCELERATION":
            enabled = bool(value)
            object.__setattr__(self, "USE_CUPY", enabled)
            object.__setattr__(self, "USE_GPU_PREPROCESSING", enabled)
            object.__setattr__(self, "ENABLE_GPU_LABEL_MAPPING", enabled)
            return
        object.__setattr__(self, name, value)

    # Feature toggles
    ENABLE_TUMOR_SEGMENTATION = True
    SHOW_TUMOR_BOUNDARIES = True
    USE_GPU_INFERENCE = True
    USE_CUPY = True
    USE_GPU_PREPROCESSING = True
    SHOW_DETAILED_INFO = True

    # Image processing
    APPLY_IMAGE_INVERSION = True
    APPLY_SOBEL_FILTER = True
    APPLY_BILATERAL_FILTER = False

    # Analysis
    MIN_ADIPOCYTE_AREA_MICRONS = 250
    MAX_ADIPOCYTE_AREA_MICRONS = 25000
    GRID_CELL_SIZE_MICRONS = 1100
    IOU_THRESHOLD = 0.3
    MERGE_IOU_THRESHOLD = 0.2
    CONFIDENCE_THRESHOLD = 0.95
    SCALING_FACTOR = 0.3
    DEFAULT_MPP = 0.50
    PIXEL_SIZE_UM = 0.50  # Same as DEFAULT_MPP

    # Processing
    DESIRED_RESOLUTION_LEVEL = 0
    MAX_IO_WORKERS = 30  # Reduced for 64GB RAM
    BATCH_INFERENCE_SIZE = 4  # Reduced for 64GB RAM
    ENABLE_GPU_LABEL_MAPPING = False # GPU label mapping can cause OOM on large masks, so disabled by default. TODO: DELETE; USABLE WHEN GPUS HAVE 200GB VRAM
    GPU_LABEL_MAPPING_THRESHOLD = 1000
    GPU_MEMORY_LIMIT_GB = 30  # Leave ~2GB free
    FORCE_CPU_LABEL_MAPPING = True
    WINDOW_SIZE = (2000, 2000)
    STRIDE = (1700, 1700) # OLD 


    # Output
    SAVE_ANNOTATED_IMAGE = False
    SAVE_FULL_MASK = False
    SAVE_SUMMARY_CSV = True
    SKIP_PROCESSED_IMAGES = True
    RETRY_FAILED_IMAGES = False  # Can be overridden by CLI
    SAVE_DISTANCE_COLORED_IMAGE = False
    SAVE_SIZE_OUTLINED_DISTANCE_IMAGE = False
    SAVE_SIZE_ALPHA_DISTANCE_IMAGE = False
    SAVE_SIZE_SYMBOL_DISTANCE_IMAGE = False
    SAVE_TUMOR_ZONE_OVERLAY_IMAGE = False

    # CSV export
    CALCULATE_EXTENDED_PROPERTIES = False

    # Annotated image
    ANNOTATED_IMAGE_SCALE = 1.0
    ANNOTATED_IMAGE_SAVE_MODE = "high_quality"  # fast, balanced, high_quality
    BENCHMARK_IMAGE_SAVING = False

    # Visualization
    SHOW_GRID = False
    SHOW_GRID_LABELS = False
    SHOW_ADIPOCYTE_IDS = False
    MIN_TUMOR_AREA_PIXELS = 100           # Minimum detection size to count as real tumor (thumbnail res)
    MIN_TUMOR_PIXELS_FOR_DISTANCE = 50    # Minimum tumor pixels at analysis res for distance computation
    TUMOR_ZONE_NEAR_UM = 1500.0
    TUMOR_ZONE_INTERMEDIATE_UM = 5000.0
    TUMOR_ZONE_ALPHA = 0.25
    TUMOR_ZONE_NEAR_COLOR_RGB = (255, 255, 0)
    TUMOR_ZONE_INTERMEDIATE_COLOR_RGB = (0, 255, 0)
    TUMOR_ZONE_DISTAL_COLOR_RGB = (0, 128, 255)
    TUMOR_INSTANCE_ALPHA = 0.35
    TUMOR_INSTANCE_COLORS_RGB = (
        (230, 57, 70),
        (29, 185, 84),
        (0, 118, 182),
        (255, 183, 3),
        (131, 56, 236),
        (251, 133, 0),
        (42, 157, 143),
        (255, 0, 110),
        (88, 129, 87),
        (67, 97, 238),
    )
    DEBUG_MODE = False
    DEBUG_SAVE_UNPROCESSED_WINDOWS = False
    VERBOSE_LOGGING = False
    ENABLE_GPU_MONITORING = True
    ENABLE_PROFILING = False

    # GPU acceleration
    GPU_MORPHOLOGY_THRESHOLD = 1000000  # Use GPU for masks >1M pixels
    GPU_STATISTICS_THRESHOLD = 500000  # Use GPU for masks >500k pixels
    GPU_CONNECTED_COMPONENTS_THRESHOLD = 1000000  # Use GPU for masks >1M pixels
    ENABLE_GPU_STREAMS = True
    GPU_STREAM_COUNT = 20
    ENABLE_GPU_MEMORY_PROFILING = False
    GPU_CLEANUP_INTERVAL = 50  # 25-100 recommended
    GPU_CLEANUP_AGGRESSIVE = False

    # Async I/O
    ENABLE_ASYNC_IO = True
    ASYNC_CACHE_SIZE_MB = 256  # Reduced for 64GB RAM
    ASYNC_MAX_WORKERS = 4
    ASYNC_PREFETCH_SIZE = 12  # Reduced for 64GB RAM
    PIPELINE_PREFETCH_MULTIPLIER = 2
    PIPELINE_DIAGNOSTIC_INTERVAL_BATCHES = 25
    SELECTED_WINDOW_DIAGNOSTIC_INTERVAL = 100

    # Experimental
    GENERATE_INTERACTIVE_MAP = False  # Not implemented yet
    SAVE_POSTPROCESSED_IMAGE = False
    SAVE_UNPROCESSED_IMAGE = False

    # QuPath integration
    ENABLE_QUPATH_EXPORT = True
    SAVE_QUPATH_GEOJSON = True
    SAVE_QUPATH_SCRIPT = True
    QUPATH_ANNOTATION_CLASS = "Adipocyte"
    INCLUDE_MEASUREMENTS_IN_QUPATH = True

    # Tissue guidance
    ENABLE_TISSUE_GUIDANCE = True
    ENABLE_TISSUE_GUIDANCE_CACHE = True
    TISSUE_CACHE_DIR = "tissue_cache"
    TISSUE_OVERLAP_THRESHOLD = 0.3
    TISSUE_CONFIDENCE_THRESHOLD = 0.5
    TISSUE_NMS_THRESHOLD = 0.3
    TISSUE_THUMBNAIL_SIZE = 2048
    TISSUE_DETECTION_DOWNSAMPLE = 32
    MIN_TISSUE_COVERAGE = 0.1
    SAVE_TISSUE_DETECTION = True
    ENABLE_MULTI_REGION_OPTIMIZATION = True
    SAVE_REGION_STATISTICS = True
    SAVE_TISSUE_WINDOW_GRID_THUMBNAIL = False

    # ROI guidance
    ENABLE_ROI_GUIDANCE = False
    ROI_POLYGON_FILE = None  # Path to pre-saved ROI polygon JSON (skips interactive GUI)
    ROI_THUMBNAIL_MAX_DIM = 2048
    ROI_MIN_COVERAGE = 0.2

    # Memory efficiency (gigapixel images)
    MEMORY_EFFICIENT_MODE = True
    MASK_CHUNK_SIZE = 2048  # Recommended: 2048-4096 depending on RAM
    INCREMENTAL_PROPERTY_COLLECTION = True
    MAX_FULL_MASK_PIXELS = 500_000_000  # ~2GB for uint32
    INPLACE_LABEL_REMAPPING = True
    PARALLEL_CHUNK_PROCESSING = True
    CHUNK_WORKERS = 12  # Reduced for 64GB RAM

    # Low memory mode (<= 64GB RAM)
    LOW_MEMORY_MODE = False
    USE_MEMMAP_MASK = False  # Disk-backed mask storage (slower I/O)
    LOW_MEMORY_THRESHOLD_GB = 96
    LOW_MEMORY_MAX_ANNOTATED_PIXELS = 150_000_000  # 150MP vs 250MP default


import os


class Paths:
    """Paths for models, input, and output."""

    # Input
    IMAGE_PATH = r"example_data\K106942.svs"

    # Model paths (override via env vars for Docker or local installs).
    # When set to None, models are automatically downloaded from HuggingFace Hub
    # on first use. Set ADIFIND_*_MODEL_DIR env vars to use local checkpoints
    # with the canonical AdiFind filenames.
    ADIPOCYTE_MODEL_DIR = os.environ.get('ADIFIND_ADIPOCYTE_MODEL_DIR', None)
    ADIPOCYTE_MODEL_CHECKPOINT = MODEL_FILENAMES["adipocyte"]

    TUMOR_MODEL_DIR = os.environ.get('ADIFIND_TUMOR_MODEL_DIR', None)
    TUMOR_MODEL_CHECKPOINT = MODEL_FILENAMES["tumor"]

    # Tissue guidance model
    TISSUE_MODEL_DIR = os.environ.get('ADIFIND_TISSUE_MODEL_DIR', None)
    TISSUE_MODEL_CHECKPOINT = MODEL_FILENAMES["tissue"]

    # Output
    OUTPUT_DIR = os.environ.get('ADIFIND_OUTPUT_DIR', "adifind_output")
    SUMMARY_CSV_PATH = None


# Global instances
config = Config()
paths = Paths()

# Startup banner (copied from adifind_v165_opt.py)
BANNER = r"""
AdiFind v16 - Professional Adipocyte Detection and Analysis
================================================================

A comprehensive tool for automated adipocyte detection and analysis in whole slide images.
Features include:
- Advanced deep learning-based adipocyte segmentation
- Tumor detection and distance analysis
- Spatial grid analysis
- Professional visualization and reporting
- High-performance parallel processing
- QuPath integration for annotation overlay

Author: Martin Eide Lien
Version: 16
"""


# Backward compatibility aliases
_COMPAT_ALIASES = {
    # Configuration
    "ENABLE_TUMOUR_SEGMENTATION": config.ENABLE_TUMOR_SEGMENTATION,
    "SHOW_TUMOUR_MICRON_BOUNDARIES": config.SHOW_TUMOR_BOUNDARIES,
    "USE_GPU": config.USE_GPU_ACCELERATION,
    "SHOW_INFO": config.SHOW_DETAILED_INFO,
    "APPLY_INVERSION": config.APPLY_IMAGE_INVERSION,
    "APPLY_SOBEL": config.APPLY_SOBEL_FILTER,
    "APPLY_BILATERAL": config.APPLY_BILATERAL_FILTER,
    "MIN_AREA_THRESHOLD_MICRONS_SQUARED": config.MIN_ADIPOCYTE_AREA_MICRONS,
    "MAX_AREA_THRESHOLD_MICRONS_SQUARED": config.MAX_ADIPOCYTE_AREA_MICRONS,
    "GRID_CELL_SIZE": config.GRID_CELL_SIZE_MICRONS,
    "IOU_THRESHOLD": config.IOU_THRESHOLD,
    "MERGE_IOU_THRESHOLD": config.MERGE_IOU_THRESHOLD,
    "SCORE_THRESHOLD": config.CONFIDENCE_THRESHOLD,
    "SCALING_FACTOR": config.SCALING_FACTOR,
    "MANUAL_MPP": config.DEFAULT_MPP,
    "DESIRED_LEVEL": config.DESIRED_RESOLUTION_LEVEL,
    "NUM_IO_WORKERS": config.MAX_IO_WORKERS,
    "BATCH_INFERENCE_SIZE": config.BATCH_INFERENCE_SIZE,
    "ENABLE_GPU_LABEL_MAPPING": config.ENABLE_GPU_LABEL_MAPPING,
    "GPU_LABEL_MAPPING_THRESHOLD": config.GPU_LABEL_MAPPING_THRESHOLD,
    "GPU_MEMORY_LIMIT_GB": config.GPU_MEMORY_LIMIT_GB,
    "WINDOW_SIZE": config.WINDOW_SIZE,
    "STRIDE": config.STRIDE,
    "SAVE_ANNOTATED_IMAGE": config.SAVE_ANNOTATED_IMAGE,
    "SAVE_FULL_MASK": config.SAVE_FULL_MASK,
    "SAVE_SUMMARY_CSV": config.SAVE_SUMMARY_CSV,
    "SKIP_PROCESSED_IMAGES": config.SKIP_PROCESSED_IMAGES,
    "RETRY_FAILED_IMAGES": config.RETRY_FAILED_IMAGES,
    "SAVE_DISTANCE_COLORED_IMAGE": config.SAVE_DISTANCE_COLORED_IMAGE,
    "SAVE_SIZE_OUTLINED_DISTANCE_IMAGE": config.SAVE_SIZE_OUTLINED_DISTANCE_IMAGE,
    "SAVE_SIZE_ALPHA_DISTANCE_IMAGE": config.SAVE_SIZE_ALPHA_DISTANCE_IMAGE,
    "SAVE_SIZE_SYMBOL_DISTANCE_IMAGE": config.SAVE_SIZE_SYMBOL_DISTANCE_IMAGE,
    "SAVE_TUMOR_ZONE_OVERLAY_IMAGE": config.SAVE_TUMOR_ZONE_OVERLAY_IMAGE,
    "SHOW_GRID": config.SHOW_GRID,
    "SHOW_GRID_LABEL": config.SHOW_GRID_LABELS,
    "SHOW_ADIPOCYTE_IDS": config.SHOW_ADIPOCYTE_IDS,
    "TUMOR_ZONE_NEAR_UM": config.TUMOR_ZONE_NEAR_UM,
    "TUMOR_ZONE_INTERMEDIATE_UM": config.TUMOR_ZONE_INTERMEDIATE_UM,
    "TUMOR_ZONE_ALPHA": config.TUMOR_ZONE_ALPHA,
    "TUMOR_ZONE_NEAR_COLOR_RGB": config.TUMOR_ZONE_NEAR_COLOR_RGB,
    "TUMOR_ZONE_INTERMEDIATE_COLOR_RGB": config.TUMOR_ZONE_INTERMEDIATE_COLOR_RGB,
    "TUMOR_ZONE_DISTAL_COLOR_RGB": config.TUMOR_ZONE_DISTAL_COLOR_RGB,
    "DEBUG_MODE": config.DEBUG_MODE,
    "DEBUG_SAVE_UNPROCESSED_WINDOWS": config.DEBUG_SAVE_UNPROCESSED_WINDOWS,
    "ENABLE_GPU_MONITORING": config.ENABLE_GPU_MONITORING,
    "ENABLE_PROFILING": config.ENABLE_PROFILING,
    "CALCULATE_EXTENDED_PROPERTIES": config.CALCULATE_EXTENDED_PROPERTIES,
    "ANNOTATED_IMAGE_SCALE": config.ANNOTATED_IMAGE_SCALE,
    "GENERATE_INTERACTIVE_MAP": config.GENERATE_INTERACTIVE_MAP,
    "SAVE_FULL_POSTPROCESSED_IMAGE": config.SAVE_POSTPROCESSED_IMAGE,
    "SAVE_FULL_UNPROCESSED_IMAGE": config.SAVE_UNPROCESSED_IMAGE,
    # QuPath
    "ENABLE_QUPATH_EXPORT": config.ENABLE_QUPATH_EXPORT,
    "SAVE_QUPATH_GEOJSON": config.SAVE_QUPATH_GEOJSON,
    "SAVE_QUPATH_SCRIPT": config.SAVE_QUPATH_SCRIPT,
    "QUPATH_ANNOTATION_CLASS": config.QUPATH_ANNOTATION_CLASS,
    "INCLUDE_MEASUREMENTS_IN_QUPATH": config.INCLUDE_MEASUREMENTS_IN_QUPATH,
    # Tissue guidance
    "ENABLE_TISSUE_GUIDANCE": config.ENABLE_TISSUE_GUIDANCE,
    "ENABLE_TISSUE_GUIDANCE_CACHE": config.ENABLE_TISSUE_GUIDANCE_CACHE,
    "TISSUE_CACHE_DIR": config.TISSUE_CACHE_DIR,
    "TISSUE_MODEL_DIR": paths.TISSUE_MODEL_DIR,
    "TISSUE_MODEL_CHECKPOINT": paths.TISSUE_MODEL_CHECKPOINT,
    "TISSUE_OVERLAP_THRESHOLD": config.TISSUE_OVERLAP_THRESHOLD,
    "TISSUE_CONFIDENCE_THRESHOLD": config.TISSUE_CONFIDENCE_THRESHOLD,
    "TISSUE_NMS_THRESHOLD": config.TISSUE_NMS_THRESHOLD,
    "TISSUE_THUMBNAIL_SIZE": config.TISSUE_THUMBNAIL_SIZE,
    "ENABLE_MULTI_REGION_OPTIMIZATION": config.ENABLE_MULTI_REGION_OPTIMIZATION,
    "SAVE_REGION_STATISTICS": config.SAVE_REGION_STATISTICS,
    # ROI guidance
    "ENABLE_ROI_GUIDANCE": config.ENABLE_ROI_GUIDANCE,
    "ROI_POLYGON_FILE": config.ROI_POLYGON_FILE,
    "ROI_THUMBNAIL_MAX_DIM": config.ROI_THUMBNAIL_MAX_DIM,
    "ROI_MIN_COVERAGE": config.ROI_MIN_COVERAGE,
    # Paths
    "IMAGE_PATH": paths.IMAGE_PATH,
    "MODEL_DIR": paths.ADIPOCYTE_MODEL_DIR,
    "MODEL_CHECKPOINT": paths.ADIPOCYTE_MODEL_CHECKPOINT,
    "TUMOUR_MODEL_DIR": paths.TUMOR_MODEL_DIR,
    "TUMOUR_MODEL_CHECKPOINT": paths.TUMOR_MODEL_CHECKPOINT,
    "OUTPUT_DIR": paths.OUTPUT_DIR,
    "SUMMARY_CSV_PATH": paths.SUMMARY_CSV_PATH,
}

globals().update(_COMPAT_ALIASES)

CPU_MEM_BW_GBPS = 40
PCIE_EFFECTIVE_GBPS = 28
GPU_SPEED_MARGIN = 0.9
GPU_LUT_FRACTION_MAX = 0.5

# Sorting configuration
SORT_BY_FILE_SIZE = False
SIZE_SORT_ASCENDING = False
SORT_BY_RESOLUTION = True
RES_SORT_ASCENDING = False
RESOLUTION_CACHE_PATH = "resolution_cache.json"
