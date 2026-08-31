Back to [Documentation Index](index.md)

# Architecture

Technical reference for AdiFind's code architecture, data flow, algorithms, and design patterns.

---

## Module Map

All application code lives in `code/`. There is no package nesting — every module is a top-level file.

| Module | Role |
|:-------|:-----|
| `main.py` | CLI entry point. `main()` → `process_single_image()`. Orchestrates the full pipeline. |
| `config.py` | Global singletons `config` (Config) and `paths` (Paths). All tunables live here. |
| `argument_parser.py` | argparse definitions. Returns `args` consumed by `main()`. |
| `configuration_manager.py` | Maps CLI args to config. `update_config_from_args()`. |
| `models.py` | Detectron2 model loading. `configure_adipocyte_model`, `CustomBatchPredictor`. |
| `core_processing.py` | Sliding-window inference loop, union-find merging, label mapping. |
| `image_processing.py` | `ImageHandler` — three-tier dispatch: OpenSlide → slideio/GDAL → Pillow. MPP extraction, window generation. |
| `tissue_guided_processing.py` | Tissue region detection to skip empty glass. |
| `tumor_detection.py` | Tumor segmentation and distance-to-tumor computation (CuPy/SciPy). |
| `visualization.py` | Annotated thumbnails, heatmaps, QuPath GeoJSON, CSV export. |
| `batch_processing.py` | `BatchProcessor` with JSON state files for resume/retry. |
| `system_utils.py` | `SystemMonitor`, `MemoryManager`, GPU memory flushing. |
| `statistics_utils.py` | Statistical computations for results. |
| `logging_utils.py` | Logging configuration. |
| `progress_utils.py` | Progress bar utilities. |
| `roi_guidance.py` | Interactive and file-based ROI selection. |
| `figure_export_policy.py` | Figure export settings and policies. |
| `model_downloader.py` | HuggingFace model download and caching. |
| `adifind_desktop.py` | PySide6-based desktop GUI. |
| `gpu_kernels.py` | Custom GPU kernel definitions. |
| `legacy_compatibility.py` | Backward compatibility wrappers. |
| `accelerated_visualization.py` | GPU-accelerated visualization routines. |

### Optimization Modules (`code/optimizations/`)

| Module | Role |
|:-------|:-----|
| `async_io.py` | Lightweight prefetching with 2–4 workers |
| `async_integration.py` | Drop-in async I/O wrapper |
| `batch_inference.py` | Batched window processing |
| `gpu_acceleration.py` | GPU morphology, filters, histogram analysis |
| `gpu_label_mapping.py` | Chunked GPU label mapping |
| `memory_streaming.py` | HDF5-backed streaming for gigapixel images |
| `performance_optimizer.py` | Auto-tuning coordinator |
| `tensor_core_acceleration.py` | FP16 inference on tensor cores |
| `tissue_cache.py` | MD5-based tissue result caching |

---

## Data Flow

```
CLI args
  │
  ▼
main() ──► parse_arguments() ──► update_config_from_args()
  │
  ▼
ImageHandler(slide_path)
  │  ├── OpenSlide can open? ──► OpenSlide (WSI formats)
  │  ├── Is .tif/.tiff? ──► slideio + GDAL (OME-TIFF, BigTIFF)
  │  └── Fallback ──► Pillow (raster images)
  │
  ├──► get_mpp() ──► calculate_optimal_window_params()
  │
  ├──► [Tissue Guidance] ──► TissueGuidanceDetector
  │         │                    ├── Load tissue model
  │         │                    ├── Run on thumbnail (2048px)
  │         │                    └── Return tissue bounding boxes
  │         │
  │         └──► Filter windows to tissue regions only
  │
  ├──► [ROI Guidance] ──► Filter windows to ROI polygon
  │
  ├──► [Tumor Segmentation] ──► configure_tumor_model()
  │         │                       ├── Run on 2000×2000 thumbnail
  │         │                       ├── Build binary tumor mask
  │         │                       └── Compute distance transform
  │         │
  │         └──► Distance map (GPU CuPy or CPU SciPy)
  │
  ▼
generate_sliding_windows()
  │
  ▼
process_all_windows() ──► BatchedInferenceManager
  │                           ├── Read window regions
  │                           ├── Preprocess (invert, Sobel, bilateral)
  │                           ├── Batch GPU inference (Detectron2)
  │                           ├── Filter by confidence + area
  │                           └── Write to full mask + union-find
  │
  ▼
Union-Find Merge
  │  ├── find() with path compression
  │  └── union() when IoU > MERGE_IOU_THRESHOLD
  │
  ▼
Label Remapping ──► regionprops ──► Area/distance statistics
  │
  ▼
Visualization & Export
  ├── export_results_csv()
  ├── export_qupath_annotations()
  ├── annotate_image_with_adipocytes()
  ├── Distance-colored visualization
  └── Tumor zone overlay
```

---

## Key Algorithms

### Sliding Window with Union-Find Merging

AdiFind processes gigapixel images through overlapping sliding windows. Each window is processed independently by Detectron2, producing instance masks. The challenge is merging detections that span window boundaries.

**Strategy:**

1. Each detected adipocyte gets a unique label in a global mask
2. When a new detection overlaps an existing label, compute IoU
3. If IoU > `MERGE_IOU_THRESHOLD` (0.2), **union** the labels using union-find
4. After all windows: apply the union-find mapping to get final merged labels
5. Compute `regionprops` on the merged mask

**Union-Find** uses path compression for efficient merging:

```python
def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # Path compression
        x = parent[x]
    return x

def union(parent, a, b, overlap_ratio):
    if overlap_ratio > config.MERGE_IOU_THRESHOLD:
        root_a, root_b = find(parent, a), find(parent, b)
        if root_a != root_b:
            parent[root_b] = root_a
```

### Label Mapping Strategies

For large slides, the label mapping step can be memory-intensive. AdiFind provides multiple strategies:

| Strategy | When Used | Description |
|:---------|:----------|:------------|
| **Lookup table** | Max label < 1 GB | Vectorized NumPy lookup array |
| **Chunked** | Max label ≥ 1 GB | Process mask in row chunks (`MASK_CHUNK_SIZE`) |
| **In-place chunked** | `INPLACE_LABEL_REMAPPING = True` | Modify mask in-place to save memory |
| **GPU mapping** | `ENABLE_GPU_LABEL_MAPPING = True` | CuPy-based GPU label mapping |

### Distance Transform

For tumor distance computation, AdiFind uses Euclidean distance transform on the binary tumor mask:

- **GPU path:** CuPy `distance_transform_edt` — fast, requires CuPy plus a compatible CuPy backend
- **CPU path:** SciPy `distance_transform_edt` — automatic fallback

The distance is computed at thumbnail resolution and scaled to physical distance (µm) using the slide's MPP value.

---

## Model Architecture

All three models use the same Detectron2 architecture:

| Property | Value |
|:---------|:------|
| **Architecture** | Mask R-CNN |
| **Backbone** | ResNeXt-101-32x8d-FPN |
| **Config** | `mask_rcnn_X_101_32x8d_FPN_3x.yaml` |
| **Framework** | Detectron2 (Facebook Research) |

| Model | Classes | Score Threshold | Max Detections |
|:------|:--------|:----------------|:---------------|
| Adipocyte | 1 | 0.80 | 100,000 |
| Tumor | 1 | 0.20 | 1,000 |
| Tissue | 1 | *configurable* | 1,000 |

The `CustomBatchPredictor` extends Detectron2's `DefaultPredictor` to support batched inference — processing multiple windows in a single GPU call.

---

## Design Patterns

### Global Configuration Singleton

```python
# config.py
config = Config()   # Global instance
paths = Paths()     # Global instance

# Any module can import and read/write
from config import config
config.CONFIDENCE_THRESHOLD = 0.90
```

### Optional Dependency Guard

Every optional import uses try/except with a module-level flag:

```python
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Usage
if CUPY_AVAILABLE and config.USE_CUPY:
    result = gpu_function(data)
else:
    result = cpu_fallback(data)
```

This pattern is used for: CuPy, slideio, psutil, colorama.

### GPU Fallback

All GPU operations follow: **try GPU → catch error → CPU fallback**:

```python
try:
    if CUPY_AVAILABLE:
        result = gpu_operation(data)
    else:
        raise RuntimeError("No GPU")
except (RuntimeError, MemoryError):
    result = cpu_operation(data)
```

### Memory-Mapped Mask

For gigapixel slides, the full mask can exceed RAM. The memmap pattern:

```python
if config.USE_MEMMAP_MASK:
    mask = np.memmap(temp_file, dtype=np.uint32, mode='w+', shape=(h, w))
    cleanup = lambda: os.unlink(temp_file)
else:
    mask = np.zeros((h, w), dtype=np.uint32)
    cleanup = lambda: None
```

### Backward Compatibility Aliases

`config.py` maintains `_COMPAT_ALIASES` mapping old names (including British spellings) to current attributes:

```python
_COMPAT_ALIASES = {
    "ENABLE_TUMOUR_SEGMENTATION": config.ENABLE_TUMOR_SEGMENTATION,
    "SCORE_THRESHOLD": config.CONFIDENCE_THRESHOLD,
    ...
}
globals().update(_COMPAT_ALIASES)
```

For split configuration renames, `Config` can also provide computed legacy accessors so an older name can map onto multiple new runtime flags.

---

## Entry Points

| Entry Point | Module | Description |
|:------------|:-------|:------------|
| `adifind` | `main:main` | Main CLI (installed via pip) |
| `adifind-curate` | `result_processing.adifind_image_curation_tool:main` | Image curation tool |
| `python main.py` | `main.py` | Direct script execution |
| `python adifind_desktop.py` | `adifind_desktop.py` | Desktop GUI |

---

## See Also

- [Configuration](configuration.md) — All tunable options
- [Performance Tuning](performance-tuning.md) — Optimization details
- [Contributing](contributing.md) — Development workflow and code style

---

Back to [Documentation Index](index.md)
