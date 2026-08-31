Back to [Documentation Index](index.md)

# Configuration

Use this page as the canonical guide to AdiFind configuration. It combines runtime environment variables, model and cache settings, and the `Config` options that shape processing behavior.

---

## How Configuration Works

AdiFind uses a global configuration singleton together with CLI overrides and a small set of environment variables.

```python
from config import config, paths

print(config.CONFIDENCE_THRESHOLD)
config.CONFIDENCE_THRESHOLD = 0.90
```

Use the layers this way:

1. CLI flags for run-specific choices such as `--tissue_guidance` or `--output_dir`
2. `config.py` or Python API overrides for defaults and advanced tuning
3. Environment variables for model paths, cache location, and deployment-specific behavior

See [CLI Reference](cli-reference.md) for flags and [Python API](python-api.md) for programmatic usage.

---

## Environment Variables

### Model paths

These variables override the default model-download behavior and point AdiFind at local model directories.
Each configured directory must contain the canonical AdiFind checkpoint filename for that model.

| Variable | Default | Description |
|:---------|:--------|:------------|
| `ADIFIND_ADIPOCYTE_MODEL_DIR` | auto-download if authenticated or public | Directory containing adipocyte model weights |
| `ADIFIND_TUMOR_MODEL_DIR` | auto-download if authenticated or public | Directory containing tumor model weights |
| `ADIFIND_TISSUE_MODEL_DIR` | auto-download if authenticated or public | Directory containing tissue-guidance model weights |

Example:

```bash
export ADIFIND_ADIPOCYTE_MODEL_DIR="/models/adipocyte"
export ADIFIND_TUMOR_MODEL_DIR="/models/tumor"
export ADIFIND_TISSUE_MODEL_DIR="/models/tissue"
```

PowerShell:

```powershell
$env:ADIFIND_ADIPOCYTE_MODEL_DIR = "C:\models\adipocyte"
$env:ADIFIND_TUMOR_MODEL_DIR = "C:\models\tumor"
$env:ADIFIND_TISSUE_MODEL_DIR = "C:\models\tissue"
```

Canonical checkpoint filenames:

- adipocyte: `adifind_adipocyte.pth`
- tumor: `adifind_tumor.pth`
- tissue guidance: `adifind_tissue_guidance.pth`

### Download and cache behavior

| Variable | Default | Description |
|:---------|:--------|:------------|
| `ADIFIND_HF_REPO` | `letarg/adifind` | HuggingFace repository used for model downloads |
| `ADIFIND_CACHE_DIR` | platform default | Override the local model cache directory |
| `HF_TOKEN` | unset | Optional Hugging Face token for private model repositories |

The current AdiFind model repo is private. Until it is public, unauthenticated users should prefer local canonical model directories.

Default cache locations:

| Platform | Default path |
|:---------|:-------------|
| Linux / macOS | `~/.cache/adifind/models/` |
| Windows | `%LOCALAPPDATA%\adifind\models\` |

### Output and runtime behavior

| Variable | Default | Description |
|:---------|:--------|:------------|
| `ADIFIND_OUTPUT_DIR` | `adifind_output` | Default output directory |
| `ADIFIND_USE_GPU` | `true` | Container-entrypoint GPU toggle used mainly by the bundled CUDA/NVIDIA container startup path |
| `ADIFIND_NO_BANNER` | unset | Suppress the startup banner |
| `ADIFIND_NO_ANIM` | unset | Disable banner animation |
| `ADIFIND_ANIM_DELAY` | built-in default | Banner animation delay |
| `ADIFIND_ANIM_STEP` | built-in default | Banner animation step size |

### OpenSlide on Windows

AdiFind expects the OpenSlide DLL directory at `C:\OpenSlide\bin` unless you override it:

```powershell
$env:OPENSLIDE_PATH = "C:\OpenSlide\bin"
```

### Docker defaults

The Docker images always pre-set the GPU toggle and auto-detect canonical bundled model files under `/app/models` when present:

```dockerfile
ENV ADIFIND_USE_GPU=true
ENV PYTHONPATH=/app/code
```

The GPU container installs OpenSlide through the pip-native `openslide-bin` path instead of Conda OpenSlide packages, which avoids base-image Conda solver conflicts while preserving runtime OpenSlide support.

If you set `ADIFIND_*_MODEL_DIR` explicitly inside a container, the referenced directory must contain the canonical filename for that model. Legacy checkpoint filenames are not supported.

For deployment examples, see [Deployment](deployment.md).

---

## Core Config Options

### Feature toggles

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `ENABLE_TUMOR_SEGMENTATION` | bool | `True` | Master toggle for tumor detection capability |
| `SHOW_TUMOR_BOUNDARIES` | bool | `True` | Display tumor boundaries on annotated images |
| `USE_GPU_INFERENCE` | bool | `True` | Run Detectron2 inference on a GPU-capable PyTorch backend when available |
| `USE_CUPY` | bool | `True` | Enable CuPy-backed distance transforms and other array-heavy GPU ops |
| `USE_GPU_PREPROCESSING` | bool | `True` | Enable GPU Sobel/inversion preprocessing when those filters are active |
| `SHOW_DETAILED_INFO` | bool | `True` | Enable detailed logging output |

Legacy note: `USE_GPU_ACCELERATION` and `ENABLE_GPU_ACCELERATION` remain as compatibility accessors for older code, but they are not the canonical configuration names anymore.

### Image preprocessing

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `APPLY_IMAGE_INVERSION` | bool | `True` | Apply image inversion before inference |
| `APPLY_SOBEL_FILTER` | bool | `True` | Apply Sobel edge detection before inference |
| `APPLY_BILATERAL_FILTER` | bool | `False` | Apply bilateral smoothing |

### Analysis parameters

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `MIN_ADIPOCYTE_AREA_MICRONS` | int | `250` | Minimum adipocyte area in square microns |
| `MAX_ADIPOCYTE_AREA_MICRONS` | int | `25000` | Maximum adipocyte area in square microns |
| `GRID_CELL_SIZE_MICRONS` | int | `1100` | Grid cell size for spatial analysis |
| `IOU_THRESHOLD` | float | `0.3` | IoU threshold for detection matching |
| `MERGE_IOU_THRESHOLD` | float | `0.2` | IoU threshold for merging detections across windows |
| `CONFIDENCE_THRESHOLD` | float | `0.80` | Minimum detection confidence score |
| `SCALING_FACTOR` | float | `0.3` | Default scaling factor for image output |
| `DEFAULT_MPP` | float | `0.50` | Fallback microns-per-pixel when metadata is missing |
| `PIXEL_SIZE_UM` | float | `0.50` | Pixel size in microns |

### Processing

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `DESIRED_RESOLUTION_LEVEL` | int | `0` | Pyramid level to read from |
| `MAX_IO_WORKERS` | int | `25` | Maximum I/O worker threads |
| `BATCH_INFERENCE_SIZE` | int | `4` | Windows per GPU inference batch |
| `WINDOW_SIZE` | tuple | `(2000, 2000)` | Default window size in pixels |
| `STRIDE` | tuple | `(1700, 1700)` | Default sliding-window stride |
| `ENABLE_GPU_LABEL_MAPPING` | bool | `False` | Use GPU for label remapping when beneficial |
| `GPU_LABEL_MAPPING_THRESHOLD` | int | `1000` | Minimum labels to trigger GPU mapping |
| `GPU_MEMORY_LIMIT_GB` | int | `30` | GPU memory limit used for headroom planning |
| `FORCE_CPU_LABEL_MAPPING` | bool | `True` | Force CPU label mapping |

---

## Outputs and Visualization

### Output control

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `SAVE_ANNOTATED_IMAGE` | bool | `config.py` | Save the base annotated TIFF output |
| `SAVE_FULL_MASK` | bool | `False` | Save the full-resolution segmentation mask |
| `SAVE_SUMMARY_CSV` | bool | `True` | Save per-adipocyte CSV results |
| `SKIP_PROCESSED_IMAGES` | bool | `True` | Skip images with existing output directories |
| `RETRY_FAILED_IMAGES` | bool | `False` | Retry failed images when resuming a batch |
| `SAVE_DISTANCE_COLORED_IMAGE` | bool | `False` | Save the distance-colored tumor visualization |
| `SAVE_SIZE_OUTLINED_DISTANCE_IMAGE` | bool | `False` | Save the size-outlined distance variant |
| `SAVE_SIZE_ALPHA_DISTANCE_IMAGE` | bool | `False` | Save the size-alpha distance variant |
| `SAVE_SIZE_SYMBOL_DISTANCE_IMAGE` | bool | `False` | Save the size-symbol distance variant |
| `SAVE_TUMOR_ZONE_OVERLAY_IMAGE` | bool | `config.py` | Save tumor-zone overlays independently of the base annotated TIFF |

For the CLI, `--save_image_annotation` and `--skip_image_annotation` override `SAVE_ANNOTATED_IMAGE` for the current run only. If the base annotated TIFF is disabled, tumor-zone overlays remain controlled by `SAVE_TUMOR_ZONE_OVERLAY_IMAGE`.

### CSV export

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `CALCULATE_EXTENDED_PROPERTIES` | bool | `False` | Include extended morphological properties |

### Annotated image settings

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `ANNOTATED_IMAGE_SCALE` | float | `1.0` | Internal scale for annotated image generation |
| `ANNOTATED_IMAGE_SAVE_MODE` | string | `high_quality` | `fast`, `balanced`, or `high_quality` |
| `BENCHMARK_IMAGE_SAVING` | bool | `False` | Benchmark image-saving methods |

### Visualization

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `SHOW_GRID` | bool | `False` | Show the analysis grid |
| `SHOW_GRID_LABELS` | bool | `False` | Show labels on grid cells |
| `SHOW_ADIPOCYTE_IDS` | bool | `True` | Show adipocyte identifiers |
| `MIN_TUMOR_AREA_PIXELS` | int | `100` | Minimum tumor area to count as real |
| `MIN_TUMOR_PIXELS_FOR_DISTANCE` | int | `50` | Minimum tumor pixels required for distance computation |
| `TUMOR_ZONE_NEAR_UM` | float | `1500.0` | Near tumor zone boundary |
| `TUMOR_ZONE_INTERMEDIATE_UM` | float | `5000.0` | Intermediate tumor zone boundary |
| `TUMOR_ZONE_ALPHA` | float | `0.25` | Tumor-zone overlay alpha |
| `TUMOR_ZONE_NEAR_COLOR_RGB` | tuple | `(255, 255, 0)` | Near zone color |
| `TUMOR_ZONE_INTERMEDIATE_COLOR_RGB` | tuple | `(0, 255, 0)` | Intermediate zone color |
| `TUMOR_ZONE_DISTAL_COLOR_RGB` | tuple | `(0, 128, 255)` | Distal zone color |

---

## Feature-Specific Settings

### QuPath export

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `ENABLE_QUPATH_EXPORT` | bool | `True` | Enable QuPath GeoJSON annotation export |
| `SAVE_QUPATH_GEOJSON` | bool | `True` | Save the QuPath GeoJSON annotation file |
| `SAVE_QUPATH_SCRIPT` | bool | `True` | Reserved compatibility setting; QuPath Groovy script export is not implemented in the current runtime path |
| `QUPATH_ANNOTATION_CLASS` | string | `Adipocyte` | Class name used for annotations |
| `INCLUDE_MEASUREMENTS_IN_QUPATH` | bool | `True` | Include measurements in GeoJSON properties |

For the CLI, `--save_qupath_annotation` and `--skip_qupath_annotation` override the QuPath GeoJSON export settings for the current run only.

### Tissue guidance

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `ENABLE_TISSUE_GUIDANCE` | bool | `True` | Enable tissue-guided processing |
| `ENABLE_TISSUE_GUIDANCE_CACHE` | bool | `True` | Cache tissue-detection results |
| `TISSUE_CACHE_DIR` | string | `tissue_cache` | Directory for cached tissue data |
| `TISSUE_OVERLAP_THRESHOLD` | float | `0.3` | Minimum window and tissue overlap |
| `TISSUE_CONFIDENCE_THRESHOLD` | float | `0.5` | Tissue detection confidence threshold |
| `TISSUE_NMS_THRESHOLD` | float | `0.3` | Non-maximum suppression threshold |
| `TISSUE_THUMBNAIL_SIZE` | int | `2048` | Thumbnail size for tissue detection |
| `TISSUE_DETECTION_DOWNSAMPLE` | int | `32` | Downsample factor for tissue analysis |
| `MIN_TISSUE_COVERAGE` | float | `0.1` | Minimum tissue coverage to include a region |
| `SAVE_TISSUE_DETECTION` | bool | `True` | Save tissue-detection outputs |
| `ENABLE_MULTI_REGION_OPTIMIZATION` | bool | `True` | Optimize for slides with multiple tissue regions |
| `SAVE_REGION_STATISTICS` | bool | `True` | Save per-region statistics |
| `SAVE_TISSUE_WINDOW_GRID_THUMBNAIL` | bool | `False` | Save the debug grid thumbnail |

### ROI guidance

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `ENABLE_ROI_GUIDANCE` | bool | `False` | Enable ROI-based window filtering |
| `ROI_POLYGON_FILE` | string | `None` | Path to a saved ROI polygon JSON |
| `ROI_THUMBNAIL_MAX_DIM` | int | `2048` | Maximum ROI thumbnail dimension |
| `ROI_MIN_COVERAGE` | float | `0.2` | Minimum ROI overlap required for a window |

---

## Performance, Debugging, and Memory

### Debug options

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `DEBUG_MODE` | bool | `False` | Enable debug mode |
| `DEBUG_SAVE_UNPROCESSED_WINDOWS` | bool | `False` | Save raw windows in debug mode |
| `ENABLE_GPU_MONITORING` | bool | `True` | Monitor GPU memory and utilization |
| `ENABLE_PROFILING` | bool | `False` | Enable detailed profiling |

### GPU-related options

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `USE_GPU_INFERENCE` | bool | `True` | Controls model inference device selection |
| `USE_CUPY` | bool | `True` | Controls CuPy-backed distance transforms and array ops |
| `USE_GPU_PREPROCESSING` | bool | `True` | Controls GPU Sobel and inversion preprocessing |
| `ENABLE_GPU_LABEL_MAPPING` | bool | `False` | Use GPU for label remapping when beneficial |
| `GPU_MORPHOLOGY_THRESHOLD` | int | `1000000` | Use GPU morphology above this pixel count |
| `GPU_STATISTICS_THRESHOLD` | int | `500000` | Use GPU statistics above this pixel count |
| `GPU_CONNECTED_COMPONENTS_THRESHOLD` | int | `1000000` | Use GPU connected components above this pixel count |
| `ENABLE_GPU_STREAMS` | bool | `True` | Use CUDA streams |
| `GPU_STREAM_COUNT` | int | `20` | Number of CUDA streams |
| `ENABLE_GPU_MEMORY_PROFILING` | bool | `False` | Profile GPU memory usage |
| `GPU_CLEANUP_INTERVAL` | int | `50` | Clean GPU memory every N windows |
| `GPU_CLEANUP_AGGRESSIVE` | bool | `False` | Aggressive GPU cleanup mode |

### Async I/O

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `ENABLE_ASYNC_IO` | bool | `True` | Enable asynchronous I/O for window reading |
| `ASYNC_CACHE_SIZE_MB` | int | `256` | Async I/O cache size |
| `ASYNC_MAX_WORKERS` | int | `4` | Number of async I/O worker threads |
| `ASYNC_PREFETCH_SIZE` | int | `12` | Number of windows to prefetch |

### Memory efficiency

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `MEMORY_EFFICIENT_MODE` | bool | `True` | Enable memory-efficient processing |
| `MASK_CHUNK_SIZE` | int | `2048` | Chunk size for mask processing |
| `INCREMENTAL_PROPERTY_COLLECTION` | bool | `True` | Collect properties incrementally |
| `MAX_FULL_MASK_PIXELS` | int | `500000000` | Maximum mask pixels before chunked mode |
| `INPLACE_LABEL_REMAPPING` | bool | `True` | Use in-place label remapping |
| `PARALLEL_CHUNK_PROCESSING` | bool | `True` | Parallelize chunk processing |
| `CHUNK_WORKERS` | int | `12` | Chunk-processing worker threads |

### Low-memory mode

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `LOW_MEMORY_MODE` | bool | `False` | Enable low-memory mode |
| `USE_MEMMAP_MASK` | bool | `False` | Use disk-backed `numpy.memmap` masks |
| `LOW_MEMORY_THRESHOLD_GB` | int | `96` | Auto-enable low-memory mode below this RAM level |
| `LOW_MEMORY_MAX_ANNOTATED_PIXELS` | int | `150000000` | Max annotated-image pixels in low-memory mode |

---

## Batch and Compatibility Details

### Batch sorting constants

These are module-level constants in `config.py`, not `Config` attributes.

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `SORT_BY_FILE_SIZE` | bool | `False` | Sort batch inputs by file size |
| `SIZE_SORT_ASCENDING` | bool | `False` | Sort file sizes ascending |
| `SORT_BY_RESOLUTION` | bool | `True` | Sort batch inputs by pixel area |
| `RES_SORT_ASCENDING` | bool | `False` | Sort resolution ascending |
| `RESOLUTION_CACHE_PATH` | string | `resolution_cache.json` | Cache file for resolution metadata |

### GPU transfer thresholds

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `CPU_MEM_BW_GBPS` | int | `40` | Assumed CPU memory bandwidth |
| `PCIE_EFFECTIVE_GBPS` | int | `28` | Assumed PCIe effective bandwidth |
| `GPU_SPEED_MARGIN` | float | `0.9` | GPU speed advantage margin |
| `GPU_LUT_FRACTION_MAX` | float | `0.5` | Maximum GPU memory fraction for lookup tables |

### Backward compatibility

AdiFind maintains compatibility aliases in `_COMPAT_ALIASES`, including legacy spellings such as `TUMOUR`, so older scripts can continue to resolve to current configuration names.

---

## Next Steps

- [CLI Workflows](cli-workflows.md) for common run patterns
- [CLI Reference](cli-reference.md) for flag-by-flag behavior
- [Deployment](deployment.md) for Docker and HPC examples
- [Performance Tuning](performance-tuning.md) for optimization guidance

---

Back to [Documentation Index](index.md)
