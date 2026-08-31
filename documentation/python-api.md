Back to [Documentation Index](index.md)

# Python API

Use AdiFind programmatically from Python scripts, Jupyter notebooks, or custom pipelines.

---

## Quick Example

```python
import argparse
from config import config, paths
from main import process_single_image

# Configure analysis parameters
args = argparse.Namespace(
    image_path="../example_data/K106942.svs",
    output_dir="my_output/",
    window_size=[2048, 2048],
    stride=[1024, 1024],
    tissue_guidance=True,
    tumor_segmentation=True,
    save_distance_map=True,
    extended_properties=False,
    annotated_scale=0.3,
    save_mode="balanced",
    save_image_annotation=False,
    skip_image_annotation=False,
    save_qupath_annotation=False,
    skip_qupath_annotation=False,
    roi_freehand=False,
    roi_polygon_file=None,
    roi_max_dim=2048,
    roi_min_coverage=0.2,
    low_memory=False,
    memmap_mask=False,
    debug=None,
    gpu_id=0,
    disable_gpu_accel=False,
    disable_gpu_ops=False,
    disable_gpu_preprocessing=False,
    batch_size=None,
    save_tissue_window_grid=False,
    profiling=False,
    benchmark_saving=False,
    show_adipocyte_ids=False,
    hide_adipocyte_ids=False,
    show_grid=False,
    hide_grid=False,
    min_area=None,
    max_area=None,
)

# Update config from args
from configuration_manager import update_config_from_args
update_config_from_args(args)

# Process
result = process_single_image(
    image_path=args.image_path,
    args=args,
    output_dir=args.output_dir
)

print(f"Detected {result['total_adipocytes']} adipocytes in {result['total_time']:.1f}s")
```

---

## Core Function: `process_single_image`

```python
process_single_image(image_path: str, args: Namespace, output_dir: str) -> dict
```

Runs the full pipeline on a single image and returns a results dictionary:

| Key | Type | Description |
|:----|:-----|:------------|
| `image_name` | string | Slide filename (without extension) |
| `image_path` | string | Full path to the slide |
| `output_dir` | string | Output directory |
| `total_adipocytes` | int | Number of detected adipocytes |
| `total_time` | float | Processing time in seconds |
| `total_windows` | int | Number of windows processed |
| `num_tumors` | int | Number of tumors detected |
| `median_size_microns` | float | Median adipocyte area (µm²) |
| `average_size_microns` | float | Mean adipocyte area (µm²) |

---

## Configuration Singleton

The global `config` and `paths` objects can be modified before calling `process_single_image`:

```python
from config import config, paths

# Adjust analysis parameters
config.CONFIDENCE_THRESHOLD = 0.90
config.MIN_ADIPOCYTE_AREA_MICRONS = 500
config.MAX_ADIPOCYTE_AREA_MICRONS = 20000

# Adjust performance
config.BATCH_INFERENCE_SIZE = 8
config.MAX_IO_WORKERS = 32
config.CHUNK_WORKERS = 16
config.USE_GPU_INFERENCE = True
config.USE_CUPY = True
config.USE_GPU_PREPROCESSING = True

# Disable specific outputs
config.SAVE_QUPATH_GEOJSON = False
config.SAVE_ANNOTATED_IMAGE = False

# Tumor-zone overlays remain independent
config.SAVE_TUMOR_ZONE_OVERLAY_IMAGE = True

# CLI overrides such as --save_image_annotation / --skip_image_annotation
# and --save_qupath_annotation / --skip_qupath_annotation take precedence
# for a single run when using the command-line interface.

# Set model paths
paths.ADIPOCYTE_MODEL_DIR = "/path/to/models/adipocyte/"
paths.TUMOR_MODEL_DIR = "/path/to/models/tumor/"
```

See [Configuration](configuration.md) for all options.

---

## ImageHandler

Read regions from whole-slide images. `ImageHandler` uses a three-tier dispatch: **OpenSlide** for native WSI formats, **slideio/GDAL** for TIFF variants OpenSlide cannot open (OME-TIFF, BigTIFF), and **Pillow** for standard raster images.

```python
from image_processing import ImageHandler, get_mpp

# Open a slide (format auto-detected)
handler = ImageHandler("slide.svs", desired_level=0)   # OpenSlide path
handler = ImageHandler("image.ome.tif", desired_level=0)  # slideio/GDAL path
handler = ImageHandler("photo.png", desired_level=0)    # Pillow path

# Read slide metadata
mpp = get_mpp("slide.svs")
print(f"Slide: {handler.width}×{handler.height} at {mpp} µm/px")

# Read a region (x, y, level, size)
region = handler.read_region((1000, 2000), 0, (2048, 2048))

# Clean up
handler.close()
```

---

## Model Loading

Load Detectron2 models directly:

```python
from models import configure_adipocyte_model, configure_tumor_model

# Load adipocyte model (downloads from Hugging Face if access is available)
predictor = configure_adipocyte_model()

# Load tumor model
tumor_predictor = configure_tumor_model()

# Load with custom path.
# The directory must contain adifind_adipocyte.pth.
predictor = configure_adipocyte_model(
    model_dir="/path/to/model/"
)
```

---

## Model Download

Programmatic model management:

```python
from model_downloader import ensure_model, get_cache_dir

# Get cache directory
cache = get_cache_dir()
print(f"Models cached at: {cache}")

# Ensure a model is available locally.
# If the Hugging Face repo is private, this requires HF_TOKEN or a prior login.
model_path = ensure_model("adipocyte")
```

---

## Batch Processing

```python
from batch_processing import BatchProcessor

processor = BatchProcessor(
    image_files=["slide1.svs", "slide2.svs", "slide3.svs"],
    args=args,
    output_dir="batch_output/"
)

results = processor.process_all()

for r in results:
    print(f"{r['image_name']}: {r['total_adipocytes']} adipocytes")
```

---

## Jupyter Notebook Integration

A complete interactive quickstart notebook is provided at `notebooks/quickstart.ipynb`. It covers:

1. **Installation verification** — Check PyTorch, Detectron2, OpenSlide
2. **Model download** — Ensure all three models are available
3. **Single-image processing** — Full pipeline with configuration
4. **Result visualization** — Annotated thumbnails, histograms, distance plots
5. **CSV analysis** — Pandas-based exploration of per-adipocyte data
6. **Batch processing** — Process multiple slides with comparative analysis

To run the notebook:

```bash
conda activate adifind
pip install -e ".[notebook]"  # Ensure Jupyter extras are installed
jupyter notebook notebooks/quickstart.ipynb
```

---

## Utility Functions

### Sliding Window Generation

```python
from image_processing import generate_sliding_windows, calculate_optimal_window_params

# Auto-calculate window parameters from MPP
window_size, stride = calculate_optimal_window_params(
    mpp=0.25,
    args_window_size=[2048, 2048],
    args_stride=[1024, 1024]
)

# Generate window coordinates
for x, y in generate_sliding_windows(width, height, window_size, stride):
    print(f"Window at ({x}, {y})")
```

### System Monitoring

```python
from system_utils import monitor, memory_manager

# Check system resources
monitor.log_system_status()

# Flush GPU memory
from system_utils import flush_gpu_memory
flush_gpu_memory()
```

---

## See Also

- [Configuration](configuration.md) — All configurable options
- [Output Reference](output-reference.md) — Output file formats
- [Architecture](architecture.md) — Module structure and data flow

---

Back to [Documentation Index](index.md)
