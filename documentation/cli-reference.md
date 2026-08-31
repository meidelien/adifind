Back to [Documentation Index](index.md)

# CLI Reference

Complete reference for all AdiFind command-line flags. Run `adifind --help` for a quick summary.



---

## Synopsis

```text
adifind [image_path] [options]
```

Equivalent source-tree command:

```text
python code/main.py [image_path] [options]
```

---

## Image and Path Options

| Flag | Type | Default | Description |
|:-----|:-----|:--------|:------------|
| `image_path` | positional | required for most runs | Path to a WSI file or directory of images |
| `--output_dir PATH` | string | auto-generated | Output directory, defaulting to `adifind_results_{name}_{timestamp}/` |
| `--config_file PATH` | string | `None` | Path to a custom configuration file |

---

## Processing Parameters

| Flag | Type | Default | Description |
|:-----|:-----|:--------|:------------|
| `--window_size W H` | int int | `2048 2048` | Sliding window size in pixels |
| `--stride SX SY` | int int | `1024 1024` | Sliding-window stride in pixels |
| `--min_area AREA` | float | config default | Minimum adipocyte area in square microns |
| `--max_area AREA` | float | config default | Maximum adipocyte area in square microns |
| `--batch_size N` | int | config default | Windows per GPU inference batch |

---

## Feature Flags

| Flag | Type | Default | Description |
|:-----|:-----|:--------|:------------|
| `--tissue_guidance` | flag | off | Enable tissue-guided processing. See [Features and Workflows](features.md#tissue-guidance). |
| `--save_tissue_window_grid` | flag | off | Save a thumbnail overlay of tissue-selected windows for debugging |
| `--tumor_segmentation` | flag | off | Enable tumor segmentation and distance analysis. See [Features and Workflows](features.md#tumor-analysis). |
| `--save_distance_map` | flag | off | Save the distance-colored visualization. Requires `--tumor_segmentation`. |
| `--save_qupath_annotation` | flag | config default | Force saving the QuPath GeoJSON annotation for the current run |
| `--skip_qupath_annotation` | flag | config default | Skip saving the QuPath GeoJSON annotation for the current run |
| `--extended_properties` | flag | off | Calculate additional morphological properties |
| `--profiling` | flag | off | Enable detailed timing and profiling output |

---

## Visualization Options

| Flag | Type | Default | Description |
|:-----|:-----|:--------|:------------|
| `--annotated_scale FACTOR` | float | `0.3` | Scale factor for the annotated thumbnail |
| `--save_mode MODE` | string | `balanced` | One of `fast`, `balanced`, `high_quality` |
| `--save_image_annotation` | flag | config default | Force saving the base annotated TIFF output for the current run |
| `--skip_image_annotation` | flag | config default | Skip saving the base annotated TIFF output for the current run |
| `--benchmark_saving` | flag | off | Benchmark and compare image-saving methods |
| `--show_adipocyte_ids` | flag | hidden by default | Show adipocyte identifiers on annotated images |
| `--hide_adipocyte_ids` | flag | default behavior | Hide adipocyte identifiers |
| `--show_grid` | flag | hidden by default | Show the analysis grid overlay |
| `--hide_grid` | flag | default behavior | Hide the analysis grid overlay |

`--show_adipocyte_ids` and `--hide_adipocyte_ids` are mutually exclusive. The same applies to `--show_grid` and `--hide_grid`.
`--save_image_annotation` and `--skip_image_annotation` are also mutually exclusive, and they override `config.SAVE_ANNOTATED_IMAGE` for the current run only.
`--save_qupath_annotation` and `--skip_qupath_annotation` are also mutually exclusive, and they override the QuPath GeoJSON export settings for the current run only.

---

## Debug Options

| Flag | Type | Default | Description |
|:-----|:-----|:--------|:------------|
| `--debug [MODE]` | optional string | `None` | Enable debug mode, optionally `processed` or `unprocessed` |
| `--benchmark_saving` | flag | off | Compare image-saving performance |

---

## GPU Options

| Flag | Type | Default | Description |
|:-----|:-----|:--------|:------------|
| `--gpu_id ID` | int | `0` | GPU device ID |
| `--disable_gpu_accel` | flag | off | Force CPU-only processing |
| `--disable_gpu_ops` | flag | off | Disable CuPy ops, GPU preprocessing, and GPU label mapping while keeping GPU inference |
| `--disable_gpu_preprocessing` | flag | off | Disable GPU Sobel/inversion preprocessing while keeping other GPU paths enabled |

---

## Batch Processing Options

| Flag | Type | Default | Description |
|:-----|:-----|:--------|:------------|
| `--resume_batch PATH` | string | `None` | Resume batch processing from a state file |
| `--resume_failed` | flag | off | Retry only the failed images during resume |
| `--list_resumable` | flag | off | List resumable batch jobs in the current directory |
| `--dry_run` | flag | off | Preview what would be processed without running inference |

See [CLI Workflows](cli-workflows.md#batch-workflows) for the recommended batch paths.

---

## ROI Selection Options

| Flag | Type | Default | Description |
|:-----|:-----|:--------|:------------|
| `--roi_freehand` | flag | off | Enable interactive freehand ROI selection |
| `--roi_polygon_file PATH` | string | `None` | Path to a saved ROI polygon JSON |
| `--roi_max_dim PIXELS` | int | `2048` | Maximum dimension for the ROI thumbnail |
| `--roi_min_coverage RATIO` | float | `0.2` | Minimum ROI coverage required for a window to be processed |

See [Features and Workflows](features.md#roi-workflow) for the canonical ROI guide.

---

## Memory Options

| Flag | Type | Default | Description |
|:-----|:-----|:--------|:------------|
| `--low_memory` | flag | off | Enable low-memory mode for constrained systems |
| `--memmap_mask` | flag | off | Use a disk-backed memory-mapped file for mask storage |

See [Performance Tuning](performance-tuning.md) for low-memory strategies.

---

## Information Options

| Flag | Type | Description |
|:-----|:-----|:------------|
| `--version` | flag | Print version and exit |
| `--help` / `-h` | flag | Show help and exit |

---

## Examples

```bash
# Basic run with tissue guidance
adifind slide.svs --tissue_guidance

# Full analysis with tumor detection
adifind slide.svs --tissue_guidance --tumor_segmentation --save_distance_map

# Skip QuPath GeoJSON export for this run
adifind slide.svs --tissue_guidance --skip_qupath_annotation

# Batch processing with custom output
adifind slides/ --tissue_guidance --output_dir results/

# Resume interrupted batch
adifind --resume_batch batch_state_abc123.json

# Low-memory mode for large slides
adifind large_slide.svs --tissue_guidance --low_memory --memmap_mask

# Custom window parameters
adifind slide.svs --window_size 2048 2048 --stride 1024 1024

# Specific GPU with extended properties
adifind slide.svs --tissue_guidance --gpu_id 1 --extended_properties

# CPU-only processing
adifind slide.svs --tissue_guidance --disable_gpu_accel

# Keep GPU inference but disable only GPU preprocessing
adifind slide.svs --tissue_guidance --disable_gpu_preprocessing

# Skip only the base annotated TIFF for this run
adifind slide.svs --tissue_guidance --skip_image_annotation

# Interactive ROI selection
adifind slide.svs --tissue_guidance --roi_freehand

# Debug mode with raw window exports
adifind slide.svs --tissue_guidance --debug unprocessed

# Dry run to preview a batch
adifind slides/ --tissue_guidance --dry_run
```

---

## See Also

- [CLI Workflows](cli-workflows.md) for guided workflows
- [Configuration](configuration.md) for defaults beyond CLI flags
- [Deployment](deployment.md) and [Configuration](configuration.md) for model paths, environment, and runtime configuration

---

Back to [Documentation Index](index.md)
