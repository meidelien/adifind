Back to [Documentation Index](index.md)

# CLI Workflows

Use this page for the canonical command-line workflow guide. It covers the most common single-slide and batch-processing paths without repeating the full flag reference or low-level schema details.

---

## Before You Begin

- Install AdiFind with [Installation Guide](INSTALL.md).
- For a first successful run, start with [Quick Start](getting-started.md).
- For every available flag, use [CLI Reference](cli-reference.md).

---

## Command Shape

Preferred installed entry point:

```bash
adifind <image_path> [options]
```

Equivalent source-tree command:

```bash
python code/main.py <image_path> [options]
```

`<image_path>` can be a single slide, a directory of slides, or a resume-state workflow depending on the options you use.

Supported inputs include common WSI formats, selected TIFF variants, and standard raster images. See [Supported File Formats](supported_file_formats.md) for the reader matrix and dispatch behavior.

---

## Single-Slide Workflows

### Basic adipocyte detection

```bash
adifind slide.svs
```

Use this when you want the simplest possible run with default settings.

### Tissue-guided processing (recommended)

```bash
adifind slide.svs --tissue_guidance
```

This is the default workflow most users should start with. Tissue guidance reduces work on empty glass regions and is typically the biggest performance win.

See [Features and Workflows](features.md#tissue-guidance) for how the guidance stage works.

### Tumor-aware analysis

```bash
adifind slide.svs --tissue_guidance --tumor_segmentation
```

This adds tumor segmentation and distance-to-tumor measurements for each adipocyte.

To save the distance-colored visualization as well:

```bash
adifind slide.svs --tissue_guidance --tumor_segmentation --save_distance_map
```

See [Features and Workflows](features.md#tumor-analysis) for the feature-level explanation and [Output Reference](output-reference.md) for the resulting files and columns.

### Extended morphology

```bash
adifind slide.svs --tissue_guidance --extended_properties
```

This adds extra morphological metrics such as eccentricity, solidity, extent, perimeter, and equivalent diameter.

### ROI-restricted processing

Interactive ROI selection:

```bash
adifind slide.svs --tissue_guidance --roi_freehand
```

Saved polygon input:

```bash
adifind slide.svs --tissue_guidance --roi_polygon_file path/to/roi.json
```

See [Features and Workflows](features.md#roi-workflow) for the canonical ROI guide.

---

## Batch Workflows

### Process a directory

```bash
adifind path/to/slides/ --tissue_guidance
```

AdiFind discovers supported inputs in the directory and processes them sequentially.

### Preview a run without processing

```bash
adifind path/to/slides/ --tissue_guidance --dry_run
```

Use this to validate discovery, configuration, and output planning before launching a large run.

### Resume an interrupted batch

```bash
adifind --resume_batch path/to/batch_state_abc123.json
```

### Retry only the failed images

```bash
adifind --resume_batch path/to/batch_state_abc123.json --resume_failed
```

### List resumable jobs

```bash
adifind --list_resumable
```

Batch state files are described in [Output Reference](output-reference.md).

### Reprocess previously completed slides

AdiFind normally skips inputs that already have output directories. To change that behavior, use [Configuration](configuration.md) rather than duplicating the workflow logic here.

---

## Workflow Chooser

| If you want to... | Go to |
|:------------------|:------|
| Learn every CLI flag | [CLI Reference](cli-reference.md) |
| Tune defaults, cache paths, or environment behavior | [Configuration](configuration.md) |
| Understand outputs and schemas | [Output Reference](output-reference.md) |
| Use GUI-assisted setup or ROI tools | [Features and Workflows](features.md) |
| Run from Python or Jupyter | [Python API](python-api.md) |
| Run in Docker or on HPC | [Deployment](deployment.md) |

---

## Next Steps

- [Quick Start](getting-started.md) for the shortest path to a first result
- [CLI Reference](cli-reference.md) for flag-by-flag details
- [Configuration](configuration.md) for environment variables and non-CLI defaults
- [Output Reference](output-reference.md) for files, CSV columns, and GeoJSON exports

---

Back to [Documentation Index](index.md)
