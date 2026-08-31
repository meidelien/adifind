Back to [Documentation Index](index.md)

# Quick Start

Use this page once AdiFind is installed and your environment is activated. If you still need setup instructions, start with [Installation Guide](INSTALL.md).

---

## Before You Begin

- Repository cloned locally: `git clone https://github.com/meidelien/adifind.git`
- Environment activated: `conda activate adifind`
- Example data available in `example_data/`

---

## 1. Run Your First Analysis

Recommended CLI entry point:

```bash
adifind example_data/K106942.svs --tissue_guidance
```

Equivalent source-tree command:

```bash
python code/main.py example_data/K106942.svs --tissue_guidance
```

`--tissue_guidance` skips empty glass regions and is the recommended default for most runs.

---

## 2. Check the Output

AdiFind writes a timestamped results directory:

```text
adifind_results_K106942_YYYYMMDD_HHMMSS/
|- K106942_adipocyte_results.csv
|- K106942_annotated_thumbnail.tiff
|- K106942_qupath_annotations.geojson
`- K106942_summary_stats.json
```

Open the annotated TIFF for a quick visual check, inspect the CSV for per-cell measurements, and import the GeoJSON into QuPath if you want interactive review.

See [Output Reference](output-reference.md) for the full file list and schema details.

---

## 3. Add Tumor Analysis

To add tumor segmentation and distance-aware exports:

```bash
adifind example_data/K106942.svs --tissue_guidance --tumor_segmentation --save_distance_map
```

This adds per-adipocyte distance-to-tumor measurements and additional tumor-context visualizations.

See [Features and Workflows](features.md#tumor-analysis) for the full workflow.

---

## 4. Continue From Here

| Want to do more? | Read |
|:-----------------|:-----|
| Understand common single-slide workflows | [CLI Workflows](cli-workflows.md) |
| Review every CLI flag | [CLI Reference](cli-reference.md) |
| Process many slides at once | [CLI Workflows](cli-workflows.md#batch-workflows) |
| Use the desktop app | [Features and Workflows](features.md#desktop-gui) |
| Work from Python or Jupyter | [Python API](python-api.md) |
| Inspect outputs in QuPath | [QuPath Integration](qupath-integration.md) |

---

Back to [Documentation Index](index.md)
