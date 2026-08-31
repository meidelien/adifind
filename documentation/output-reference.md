Back to [Documentation Index](index.md)

# Output Reference

Complete reference for all files and data formats produced by AdiFind.

---

## Directory Structure

Each run creates a timestamped output directory:

```
adifind_results_{slide_name}_{YYYYMMDD_HHMMSS}/
├── {name}_adipocyte_results.csv              # Per-adipocyte measurements
├── {name}_annotated_thumbnail.tiff           # Visual overlay of all detections
├── {name}_qupath_annotations.geojson         # QuPath-compatible annotations (if QuPath GeoJSON export is enabled)
├── {name}_summary_stats.json                 # Aggregate statistics
├── {name}_tumor_zone_overlay.tiff            # Tumor zone visualization (if tumor enabled)
├── {name}_distance_colored.tiff              # Distance-colored image (if --save_distance_map)
├── {name}_tissue_detection.json              # Tissue guidance results (if tissue enabled)
├── {name}_tissue_window_grid.tiff            # Debug grid overlay (if --save_tissue_window_grid)
└── {name}_region_statistics.json             # Per-region tissue stats (if tissue enabled)
```

> ℹ️ **Note:** Not all files are produced in every run. Optional files depend on enabled features (`--tissue_guidance`, `--tumor_segmentation`, `--save_distance_map`, QuPath GeoJSON export settings, etc.).

---

## CSV Output — Per-Adipocyte Results

**File:** `{name}_adipocyte_results.csv`

This is the primary data output. Each row represents one detected adipocyte.

### Standard Columns

| Column | Type | Unit | Description |
|:-------|:-----|:-----|:------------|
| `Adipocyte_ID` | int | — | Unique identifier for the adipocyte |
| `Area_Microns_Squared` | float | µm² | Cross-sectional area |
| `Centroid_X` | float | pixels | X coordinate of cell center (full-resolution) |
| `Centroid_Y` | float | pixels | Y coordinate of cell center (full-resolution) |

### Tumor Distance Columns

Present when `--tumor_segmentation` is enabled:

| Column | Type | Unit | Description |
|:-------|:-----|:-----|:------------|
| `Distance_To_Closest_Tumour` | float | µm | Euclidean distance from centroid to nearest tumor boundary |
| `Distance_Bin` | string | — | Categorical bin: `Close` (≤100 µm), `Medium` (100–500 µm), or `Far` (>500 µm) |

### Extended Property Columns

Present when `--extended_properties` is enabled:

| Column | Type | Unit | Description |
|:-------|:-----|:-----|:------------|
| `Eccentricity` | float | — | Shape eccentricity (0 = circle, 1 = line) |
| `Solidity` | float | — | Area / convex hull area (1 = fully convex) |
| `Extent` | float | — | Area / bounding box area |
| `Perimeter` | float | pixels | Perimeter length |
| `Equivalent_Diameter` | float | pixels | Diameter of a circle with the same area |

---

## Summary Statistics — JSON

**File:** `{name}_summary_stats.json`

```json
{
  "image_name": "K106942",
  "image_path": "/path/to/K106942.svs",
  "total_adipocytes": 15234,
  "total_windows_processed": 428,
  "mpp": 0.2528,
  "window_size": [2000, 2000],
  "stride": [1700, 1700],
  "processing_time_seconds": 360.5,
  "median_area_microns_squared": 1850.3,
  "mean_area_microns_squared": 2140.7,
  "std_area_microns_squared": 1420.1,
  "min_area_microns_squared": 250.0,
  "max_area_microns_squared": 24890.5,
  "num_tumors_detected": 3,
  "confidence_threshold": 0.80,
  "tissue_guidance_enabled": true,
  "tumor_segmentation_enabled": true
}
```

---

## Annotated Thumbnail — TIFF

**File:** `{name}_annotated_thumbnail.tiff`

A downscaled version of the slide with all detected adipocytes outlined. The scale is controlled by `--annotated_scale` (default: 0.3 = 30% of full resolution).
This file is omitted when annotated-image saving is disabled via `config.SAVE_ANNOTATED_IMAGE = False` or `--skip_image_annotation`.

### Save Modes

| Mode | Format | Compression | Use Case |
|:-----|:-------|:------------|:---------|
| `fast` | JPEG in TIFF | JPEG (~85% quality) | Fastest save, smallest files |
| `balanced` | TIFF | LZW compressed | Good balance of speed and quality |
| `high_quality` | TIFF | Optimized deflate | Best visual quality, larger files |

Set via `--save_mode` on the command line.

---

## QuPath Annotations — GeoJSON

**File:** `{name}_qupath_annotations.geojson`

Standard GeoJSON `FeatureCollection` compatible with QuPath's import function. See [QuPath Integration](qupath-integration.md) for import instructions.

Each feature contains:
- **Polygon geometry** — Adipocyte boundary coordinates
- **Classification** — `"Adipocyte"` (configurable)
- **Measurements** — Area in µm² (when enabled)

---

## Tumor Zone Overlay — TIFF

**File:** `{name}_tumor_zone_overlay.tiff`

Generated when `--tumor_segmentation` is enabled and `SAVE_TUMOR_ZONE_OVERLAY_IMAGE = True`.

Shows the slide with color-coded proximity zones:

| Zone | Color | Distance |
|:-----|:------|:---------|
| Near | Yellow | ≤ 1,500 µm |
| Intermediate | Green | 1,500–5,000 µm |
| Distal | Blue | > 5,000 µm |

---

## Distance-Colored Visualization — TIFF

**File:** `{name}_distance_colored.tiff`

Generated when `--save_distance_map` is passed (requires `--tumor_segmentation`).

Each adipocyte is color-coded by its distance to the nearest tumor, using a continuous color gradient. Closer adipocytes are warmer (red/yellow), distant ones are cooler (blue).

---

## Tissue Detection Results — JSON

**File:** `{name}_tissue_detection.json`

Generated when `--tissue_guidance` is enabled. Contains the detected tissue bounding boxes and their confidence scores. Used for caching.

---

## Region Statistics — JSON

**File:** `{name}_region_statistics.json`

Generated when tissue guidance is enabled and `SAVE_REGION_STATISTICS = True`. Contains per-tissue-region statistics including area coverage and window counts.

---

## Batch Outputs

When processing multiple images, additional batch-level files are created in the base output directory:

| File | Description |
|:-----|:------------|
| `batch_summary.json` | Machine-readable summary with per-image results |
| `batch_statistics.txt` | Human-readable statistics table |
| `processing_log.txt` | Detailed processing log with timing |
| `batch_state_{id}.json` | State file for resume/retry (see [CLI Workflows](cli-workflows.md#batch-workflows)) |

---

## See Also

- [CLI Workflows](cli-workflows.md) — How to generate these outputs
- [Features and Workflows](features.md#tumor-analysis) — Distance columns and zone visualization
- [QuPath Integration](qupath-integration.md) — Importing GeoJSON into QuPath
- [CLI Reference](cli-reference.md) — Flags that control output

---

Back to [Documentation Index](index.md)
