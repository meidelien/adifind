Back to [Documentation Index](index.md)

# Features and Workflows

Use this page as the canonical guide to AdiFind's feature-level workflows. It explains what each major capability does, when to enable it, and how the pieces fit together in a real analysis run.

---

## Processing Flow

AdiFind is built around a multi-resolution workflow. It uses low-resolution overview passes to understand where tissue and tumor context are located, then spends full-resolution inference only where that context matters.

### Slide ingestion and resolution handling

`ImageHandler` selects the best available reader for the input:

- OpenSlide for common WSI formats
- `slideio` with GDAL for selected TIFF variants and CZI support when available
- Pillow for standard raster images

For pyramidal slides, AdiFind reads from the most appropriate level instead of always forcing full-resolution I/O during overview stages.

### MPP-aware analysis

AdiFind reads microns-per-pixel metadata when available and falls back to configured defaults only when metadata is missing or invalid. That physical scale is then used to adapt window sizing and convert measurements into biologically meaningful units.

<p align="center">
  <img src="../media/Asset%2029@4x.png" alt="Annotated whole-slide AdiFind output with tumor context, adipose tissue labels, and distance-to-tumor overlay." width="960"/>
  <br/>
  <em>AdiFind links whole-slide context, tumor proximity, and full-resolution adipocyte measurement in one workflow.</em>
</p>

### Full-resolution adipocyte inference

Once the relevant regions are known, AdiFind runs overlapping sliding windows over the selected full-resolution areas. Adipocyte detection uses Detectron2-based instance segmentation, and overlapping detections are merged afterward so each final adipocyte is represented once rather than as multiple window fragments.

### Outputs

A standard run can produce merged label masks, annotated review images, CSV measurements, summary JSON, and QuPath-ready annotation exports. Tumor-aware runs add distance measurements and tumor-context visualizations.

See [Output Reference](output-reference.md) for file-level details.

---

## Tissue Guidance

Tissue-guided processing is often the first workflow option to enable for whole-slide analysis. It detects tissue regions on a low-resolution thumbnail and restricts full-resolution adipocyte inference to the windows that overlap real tissue instead of empty glass.

Enable it with:

```bash
adifind slide.svs --tissue_guidance
```

Use the debug overlay when you need to inspect the selected windows:

```bash
adifind slide.svs --tissue_guidance --save_tissue_window_grid
```

The effect depends on how much of the slide is tissue versus empty background:

| Tissue coverage | Effect |
|:----------------|:-------|
| 5-10% | Often skips a large amount of empty background before full-resolution inference starts. |
| 10-25% | Usually removes many windows that would otherwise be scanned. |
| 25-50% | Still reduces background work, but more of the slide already contains tissue. |
| > 50% | The effect is smaller because much of the slide still needs analysis. |

If the tissue-guidance model is unavailable, AdiFind falls back to full-slide scanning with a warning rather than failing the run.

Configuration details live in [Configuration](configuration.md). Workflow examples live in [CLI Workflows](cli-workflows.md).

---

## Tumor Analysis

Tumor analysis adds a second overview-stage model that segments tumor regions and computes the Euclidean distance from each adipocyte to the nearest tumor boundary.

Enable tumor analysis:

```bash
adifind slide.svs --tissue_guidance --tumor_segmentation
```

Save the distance-colored visualization:

```bash
adifind slide.svs --tissue_guidance --tumor_segmentation --save_distance_map
```

The resulting CSV adds tumor-distance columns:

| Column | Description |
|:-------|:------------|
| `Distance_To_Closest_Tumour` | Distance in microns from the adipocyte centroid to the nearest tumor boundary |
| `Distance_Bin` | Categorical tumor-distance bin |

When no tumor is found, adipocyte detection still proceeds. The tumor-distance fields are left empty or `NaN`, and tumor-specific visualizations are skipped.

Use [Output Reference](output-reference.md) for the exact file and schema behavior, and [Configuration](configuration.md) for the tumor-related thresholds and output controls.

---

## Desktop GUI

AdiFind includes a desktop application for users who want a guided setup experience without leaving the same underlying CLI pipeline. The current GUI is implemented with **PySide6** and launches the analysis in a subprocess so GUI runs stay aligned with the reproducible command-line workflow.

Launch it with:

```bash
conda activate adifind
python code/adifind_desktop.py
```

The desktop app provides:

- file and directory selection
- run configuration for core analysis options
- tissue-guidance, tumor-analysis, and ROI controls
- GPU and output configuration
- live progress and log streaming
- a preview of the equivalent CLI command

The desktop GUI requires a graphical display. For headless environments such as Docker without display passthrough or HPC compute nodes, use the CLI instead.

---

## ROI Workflow

ROI selection lets you restrict analysis to the part of the slide you actually want to measure. This is useful for targeted review, reproducible comparisons, and excluding irrelevant tissue or background.

Interactive ROI drawing:

```bash
adifind slide.svs --tissue_guidance --roi_freehand
```

Saved polygon input:

```bash
adifind slide.svs --tissue_guidance --roi_polygon_file path/to/roi.json
```

Expected JSON structure:

```json
{
  "polygon": [[100, 200], [300, 200], [300, 500], [100, 500]],
  "thumbnail_size": [2048, 1536]
}
```

ROI restriction can be combined with tissue guidance and tumor analysis. In those cases, windows must satisfy both the ROI filter and the relevant feature filters.

Interactive ROI selection requires a display. For headless environments, use `--roi_polygon_file`.

---

## Interoperability

AdiFind outputs are designed to fit into existing pathology workflows rather than replace them. In particular, the pipeline exports adipocytes as QuPath-compatible GeoJSON polygons with per-object metadata such as adipocyte ID, centroid coordinates, and area.

<p align="center">
  <img src="../media/adifind_stardist_example.png" alt="QuPath view showing AdiFind adipocyte masks together with StarDist nuclei detections." width="960"/>
  <br/>
  <em>AdiFind integrates cleanly with existing QuPath workflows, including side-by-side review with StarDist nuclei segmentation.</em>
</p>

For the exact import workflow, use [QuPath Integration](qupath-integration.md).

---

## Next Steps

- [CLI Workflows](cli-workflows.md) for common command-line runs
- [CLI Reference](cli-reference.md) for every available flag
- [Configuration](configuration.md) for thresholds, cache paths, and environment variables
- [Output Reference](output-reference.md) for files, columns, and schemas
- [QuPath Integration](qupath-integration.md) for downstream review in QuPath

---

Back to [Documentation Index](index.md)
