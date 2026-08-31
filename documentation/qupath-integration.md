Back to [Documentation Index](index.md)

# QuPath Integration

AdiFind exports detection results as GeoJSON annotations compatible with [QuPath](https://qupath.github.io/), enabling visual overlay of detected adipocytes on the original whole-slide image in QuPath's interactive viewer.

---

## What Gets Exported

Each detected adipocyte is exported as a polygon annotation in GeoJSON format with:

- geometry: the adipocyte boundary polygon
- classification: `"Adipocyte"` by default
- measurements: area in square microns when measurement export is enabled

---

## Importing into QuPath

1. Open the same WSI in QuPath.
2. Choose **File -> Import Objects**.
3. Select the `*_qupath_annotations.geojson` file from the AdiFind output directory.
4. Review the imported polygon overlays on the slide.

After import, you can continue with QuPath classification, measurement review, and overlay inspection alongside other pathology annotations.

---

## Output File

The GeoJSON file is written to the output directory:

```text
adifind_results_{name}_{timestamp}/
`- {name}_qupath_annotations.geojson
```

### GeoJSON structure

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[x1, y1], [x2, y2], [x3, y3]]]
      },
      "properties": {
        "classification": {"name": "Adipocyte"},
        "measurements": {"Area_Microns_Squared": 1850.3}
      }
    }
  ]
}
```

---

## Configuration

| Option | Default | Description |
|:-------|:--------|:------------|
| `ENABLE_QUPATH_EXPORT` | `True` | Enable QuPath GeoJSON export |
| `SAVE_QUPATH_GEOJSON` | `True` | Save GeoJSON output |
| `SAVE_QUPATH_SCRIPT` | `True` | Reserved compatibility setting; Groovy script export is not implemented in the current runtime path |
| `QUPATH_ANNOTATION_CLASS` | `"Adipocyte"` | Classification name for exported annotations |
| `INCLUDE_MEASUREMENTS_IN_QUPATH` | `True` | Include measurement properties in the GeoJSON |

Disable QuPath export if you do not need it:

```python
config.ENABLE_QUPATH_EXPORT = False
```

Or disable the outputs individually:

```python
config.SAVE_QUPATH_GEOJSON = False
```

From the CLI, use `--save_qupath_annotation` or `--skip_qupath_annotation` to override the QuPath GeoJSON export setting for a single run.

---

## See Also

- [Output Reference](output-reference.md) for all output file formats
- [Configuration](configuration.md) for QuPath export options
- [CLI Workflows](cli-workflows.md) for processing workflows

---

Back to [Documentation Index](index.md)
