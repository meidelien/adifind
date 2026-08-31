# Supported File Formats

This page lists the image and whole-slide file formats that AdiFind can read and write, and how support is provided.

## Overview

AdiFind is designed primarily for gigapixel whole-slide images (WSI) but also accepts standard image files for smaller inputs and testing. The project uses `openslide` as the primary WSI reader, `slideio` (optional) for additional WSI/OME-TIFF support, and `Pillow`/`tifffile` for regular images and output writing.

## Input formats (read)

- OpenSlide-compatible WSI formats (via `openslide`):
  - AdiFind uses OpenSlide as the primary WSI reader when available. See the OpenSlide documentation for complete details: https://openslide.org/formats/

  OpenSlide understands a wide range of vendor slide formats. Common vendor backends and file extensions include:

  - **Aperio** — single-file pyramidal tiled TIFF
    - Extensions: `.svs`, `.tif`
  - **Hamamatsu** — multi-file or single-file NDPI-like formats
    - Extensions: `.vms`, `.vmu`, `.ndpi`
  - **Leica** — single-file BigTIFF-style slides
    - Extensions: `.scn`
  - **MIRAX** — multi-file proprietary format
    - Extensions: `.mrxs`
  - **Philips** — single-file pyramidal TIFF/BigTIFF
    - Extensions: `.tiff`, `.tif`
  - **Ventana** — single-file pyramidal BigTIFF
    - Extensions: `.bif`, `.tif`
  - **Zeiss** — CZI container (OpenSlide backend parses common variants)
    - Extensions: `.czi`
  - **Sakura** — SQLite-based tile store
    - Extensions: `.svslide`
  - **Trestle** — single-file pyramidal TIFF with overlap metadata
    - Extensions: `.tif`
  - **MIRAX / Generic tiled TIFF / DICOM** — other backends (e.g., `.mrxs`, `.tif`, `.dcm`)


- OME-TIFF / GDAL-backed TIFFs (via `slideio`, optional):
  - AdiFind will use `slideio` (when installed) to open OME-TIFFs and other formats accessible through the GDAL driver.
  - These are handled in `ImageHandler` as `is_ome_tiff` and read using `slideio.open_slide(..., "GDAL")`.

- Standard image files (via `Pillow` / `PIL`):
  - JPEG, PNG, single-page TIFF (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`) are read via Pillow as fallback for non-WSI images.

## Output formats (write)

- Per-cell CSV measurements: `pandas` CSV export (`.csv`).
- Annotated thumbnails and stitched outputs: BigTIFF via `tifffile` (supports `bigtiff=True` for very large images).
- QuPath GeoJSON annotations: GeoJSON exported as `.geojson` for QuPath import.

## How AdiFind decides which reader to use

- The `ImageHandler` class (in `code/image_processing.py`) first attempts to open a file with `OpenSlide`. If that succeeds the file is treated as a digital slide (WSI).
- If OpenSlide fails and the filename looks like an OME-TIFF/TIFF, AdiFind attempts to open it with `slideio` (GDAL driver) when `slideio` is installed.
- Otherwise a regular image reader via `Pillow` is used.

This logic is encapsulated in `is_digital_slide()` and the `ImageHandler` constructor.

## Supported extensions (summary)

- WSI (OpenSlide): `.svs`, `.ndpi`, `.mrxs`, `.scn`, other OpenSlide-recognized TIFF-based slides
- OME/GDAL TIFFs: `.ome.tif`, `.ome.tiff`, `.tif`, `.tiff` (when `slideio` is available)
- Regular images: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

Note: file extension alone is not authoritative — the reader probes the file and uses OpenSlide where possible.

## Optional dependencies and environment notes

- `openslide` / `openslide-python`: required for primary WSI support. On Windows you may need to set `OPENSLIDE_PATH` to the folder containing the OpenSlide binary/DLLs (the code looks for `C:\OpenSlide\bin` by default).
- `slideio`: optional. Install to enable better OME-TIFF and additional WSI format support.
- `tifffile`: used for writing large stitched TIFF outputs.

## Troubleshooting

- "OpenSlideError" when opening a slide: ensure OpenSlide is installed and the path is available to Python. On Windows, either install OpenSlide via conda or set `OPENSLIDE_PATH` to the OpenSlide `bin` directory.
- "SlideIO not available": OME-TIFFs requiring GDAL will fail unless `slideio` (and the underlying GDAL drivers) are installed. See `INSTALL.md` for recommended optional packages.
- If in doubt, run the simple check in Python:

```python
from code.image_processing import is_digital_slide
print(is_digital_slide('path/to/slide.svs'))
```

This returns `True` for files `OpenSlide` recognizes.

## Want to add support for another format?

If you need additional reader support, the project can be extended by adding a reader branch in `ImageHandler` (see `code/image_processing.py`). For example, adding a custom GDAL/openslide backend or an external SDK-based reader is straightforward — contact the maintainers or open an issue with the desired format and sample files.

---

For general environment and installation instructions that affect file format support, see [Installation Guide](INSTALL.md) and [Quick Start](getting-started.md).
