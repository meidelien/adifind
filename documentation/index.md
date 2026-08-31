<p align="center">
  <img src="../media/adifind_logo_cropped.png" alt="AdiFind Logo" width="400">
</p>

<h1 align="center">AdiFind Documentation</h1>

<p align="center">
  <strong>Automated adipocyte detection and analysis in whole-slide images</strong>
</p>

This page is the full documentation portal for AdiFind. Start here if you want the complete map; use the README for the shortest front-door path.

---

## Start Here

| Step | Read this | Why it exists |
|:-----|:----------|:--------------|
| 1 | [Installation Guide](INSTALL.md) | Canonical setup for Conda, Docker, Apptainer, and manual installation |
| 2 | [Quick Start](getting-started.md) | First successful run and expected outputs |
| 3 | [CLI Workflows](cli-workflows.md) | Canonical single-slide and batch command-line workflows |

---

## Common Tasks

| I want to... | Go to |
|:-------------|:------|
| Process slides from the command line | [CLI Workflows](cli-workflows.md) |
| Inspect output files and CSV / GeoJSON schemas | [Output Reference](output-reference.md) |
| Review outputs in QuPath | [QuPath Integration](qupath-integration.md) |
| Understand the product workflow, tissue guidance, tumor analysis, GUI, and ROI features | [Features and Workflows](features.md) |
| Work from Python or Jupyter | [Python API](python-api.md) and [notebooks/quickstart.ipynb](../notebooks/quickstart.ipynb) |
| Configure defaults, model paths, cache locations, and environment variables | [Configuration](configuration.md) |
| Deploy with Docker or on HPC | [Deployment](deployment.md) |

---

## Reference

| Document | Description |
|:---------|:------------|
| [CLI Reference](cli-reference.md) | Complete command-line flag reference with types and defaults |
| [Configuration](configuration.md) | Runtime variables, model paths, and `Config` options |
| [Output Reference](output-reference.md) | Output directory structure, CSV columns, and GeoJSON schema |
| [Supported File Formats](supported_file_formats.md) | Reader behavior and supported input/output formats |
| [Python API](python-api.md) | Programmatic usage from scripts and notebooks |
| [QuPath Integration](qupath-integration.md) | Importing AdiFind annotations into QuPath |

---

## Advanced and Development

| Document | Description |
|:---------|:------------|
| [Performance Tuning](performance-tuning.md) | GPU inference, CuPy ops, preprocessing, async I/O, memory management, and optimization strategy |
| [Architecture](architecture.md) | Module map, data flow, algorithms, and design patterns |
| [Troubleshooting](troubleshooting.md) | Common setup, runtime, and deployment issues |
| [Contributing](contributing.md) | Development setup, code style, and project workflow |

---

## Next Steps

- New to the project: start with [Installation Guide](INSTALL.md) and [Quick Start](getting-started.md).
- Running analyses regularly: keep [CLI Workflows](cli-workflows.md), [Configuration](configuration.md), and [Output Reference](output-reference.md) close at hand.
- Working on the codebase: use [Architecture](architecture.md) and [Contributing](contributing.md).
