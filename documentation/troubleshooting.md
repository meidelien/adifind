Back to [Documentation Index](index.md)

# Troubleshooting

Solutions for common issues encountered when installing, configuring, or running AdiFind.

---

## Installation Issues

### OpenSlide Not Found

**Symptom:** `ImportError: OpenSlide library not found` or `DLL load failed`

**Windows:**
1. Download OpenSlide from [openslide.org](https://openslide.org/download/)
2. Extract to `C:\OpenSlide`
3. Verify `C:\OpenSlide\bin` contains `.dll` files
4. Restart your terminal / IDE

**Linux:**
```bash
sudo apt-get install openslide-tools libopenslide0
```

**macOS:**
```bash
brew install openslide
```

---

### Detectron2 Build Fails (Windows)

**Symptom:** Compilation errors when installing Detectron2 from source

**Solution:** Use prebuilt wheels instead of building from source:
- Visit [Detectron2 installation docs](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
- Download the wheel matching your PyTorch build and backend
- Install with `pip install <wheel_file>.whl`

---

### Missing `pkg_resources`

**Symptom:** `ModuleNotFoundError: No module named 'pkg_resources'`

**Cause:** Recent `setuptools` releases removed `pkg_resources`, while current Detectron2 and some legacy dependencies may still expect it. This is a dependency compatibility issue, not a model-weights or slide-format problem.

**Fix for existing installs:**

```bash
python -m pip install "setuptools<82"
```

Then rerun the validation command or `--dry_run`.

---

### PyTorch CUDA Version Mismatch

**Symptom:** `RuntimeError: CUDA error: no kernel image is available for execution`

**Solution:** Ensure PyTorch CUDA version matches your installed CUDA toolkit:

```bash
# Check CUDA version
nvidia-smi  # Shows driver CUDA version

# Reinstall PyTorch with correct CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### Conda Environment Creation Fails

**Symptom:** `ResolvePackageNotFound` or dependency conflicts

**Solution:**
```bash
# Update conda first
conda update -n base conda

# Try creating with --force
conda env create -f environment.yml --force

# Or create manually
conda create -n adifind python=3.10
conda activate adifind
# Then follow manual pip installation steps
```

---

## Runtime Issues

### CUDA Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions (in order of impact):**

1. **Reduce batch size:**
   ```bash
   adifind slide.svs --tissue_guidance --batch_size 1
   ```

2. **Enable low-memory mode:**
   ```bash
   adifind slide.svs --tissue_guidance --low_memory --memmap_mask
   ```

3. **Reduce annotated image scale:**
   ```bash
   adifind slide.svs --tissue_guidance --annotated_scale 0.1
   ```

4. **Disable CuPy ops, GPU preprocessing, and GPU label mapping:**
   ```bash
   adifind slide.svs --tissue_guidance --disable_gpu_ops
   ```

5. **Switch to CPU-only:**
   ```bash
   adifind slide.svs --tissue_guidance --disable_gpu_accel
   ```

---

### Slow Processing

**Symptom:** Processing is unexpectedly slow for a single slide

**Solutions:**

1. **Enable tissue guidance** first:
   ```bash
   adifind slide.svs --tissue_guidance
   ```

2. **Verify that PyTorch sees a GPU-capable backend:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))
   ```

3. **Install CuPy** for optional GPU-backed distance transforms and array ops:
   ```bash
   pip install cupy-cuda12x  # CUDA wheel example
   ```

4. **Copy slides to fast storage** (NVMe SSD vs. NAS)

5. **Increase batch size** (if VRAM allows):
   ```bash
   adifind slide.svs --tissue_guidance --batch_size 8
   ```

See [Performance Tuning](performance-tuning.md) for comprehensive optimization.

---

### Model Download Fails

**Symptom:** `ConnectionError`, `403`, or download hangs during first run

**Solutions:**

1. **Check internet connection** on the machine running AdiFind

2. **Private model repo:** the current Hugging Face model repo is private. Use `HF_TOKEN`, `huggingface-cli login`, or local canonical model directories for no-token runs.

3. **Manual download:** Download models directly from HuggingFace and set environment variables:
   ```bash
   export ADIFIND_ADIPOCYTE_MODEL_DIR="/path/to/downloaded/model/"
   export ADIFIND_TUMOR_MODEL_DIR="/path/to/downloaded/tumor/"
   export ADIFIND_TISSUE_MODEL_DIR="/path/to/downloaded/tissue/"
   ```

4. **Set custom cache directory:**
   ```bash
   export ADIFIND_CACHE_DIR="/path/with/write/access/"
   ```

5. **Proxy settings:** If behind a corporate proxy:
   ```bash
   export HTTPS_PROXY="http://proxy:port"
   export HTTP_PROXY="http://proxy:port"
   ```

---

### "No valid image files found"

**Symptom:** `FileNotFoundError: No valid image files found in directory`

**Check:**
- Batch discovery recognizes these extensions: `.svs`, `.ndpi`, `.tiff`, `.tif`, `.vms`, `.vmu`, `.scn`, `.mrxs`, `.svslide`, `.bif`, `.czi`, `.png`, `.jpg`, `.jpeg`
- Files with other extensions (e.g., `.ome.tif`) are matched by the `.tif` suffix — they work in batch mode
- Files must be in the directory root (not in subdirectories)
- File permissions must allow reading
- For OME-TIFF or CZI support, install slideio: `pip install slideio`

See [Supported File Formats](supported_file_formats.md) for the full format dispatch.

---

### Black Windows Skipped

**Symptom:** Log shows many "skipping predominantly black window" messages

**This is normal behavior.** AdiFind skips windows that are mostly empty (black) to save processing time. This frequently occurs in slides with tissue surrounded by empty glass when tissue guidance is not enabled.

**Solution:** Enable `--tissue_guidance` to avoid scanning empty regions entirely.

---

### Large Slide Memory Issues

**Symptom:** Python process killed or system becomes unresponsive

**Solutions:**

1. **Low-memory + memmap mode:**
   ```bash
   adifind slide.svs --tissue_guidance --low_memory --memmap_mask
   ```

2. **Reduce annotated image size:**
   ```bash
   adifind slide.svs --tissue_guidance --annotated_scale 0.1
   ```

3. **Disable full mask saving:**
   ```python
   config.SAVE_FULL_MASK = False  # Default is already False
   ```

4. **Disable postprocessed/unprocessed image stitching:**
   ```python
   config.SAVE_POSTPROCESSED_IMAGE = False
   config.SAVE_UNPROCESSED_IMAGE = False
   ```

---

## Docker Issues

### "Adipocyte model not found"

**Solution:** Add model weights to `docker/models/adipocyte/` before building, or mount a volume:
```bash
docker run -v /path/to/models:/app/models:ro ...
```

### Container OpenSlide Packaging Drift

**Symptom:** Docker or Apptainer builds fail after mixing Conda, distro, and pip OpenSlide packages, or CPU and GPU containers behave differently around slide loading.

**Resolution:** the supported container path is `openslide-python` plus pip-installed `openslide-bin`. The Docker CPU image, Docker GPU image, and Apptainer definition intentionally avoid distro and Conda OpenSlide packages inside the container runtime to reduce ABI drift and keep CI behavior consistent.

### GPU Not Detected in Container

**Check:**
1. `nvidia-smi` works on the host
2. NVIDIA Container Toolkit is installed for the provided CUDA/NVIDIA container path: `dpkg -l | grep nvidia-container`
3. Docker runtime is configured: check `/etc/docker/daemon.json`
4. Use `--gpus all` flag: `docker run --gpus all ...`

### Permission Denied on Output

```bash
# Fix permissions on host
chmod 777 /path/to/output

# Or run container with matching user
docker run --user $(id -u):$(id -g) ...
```

---

## HPC Issues

### GPU Not Available in SLURM Job

**Check:**
1. `--gres=gpu:1` in SLURM script
2. `--nv` flag passed to `apptainer run`
3. GPU partition is correctly specified: `--partition=gpu`

### Model Download on Compute Nodes

Compute nodes usually lack internet access. Pre-cache models:
```bash
# On login node (internet access)
export ADIFIND_CACHE_DIR=/shared/models/adifind
adifind --help  # Downloads models

# On compute node
export ADIFIND_CACHE_DIR=/shared/models/adifind
```

See [Deployment](deployment.md#hpc-deployment) for detailed instructions.

---

## Getting Help

If your issue is not listed here:

1. Run with `--debug` to get detailed logging output
2. Run with `--profiling` to identify bottlenecks
3. Check the [GitHub Issues](https://github.com/meidelien/adifind/issues) page
4. Run the smoke tests: `pytest code/test_smoke.py -v`

---

## See Also

- [Installation Guide](INSTALL.md) — Detailed setup instructions
- [Performance Tuning](performance-tuning.md) — Optimization strategies
- [Deployment](deployment.md#docker-deployment) — Container troubleshooting
- [Deployment](deployment.md#hpc-deployment) — Cluster troubleshooting

---

Back to [Documentation Index](index.md)
