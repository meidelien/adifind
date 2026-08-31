Back to [Documentation Index](index.md)

# Performance Tuning

Guide to the main performance-related controls in AdiFind, from GPU usage to memory management for large slides.

---

## Optimization Overview

These are the main levers that change how much work AdiFind does or where that work runs.

| Optimization | Effect | Effort |
|:-------------|:-------|:-------|
| **Tissue guidance** | Skips many empty-background windows before full-resolution inference starts. | `--tissue_guidance` |
| **GPU inference** | Runs the core model on the GPU instead of the CPU. | Enabled by default on supported systems |
| **Batch inference** | Processes more windows in each inference call when VRAM allows it. | Tune `BATCH_INFERENCE_SIZE` |
| **Async I/O** | Reads upcoming windows while the current batch is still being processed. | Enabled by default |
| **GPU preprocessing** | Moves Sobel and inversion preprocessing off the CPU when those filters are enabled. | Enabled by default |
| **GPU label mapping** | Moves large-mask relabeling work off the CPU. | Advanced tuning |
| **Tensor core acceleration** | Uses lower-precision inference on supported accelerator hardware when the stack supports it. | Automatic on supported hardware |

> Tip: Start with `--tissue_guidance` because it reduces unnecessary work on empty slide regions.

---

## GPU Paths

### Overview

AdiFind can use the GPU at several stages:

1. Detectron2 inference for the main model
2. GPU preprocessing for inversion and Sobel filtering
3. CuPy operations for distance transforms and other array-heavy work
4. GPU label mapping for large masks
5. CUDA streams for overlapping GPU tasks

### CuPy

CuPy is optional. It is useful for the tumor distance transform and related array operations, but GPU inference can still work without it when your PyTorch and Detectron2 stack has a supported GPU backend.

CUDA wheel examples:

```bash
pip install cupy-cuda12x
pip install cupy-cuda11x
```

Without CuPy, these operations fall back to CPU implementations.

### GPU Configuration

| Option | Default | Description |
|:-------|:--------|:------------|
| `USE_GPU_INFERENCE` | `True` | Run Detectron2 inference on the GPU |
| `USE_CUPY` | `True` | Enable CuPy-backed distance transforms and array ops |
| `USE_GPU_PREPROCESSING` | `True` | Enable GPU Sobel/inversion preprocessing |
| `ENABLE_GPU_LABEL_MAPPING` | `False` | Enable GPU label remapping for large masks |
| `GPU_MEMORY_LIMIT_GB` | `30` | GPU memory limit with headroom left free |
| `ENABLE_GPU_STREAMS` | `True` | Use CUDA streams for parallel GPU work |
| `GPU_STREAM_COUNT` | `20` | Number of concurrent streams |
| `GPU_CLEANUP_INTERVAL` | `50` | Flush GPU memory every N windows |
| `GPU_CLEANUP_AGGRESSIVE` | `False` | Force more aggressive cleanup |

### Disable GPU Paths

```bash
# Full CPU mode
adifind slide.svs --tissue_guidance --disable_gpu_accel

# Keep GPU inference, disable CuPy, GPU preprocessing, and GPU label mapping
adifind slide.svs --tissue_guidance --disable_gpu_ops

# Keep inference and CuPy, disable only GPU preprocessing
adifind slide.svs --tissue_guidance --disable_gpu_preprocessing
```

---

## Batch Inference Tuning

`BATCH_INFERENCE_SIZE` controls how many windows are grouped into each GPU inference call.

| GPU VRAM | Recommended batch size |
|:---------|:-----------------------|
| 8 GB | 1-2 |
| 16 GB | 2-4 |
| 24 GB | 4-8 |
| 48+ GB | 8-16 |

```bash
adifind slide.svs --tissue_guidance --batch_size 8
```

Larger batches can improve utilization, but they also increase VRAM pressure. If you hit CUDA out-of-memory errors, reduce the batch size.

---

## Async I/O

Async I/O prefetches slide windows while the current batch is still being processed.

| Option | Default | Description |
|:-------|:--------|:------------|
| `ENABLE_ASYNC_IO` | `True` | Enable async prefetching |
| `ASYNC_CACHE_SIZE_MB` | `256` | Prefetch cache size |
| `ASYNC_MAX_WORKERS` | `4` | I/O worker threads |
| `ASYNC_PREFETCH_SIZE` | `12` | Windows to prefetch ahead |

For fast local storage, increase `ASYNC_PREFETCH_SIZE`. For slower or network-backed storage, tune worker count more conservatively.

---

## Memory Management

### Standard Mode

For systems with 64 GB RAM or more, default settings are usually sufficient:

| Option | Default | Description |
|:-------|:--------|:------------|
| `MAX_IO_WORKERS` | `25` | I/O thread pool size |
| `CHUNK_WORKERS` | `12` | Mask-processing threads |
| `MASK_CHUNK_SIZE` | `2048` | Mask chunk size in rows |
| `MAX_FULL_MASK_PIXELS` | `500000000` | Full-mask pixel limit before chunked handling |

### Low-Memory Mode

For tighter RAM budgets or very large slides:

```bash
adifind slide.svs --tissue_guidance --low_memory
```

This reduces worker counts, batch size, cache size, and some output scales so the run stays within a smaller memory budget.

### Disk-Backed Masks

For slides that exceed comfortable in-memory mask sizes:

```bash
adifind slide.svs --tissue_guidance --low_memory --memmap_mask
```

This stores the full-slide mask on disk via `numpy.memmap` instead of keeping it fully in RAM.

### Auto-Detection

AdiFind can inspect system RAM and enable low-memory mode automatically when available memory is below `LOW_MEMORY_THRESHOLD_GB`.

---

## I/O Optimization

| Option | Default | Description |
|:-------|:--------|:------------|
| `MAX_IO_WORKERS` | `25` | Thread pool for slide I/O |
| `PARALLEL_CHUNK_PROCESSING` | `True` | Parallelize mask processing |
| `INPLACE_LABEL_REMAPPING` | `True` | Remap labels in place to reduce memory pressure |
| `INCREMENTAL_PROPERTY_COLLECTION` | `True` | Collect properties incrementally |

### Storage Recommendations

| Storage | I/O workers | Notes |
|:--------|:------------|:------|
| NVMe SSD | 25-50 | Good fit for default settings |
| SATA SSD | 15-25 | Tune down if I/O becomes the bottleneck |
| HDD | 5-10 | More limited random-read performance |
| Network storage | 5-15 | Copying to local storage often helps |

---

## Performance Optimizer

AdiFind includes an automatic performance optimizer in `code/optimizations/performance_optimizer.py` that inspects the hardware and adjusts relevant settings:

- GPU detection
- tensor core detection
- memory-aware buffer sizing
- CUDA stream tuning

The optimizer runs automatically at startup.

---

## Benchmarking

### Profiling

```bash
adifind slide.svs --tissue_guidance --profiling
```

Adds detailed per-stage timing to the log output.

### Image-Saving Benchmark

```bash
adifind slide.svs --tissue_guidance --benchmark_saving
```

Compares the available save modes and reports timing and file-size differences.

### GPU Memory Profiling

```python
config.ENABLE_GPU_MEMORY_PROFILING = True
```

Logs GPU memory usage at each processing stage.

---

## Optimization Modules

AdiFind includes specialized optimization modules in `code/optimizations/`:

| Module | Purpose | Effect |
|:-------|:--------|:-------|
| `async_io.py` | Asynchronous window prefetching | Reads upcoming windows while compute is busy |
| `async_integration.py` | Drop-in async I/O wrapper | Applies the same prefetching approach to more call sites |
| `batch_inference.py` | Batched GPU inference | Groups more windows into each inference call |
| `gpu_acceleration.py` | GPU morphology, preprocessing, statistics | Moves array-heavy operations to the GPU when available |
| `gpu_label_mapping.py` | Chunked GPU label remapping | Offloads large-mask relabeling work from the CPU |
| `tensor_core_acceleration.py` | FP16 inference on tensor cores | Uses tensor cores on supported GPUs |
| `memory_streaming.py` | HDF5-backed streaming for gigapixel images | Keeps very large workloads within memory limits |
| `tissue_cache.py` | MD5-based tissue result caching | Avoids repeating tissue-detection work on reruns |
| `performance_optimizer.py` | Auto-tuning coordinator | Chooses settings based on detected hardware |

---

## Quick Tuning Checklist

1. Start with `--tissue_guidance` so empty background regions are skipped early.
2. Install CuPy if you use tumor-distance workflows and want GPU-backed array operations.
3. Match batch size to VRAM with `--batch_size`.
4. Use `--save_mode fast` when output image quality is less important than write time.
5. Use `--low_memory --memmap_mask` for constrained systems or very large slides.
6. Keep slides on fast local storage when possible.
7. Use SLURM arrays for large HPC batches when one-slide-per-job is practical.

---

## See Also

- [Features and Workflows](features.md#tissue-guidance) - Tissue guidance and related workflow effects
- [Configuration](configuration.md) - Tunable options and defaults
- [Architecture](architecture.md) - How the optimization paths fit into the pipeline
- [Deployment](deployment.md#hpc-deployment) - Cluster-specific runtime guidance

---

Back to [Documentation Index](index.md)
