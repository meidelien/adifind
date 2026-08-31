#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Execution Module
=====================

Main orchestration function for AdiFind WSI analysis.
Delegates processing to the pipeline module.
"""

import os
import time
import sys
import random
import logging
import gc
import glob
import tempfile
import traceback
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import json

# Optional: improves ANSI handling if someone runs in older Windows consoles.
# (Windows Terminal / VS Code terminal / Linux terminals are already fine.)
try:
    import colorama
    colorama.just_fix_windows_console()
except Exception:
    pass

from config import (
    config, paths, SORT_BY_FILE_SIZE, SIZE_SORT_ASCENDING,
    RESOLUTION_CACHE_PATH, RES_SORT_ASCENDING, SORT_BY_RESOLUTION
)

# Import new modular utilities
from argument_parser import parse_arguments, validate_inputs, get_image_files
from logging_utils import setup_logging, enable_console_logging
from configuration_manager import update_config_from_args

# Note: heavy runtime imports are loaded lazily so --help and --version can work
# without importing Torch, Detectron2, or OpenSlide-backed modules.
BatchProcessor = None
find_resumable_batches = None
calculate_area_statistics = None
calculate_distance_statistics = None
monitor = None
memory_manager = None
flush_gpu_memory = None
setup_gpu_device = None
get_gpu_runtime_summary = None
format_gpu_runtime_summary = None
ImageHandler = None
generate_sliding_windows = None
get_mpp = None
calculate_optimal_window_params = None
configure_adipocyte_model = None
process_all_windows = None
process_selected_windows_threaded = None
get_last_execution_diagnostics = None
log_resource_snapshot = None
apply_tissue_guidance = None
annotate_image_with_adipocytes = None
create_heatmap_visualization = None
export_qupath_annotations = None
export_results_csv = None
stitch_postprocessed_windows = None
save_mask_visualization = None
save_plain_image = None
save_sobel_inverted_image = None
_OPENSLIDE_MODULE = False
RESUME_INVOCATION_ARGS = {'resume_batch', 'resume_failed'}
TUMOR_DISTANCE_WARNING_MESSAGE = (
    "Adipocytes were detected and saved, but tumour-distance calculation failed. "
    "Tumour-distance values are unavailable for this image."
)
CLEANUP_FAILURE_MESSAGE = (
    "Detection appears to have completed, but AdiFind failed during internal mask cleanup "
    "before final saving. This is a software cleanup issue, not a slide-quality issue."
)

GPU_PROBE_STAGE_ORDER = (
    "cuda_device",
    "cuda_tensor",
    "gpu_inference",
    "cupy",
    "gpu_preprocessing",
    "gpu_label_mapping",
)

GPU_STAGE_LABELS = {
    "cuda_device": "CUDA device selection",
    "cuda_tensor": "CUDA tensor allocation/sync",
    "gpu_inference": "GPU inference model/device preparation",
    "cupy": "CuPy alloc/free/sync",
    "gpu_preprocessing": "GPU preprocessing micro-path",
    "gpu_label_mapping": "GPU label-mapping micro-path",
    "runtime_startup": "runtime startup",
    "runtime_image_load": "image load",
    "runtime_model_setup": "model setup",
    "runtime_window_generation": "window generation",
    "runtime_window_execution": "window execution",
}


def _should_run_annotated_image_phase():
    """Return True when any annotated-image-derived output needs to be generated."""
    return bool(
        getattr(config, 'SAVE_ANNOTATED_IMAGE', True)
        or (
            getattr(config, 'SAVE_TUMOR_ZONE_OVERLAY_IMAGE', False)
            and getattr(config, 'ENABLE_TUMOR_SEGMENTATION', False)
        )
    )


def _should_export_qupath_geojson():
    """Return True when QuPath GeoJSON export is enabled for this run."""
    return bool(
        getattr(config, 'ENABLE_QUPATH_EXPORT', False)
        and getattr(config, 'SAVE_QUPATH_GEOJSON', False)
    )


def _maybe_export_qupath_geojson(mask_areas, full_mask, output_dir, image_name, final_properties):
    """Run the QuPath GeoJSON export when enabled and report whether it executed."""
    if not _should_export_qupath_geojson():
        logging.info("\U0001F4C1 QuPath GeoJSON export skipped (disabled in config or via CLI override)")
        return False

    logging.info("\U0001F4C1 Saving QuPath GeoJSON export...")
    export_qupath_annotations(mask_areas, full_mask, output_dir, image_name, final_properties)
    return True


def _apply_saved_batch_args(args, saved_args):
    """Apply saved batch args while preserving explicit resume-run opt-ins."""
    keep_current_extended_properties = bool(getattr(args, 'extended_properties', False))

    for key, value in saved_args.items():
        if not hasattr(args, key):
            continue
        if key == 'resume_batch':
            continue
        if key == 'extended_properties' and keep_current_extended_properties:
            continue
        setattr(args, key, value)


def _safe_phase_pct(phase_time, total_time):
    """Return a safe phase percentage for profiling summaries."""
    if total_time <= 0:
        return 0.0
    return (phase_time / total_time) * 100.0


def _safe_ratio(numerator, denominator):
    """Return a safe ratio for derived metrics that can see zero-sized totals."""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _friendly_processing_error(exc):
    """Convert known internal exceptions into non-technical batch messages."""
    message = str(exc)
    if "_adifind_cleanup" in message:
        return CLEANUP_FAILURE_MESSAGE
    return message


def _compute_optional_tumor_distances(
    mask_areas,
    full_mask,
    tumor_mask_analysis,
    mpp,
    full_shape,
    image_name,
    distance_func,
    stats_func,
    analysis_downsample=160,
    return_closest_tumor_ids=False,
):
    """Compute optional tumour distances without failing completed adipocyte results."""
    try:
        distance_result = distance_func(
            mask_areas=mask_areas,
            full_mask=full_mask,
            tumor_mask_analysis=tumor_mask_analysis,
            mpp=mpp,
            full_shape=full_shape,
            analysis_downsample=analysis_downsample,
        )
        if isinstance(distance_result, tuple):
            adipocyte_distances, adipocyte_closest_tumor_ids = distance_result
        else:
            adipocyte_distances = distance_result
            adipocyte_closest_tumor_ids = {}

        if stats_func is not None:
            stats_func(adipocyte_distances)

        if return_closest_tumor_ids:
            return adipocyte_distances, adipocyte_closest_tumor_ids, ""
        return adipocyte_distances, ""
    except Exception:
        logging.exception("Tumour-distance calculation failed for %s", image_name)
        logging.warning(TUMOR_DISTANCE_WARNING_MESSAGE)
        if return_closest_tumor_ids:
            return {}, {}, TUMOR_DISTANCE_WARNING_MESSAGE
        return {}, TUMOR_DISTANCE_WARNING_MESSAGE


def _log_visualization_summary(
    props_extraction_time,
    annotated_image_time,
    distance_image_time,
    plain_image_time,
    sobel_image_time,
    total_visualization_time,
    adipocyte_count,
    image_width,
    image_height,
):
    """Log the visualization profiling summary without raising on zero-time runs."""
    logging.info("\U0001F3A8 VISUALIZATION PROFILING COMPLETE:")
    logging.info("=" * 60)
    logging.info(
        f"\U0001F4CA Properties Extraction:    {props_extraction_time:8.3f}s "
        f"({_safe_phase_pct(props_extraction_time, total_visualization_time):5.1f}%)"
    )
    logging.info(
        f"\U0001F5BC\uFE0F Annotated Image Creation: {annotated_image_time:8.3f}s "
        f"({_safe_phase_pct(annotated_image_time, total_visualization_time):5.1f}%)"
    )
    logging.info(
        f"\U0001F308 Distance Image Creation:  {distance_image_time:8.3f}s "
        f"({_safe_phase_pct(distance_image_time, total_visualization_time):5.1f}%)"
    )
    logging.info(
        f"\U0001F5BC\uFE0F Plain Image Export:      {plain_image_time:8.3f}s "
        f"({_safe_phase_pct(plain_image_time, total_visualization_time):5.1f}%)"
    )
    logging.info(
        f"\U0001F9EA Sobel+Inverted Export:    {sobel_image_time:8.3f}s "
        f"({_safe_phase_pct(sobel_image_time, total_visualization_time):5.1f}%)"
    )
    logging.info("=" * 60)
    logging.info(f"\U0001F3A8 TOTAL VISUALIZATION TIME:  {total_visualization_time:8.3f}s")
    if total_visualization_time > 0 and adipocyte_count > 0:
        logging.info(
            f"\U0001F4C8 Visualization efficiency: "
            f"{_safe_ratio(adipocyte_count, total_visualization_time):.1f} adipocytes/second"
        )
    else:
        logging.info("\U0001F4C8 Visualization efficiency: N/A (no adipocytes or instant completion)")

    image_size_mb = ((image_width * image_height * 3) / 1024 / 1024) if image_width > 0 and image_height > 0 else 0.0
    logging.info("\U0001F5BC\uFE0F Image Metrics:")
    logging.info(f"   \u2022 Image dimensions: {image_width} x {image_height}")
    logging.info(f"   \u2022 Estimated image size: {image_size_mb:.1f} MB")
    logging.info(f"   \u2022 Adipocytes visualized: {adipocyte_count:,}")
    if image_size_mb > 0:
        logging.info(f"   \u2022 Visualization density: {_safe_ratio(adipocyte_count, image_size_mb):.1f} adipocytes/MB")
    else:
        logging.info("   \u2022 Visualization density: N/A (invalid image size)")


def _load_runtime_dependencies():
    """Import heavyweight runtime modules only when real execution is needed."""
    global calculate_area_statistics
    global calculate_distance_statistics
    global monitor
    global memory_manager
    global flush_gpu_memory
    global get_gpu_runtime_summary
    global format_gpu_runtime_summary
    global setup_gpu_device
    global ImageHandler
    global generate_sliding_windows
    global get_mpp
    global calculate_optimal_window_params
    global configure_adipocyte_model
    global process_all_windows
    global process_selected_windows_threaded
    global get_last_execution_diagnostics
    global log_resource_snapshot
    global apply_tissue_guidance
    global annotate_image_with_adipocytes
    global create_heatmap_visualization
    global export_qupath_annotations
    global export_results_csv
    global stitch_postprocessed_windows
    global save_mask_visualization
    global save_plain_image
    global save_sobel_inverted_image

    if monitor is not None:
        return

    from statistics_utils import (
        calculate_area_statistics as _calculate_area_statistics,
        calculate_distance_statistics as _calculate_distance_statistics,
    )
    from system_utils import (
        monitor as _monitor,
        memory_manager as _memory_manager,
        flush_gpu_memory as _flush_gpu_memory,
        get_gpu_runtime_summary as _get_gpu_runtime_summary,
        format_gpu_runtime_summary as _format_gpu_runtime_summary,
        setup_gpu_device as _setup_gpu_device,
    )
    from image_processing import (
        ImageHandler as _ImageHandler,
        generate_sliding_windows as _generate_sliding_windows,
        get_mpp as _get_mpp,
        calculate_optimal_window_params as _calculate_optimal_window_params,
    )
    from models import configure_adipocyte_model as _configure_adipocyte_model
    from core_processing import (
        process_all_windows as _process_all_windows,
        process_selected_windows_threaded as _process_selected_windows_threaded,
        get_last_execution_diagnostics as _get_last_execution_diagnostics,
        log_resource_snapshot as _log_resource_snapshot,
    )
    from tissue_guided_processing import apply_tissue_guidance as _apply_tissue_guidance
    from visualization import (
        annotate_image_with_adipocytes as _annotate_image_with_adipocytes,
        create_heatmap_visualization as _create_heatmap_visualization,
        export_qupath_annotations as _export_qupath_annotations,
        export_results_csv as _export_results_csv,
        stitch_postprocessed_windows as _stitch_postprocessed_windows,
        save_mask_visualization as _save_mask_visualization,
        save_plain_image as _save_plain_image,
        save_sobel_inverted_image as _save_sobel_inverted_image,
    )


    calculate_area_statistics = _calculate_area_statistics
    calculate_distance_statistics = _calculate_distance_statistics
    monitor = _monitor
    memory_manager = _memory_manager
    flush_gpu_memory = _flush_gpu_memory
    get_gpu_runtime_summary = _get_gpu_runtime_summary
    format_gpu_runtime_summary = _format_gpu_runtime_summary
    setup_gpu_device = _setup_gpu_device
    ImageHandler = _ImageHandler
    generate_sliding_windows = _generate_sliding_windows
    get_mpp = _get_mpp
    calculate_optimal_window_params = _calculate_optimal_window_params
    configure_adipocyte_model = _configure_adipocyte_model
    process_all_windows = _process_all_windows
    process_selected_windows_threaded = _process_selected_windows_threaded
    get_last_execution_diagnostics = _get_last_execution_diagnostics
    log_resource_snapshot = _log_resource_snapshot
    apply_tissue_guidance = _apply_tissue_guidance
    annotate_image_with_adipocytes = _annotate_image_with_adipocytes
    create_heatmap_visualization = _create_heatmap_visualization
    export_qupath_annotations = _export_qupath_annotations
    export_results_csv = _export_results_csv
    stitch_postprocessed_windows = _stitch_postprocessed_windows
    save_mask_visualization = _save_mask_visualization
    save_plain_image = _save_plain_image
    save_sobel_inverted_image = _save_sobel_inverted_image


def _load_batch_processing_dependencies():
    """Import batch-processing helpers only when batch commands are exercised."""
    global BatchProcessor
    global find_resumable_batches

    if BatchProcessor is not None and find_resumable_batches is not None:
        return

    from batch_processing import (
        BatchProcessor as _BatchProcessor,
        find_resumable_batches as _find_resumable_batches,
    )

    BatchProcessor = _BatchProcessor
    find_resumable_batches = _find_resumable_batches


def _get_openslide_module():
    """Load OpenSlide lazily so CLI help is not coupled to native libraries."""
    global _OPENSLIDE_MODULE

    if _OPENSLIDE_MODULE is False:
        try:
            import openslide as _openslide
        except Exception:
            _OPENSLIDE_MODULE = None
        else:
            _OPENSLIDE_MODULE = _openslide

    return _OPENSLIDE_MODULE


def display_startup_banner():
    """Display professional startup banner with branding."""
    # --- Neon/Rainbow ASCII logo (generated) ---
    import colorsys

    RAW_LOGO_ADI = [
        " █████╗ ██████╗ ██╗",
        "██╔══██╗██╔══██╗██║",
        "███████║██║  ██║██║",
        "██╔══██║██║  ██║██║",
        "██║  ██║██████╔╝██║",
        "╚═╝  ╚═╝╚═════╝ ╚═╝",
    ]

    RAW_LOGO_FIND = [
        "███████╗██╗███╗   ██╗██████╗",
        "██╔════╝██║████╗  ██║██╔══██╗",
        "█████╗  ██║██╔██╗ ██║██║  ██║",
        "██╔══╝  ██║██║╚██╗██║██║  ██║",
        "██║     ██║██║ ╚████║██████╔╝",
        "╚═╝     ╚═╝╚═╝  ╚═══╝╚═════╝",
    ]
    def _neon_logo_lines(raw_lines, indent="        ", group=5, hue_shift=0.0, glow_div=20, row_hue_step=0.51, value_scale=1.0):
        """
        Creates a per-character neon gradient.
        group: higher => more blocky/pixelated gradient; 1 => smoothest
        hue_shift: rotate rainbow (0..1)
        glow_div: (unused now; kept for compatibility)
        row_hue_step: slight hue shift per row for depth
        """
        max_w = max(len(s) for s in raw_lines)
        out = []
        for y, line in enumerate(raw_lines):
            line = line.ljust(max_w)
            buf = []
            for x, ch in enumerate(line):
                xg = (x // group) * group
                h = ((xg / max_w) + hue_shift + (y * row_hue_step)) % 1.0
                r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
                R, G, B = [max(0, min(255, int(c * 255 * value_scale))) for c in (r, g, b)]

                if ch == " ":
                    buf.append(" ")
                else:
                    # Foreground-only (no background fill)
                    buf.append(f"\033[1m\033[38;2;{R};{G};{B}m{ch}\033[0m")
            out.append(indent + "".join(buf))
        return out

    # Two-part block logo (ADI + FIND) to preserve the full wordmark.
    hue_shift_adi = 0.56  # deep blue/teal blend
    adi_lines = _neon_logo_lines(
        RAW_LOGO_ADI,
        indent="        ",
        group=9999,
        hue_shift=hue_shift_adi,
        glow_div=12,
        row_hue_step=0.0,
        value_scale=0.80
    )
    find_color = "\033[1;32m"  # match "Detection & Analysis" green
    find_lines = [f"{find_color}{line}\033[0m" for line in RAW_LOGO_FIND]
    logo_lines = [f"{a} {b}" for a, b in zip(adi_lines, find_lines)]

    def _format_setting_line(label: str, enabled: bool, disabled_detail: str = "disabled \u26A0\uFE0F") -> str:
        if enabled:
            return f"             \033[0;36m  \u2022 {label:<19}: enabled \u2714\uFE0F\033[0m"
        return f"             \033[1;33m  \u2022 {label:<19}: {disabled_detail}\033[0m"

    banner_lines = [
        "",
        "",
         *logo_lines,
        "",
        "                \033[1;34mAdipocyte\033[0m \033[1;32mDetection & Analysis\033[0m",
        "                 \033[0;33mVersion 1.0 - April 2026\033[0m",
        "              \033[1;34mhttps://github.com/meidelien/AdiFind\033[0m",
        "                \033[1;35mDeveloped by Martin Eide Lien\033[0m",
        "                 \033[1;36mUniversity of Bergen, Norway\033[0m",
        "",
        "              \033[1;37m\u2699\uFE0F  Settings:\033[0m",
        _format_setting_line("Tumor segmentation", config.ENABLE_TUMOR_SEGMENTATION),
        _format_setting_line("Tissue-Guidance", config.ENABLE_TISSUE_GUIDANCE),
        _format_setting_line("GPU inference", config.USE_GPU_INFERENCE, "disabled (CPU only) \u26A0\uFE0F"),
        _format_setting_line("CuPy ops", config.USE_CUPY),
        _format_setting_line("GPU preprocessing", config.USE_GPU_PREPROCESSING),
        _format_setting_line("QuPath output", _should_export_qupath_geojson()),
        ""
    ]

    def _env_flag_true(name: str) -> bool:
        v = os.environ.get(name, "").strip().lower()
        return v in {"1", "true", "yes", "on"}

    def _isatty() -> bool:
        try:
            return sys.stdout.isatty()
        except Exception:
            return False

    # Optional environment controls:
    #   ADIFIND_NO_BANNER=1  -> suppress banner entirely
    #   ADIFIND_NO_ANIM=1    -> print banner without animation
    if _env_flag_true("ADIFIND_NO_BANNER"):
        return

    animate = not _env_flag_true("ADIFIND_NO_ANIM")

    # Try to ensure UTF-8 output (helps avoid odd glyph substitution on some setups)
    try:
        if getattr(sys.stdout, "reconfigure", None) and (sys.stdout.encoding or "").lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not animate:
        for line in banner_lines:
            print(line)
        return

    # --- Smooth column wipe for the ASCII-art block only ---
    import re

    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
    ART_CHARS = ("\u2588", "\u2554", "\u255A", "\u2557", "\u255D", "\u2550", "\u2551")

    def _is_art_line(s: str) -> bool:
        return any(ch in s for ch in ART_CHARS)

    def _leading_spaces(s: str) -> str:
        n = len(s) - len(s.lstrip(" "))
        return s[:n]

    def _tokenize_visible_cells(s: str):
        """
        Split into 'cells' where each cell is (any ANSI codes) + (one visible char),
        and attach immediate reset codes (\x1b[0m / \x1b[m) to that same cell.
        This prevents color bleed during partial reveal frames.
        """
        out = []
        i = 0
        pending = ""
        RESET_CODES = {"\x1b[0m", "\x1b[m"}

        while i < len(s):
            if s[i] == "\x1b":
                m = ANSI_RE.match(s, i)
                if m:
                    pending += m.group(0)
                    i = m.end()
                    continue

            # One visible character
            cell = pending + s[i]
            pending = ""
            i += 1

            # Attach immediate reset codes to the same cell (but don't consume new styling codes)
            while i < len(s) and s[i] == "\x1b":
                m = ANSI_RE.match(s, i)
                if not m:
                    break
                code = m.group(0)
                if code in RESET_CODES:
                    cell += code
                    i = m.end()
                else:
                    break

            out.append(cell)

        if pending:
            # trailing ANSI; attach to last cell if possible
            if out:
                out[-1] += pending
            else:
                out.append(pending)
        return out

    def _cursor_to(row_1_indexed: int, col_1_indexed: int = 1):
        sys.stdout.write(f"\x1b[{row_1_indexed};{col_1_indexed}H")

    # Locate first contiguous "art block" in the banner
    art_start = None
    for i, line in enumerate(banner_lines):
        if _is_art_line(line):
            art_start = i
            break

    if art_start is None:
        # Fallback: no art detected
        for line in banner_lines:
            print(line)
        return

    art_end = art_start
    while art_end < len(banner_lines) and _is_art_line(banner_lines[art_end]):
        art_end += 1  # exclusive

    # Pre-tokenize art lines (excluding left indentation spaces)
    art_prefixes = []
    art_tokens = []
    widths = []
    for i in range(art_start, art_end):
        line = banner_lines[i]
        prefix = _leading_spaces(line)
        rest = line[len(prefix):]
        tokens = _tokenize_visible_cells(rest)
        art_prefixes.append(prefix)
        art_tokens.append(tokens)
        widths.append(len(tokens))

    max_w = max(widths) if widths else 0

    # Animation parameters
    delay = float(os.environ.get("ADIFIND_ANIM_DELAY", "0.015"))
    step = int(os.environ.get("ADIFIND_ANIM_STEP", "1"))
    if step < 1:
        step = 1

    # Clear screen once, print full banner scaffold with art hidden
    sys.stdout.write("\x1b[2J\x1b[H")
    for i, line in enumerate(banner_lines):
        if art_start <= i < art_end:
            prefix = art_prefixes[i - art_start]
            sys.stdout.write(prefix + (" " * max_w) + "\n")
        else:
            sys.stdout.write(line + "\n")
    sys.stdout.flush()

    # Reveal columns from left -> right WITHOUT shifting any lines.
    # Treat each art line as left-aligned content padded on the RIGHT to max_w.
    for revealed in range(0, max_w + step, step):
        revealed = min(revealed, max_w)
        visible = revealed

        _cursor_to(art_start + 1, 1)

        for prefix, tokens in zip(art_prefixes, art_tokens):
            padded = tokens + ([" "] * (max_w - len(tokens)))
            sys.stdout.write("\x1b[2K")
            sys.stdout.write(prefix + "".join(padded[:visible]) + (" " * (max_w - visible)) + "\x1b[0m\n")

        sys.stdout.flush()
        time.sleep(delay)

    # Put cursor at the bottom of the banner so subsequent prints/logs don't overwrite it
    _cursor_to(len(banner_lines) + 1, 1)
    sys.stdout.flush()


def _print_startup_banner():
    """Render the runtime banner with a simple fallback if ANSI art fails."""
    try:
        display_startup_banner()
    except Exception:
        try:
            from config import BANNER
            print(BANNER)
        except Exception:
            print("AdiFind: Advanced Adipocyte Detection in Whole Slide Images")
            print("=" * 60)


def planned_gpu_probe_stages(
    *,
    use_cupy: bool,
    use_gpu_preprocessing: bool,
    enable_gpu_label_mapping: bool,
):
    """Return the ordered GPU probe stages for the effective runtime flags."""
    stages = ["cuda_device", "cuda_tensor", "gpu_inference"]
    if use_cupy:
        stages.append("cupy")
    if use_gpu_preprocessing:
        stages.append("gpu_preprocessing")
    if enable_gpu_label_mapping:
        stages.append("gpu_label_mapping")
    return stages


def get_gpu_probe_stage_plan():
    """Return the ordered GPU probe stages for the current runtime config."""
    return planned_gpu_probe_stages(
        use_cupy=bool(getattr(config, 'USE_CUPY', False)),
        use_gpu_preprocessing=bool(getattr(config, 'USE_GPU_PREPROCESSING', False)),
        enable_gpu_label_mapping=bool(getattr(config, 'ENABLE_GPU_LABEL_MAPPING', False)),
    )


def _emit_stage_marker(prefix: str, stage: str):
    print(f"{prefix}: {stage}", flush=True)


def _emit_probe_stage(stage: str):
    _emit_stage_marker("GPU_PROBE_STAGE", stage)


def _emit_runtime_stage(stage: str):
    _emit_stage_marker("GPU_RUNTIME_STAGE", stage)


def _log_gpu_runtime_stack(args):
    """Emit a small, centralized summary of the effective GPU runtime stack."""
    global get_gpu_runtime_summary
    global format_gpu_runtime_summary

    if get_gpu_runtime_summary is None or format_gpu_runtime_summary is None:
        from system_utils import (
            get_gpu_runtime_summary as _get_gpu_runtime_summary,
            format_gpu_runtime_summary as _format_gpu_runtime_summary,
        )
        get_gpu_runtime_summary = _get_gpu_runtime_summary
        format_gpu_runtime_summary = _format_gpu_runtime_summary

    summary = get_gpu_runtime_summary(getattr(args, 'gpu_id', 0))
    summary_text = format_gpu_runtime_summary(summary)
    logging.info("GPU runtime stack: %s", summary_text)
    print(f"GPU_RUNTIME_SUMMARY: {summary_text}", flush=True)
    return summary


def _setup_probe_logging(verbose: bool):
    """Configure console logging for the hidden GPU probe mode."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        force=True,
    )


def _run_gpu_probe(args):
    """Run staged GPU self-tests before the real desktop analysis starts."""
    current_stage = "startup"

    def _probe_fail(message: str, *, stage: Optional[str] = None, with_traceback: bool = False):
        stage_name = stage or current_stage
        logging.error("GPU probe failed during %s: %s", stage_name, message)
        print(f"GPU_PROBE_ERROR: {stage_name}: {message}", flush=True)
        if with_traceback:
            traceback.print_exc()
        return 2

    try:
        logging.info("Starting staged GPU probe...")
        logging.info("GPU probe stage plan: %s", " -> ".join(get_gpu_probe_stage_plan()))

        for stage in get_gpu_probe_stage_plan():
            current_stage = stage
            _emit_probe_stage(stage)

            if stage == "cuda_device":
                import torch
                if not torch.cuda.is_available():
                    return _probe_fail("CUDA is not available for the requested GPU mode")
                torch.cuda.set_device(args.gpu_id)
                summary = _log_gpu_runtime_stack(args)
                logging.info("GPU probe: using CUDA device %s", summary.get('device_name'))

            elif stage == "cuda_tensor":
                tensor = torch.zeros((8, 8), device=f"cuda:{args.gpu_id}", dtype=torch.float32)
                tensor += 1.0
                torch.cuda.synchronize()
                del tensor

            elif stage == "gpu_inference":
                _load_runtime_dependencies()
                predictor = configure_adipocyte_model()
                del predictor
                flush_gpu_memory()
                gc.collect()

            elif stage == "cupy":
                try:
                    import cupy as cp
                except ImportError as exc:
                    return _probe_fail(f"CuPy is required for the selected GPU mode but is unavailable: {exc}")
                cp.cuda.Device(args.gpu_id).use()
                arr = cp.arange(64, dtype=cp.float32).reshape(8, 8)
                arr = arr * 2.0
                cp.cuda.Stream.null.synchronize()
                del arr
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass

            elif stage == "gpu_preprocessing":
                try:
                    from optimizations.gpu_acceleration import gpu_image_inversion, gpu_sobel_preprocessing
                except Exception as exc:
                    return _probe_fail(f"GPU preprocessing helpers could not be loaded: {exc}")

                sample = np.arange(64 * 64 * 3, dtype=np.uint8).reshape((64, 64, 3))
                inverted = gpu_image_inversion(sample)
                processed = gpu_sobel_preprocessing(sample, apply_bilateral=False)
                if inverted.shape != sample.shape or processed.shape != sample.shape:
                    return _probe_fail("GPU preprocessing returned unexpected output dimensions")
                del inverted, processed

            elif stage == "gpu_label_mapping":
                try:
                    from optimizations.gpu_acceleration import GPUAcceleratedOperations
                except Exception as exc:
                    return _probe_fail(f"GPU label-mapping helpers could not be loaded: {exc}")

                ops = GPUAcceleratedOperations()
                if not getattr(ops, 'gpu_available', False):
                    return _probe_fail("GPU label mapping is enabled but the GPU label-mapping backend is unavailable")
                mask = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint32)
                mapped = ops.gpu_label_mapping(mask, {1: 1, 2: 2})
                if mapped.shape != mask.shape:
                    return _probe_fail("GPU label mapping returned unexpected output dimensions")
                del mapped

        logging.info("GPU probe completed successfully")
        print("GPU probe completed successfully.", flush=True)
        return 0
    except Exception as exc:
        return _probe_fail(str(exc), with_traceback=bool(getattr(args, 'verbose', False)))


# ================================================================
# BATCH SUMMARY FUNCTIONS
# ================================================================

def _create_batch_summary(base_output_dir, batch_results, args, total_batch_time):
    """
    Create comprehensive batch processing summary folder and files.

    Args:
        base_output_dir: Base directory for batch processing
        batch_results: List of processing results for each image
        args: Command line arguments
        total_batch_time: Total processing time for the batch
    """
    import csv
    import json
    from datetime import datetime

    # Create summary subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = os.path.join(base_output_dir, f"summary_{timestamp}")
    os.makedirs(summary_dir, exist_ok=True)

    # 1. Create detailed summary CSV (matching original format)
    summary_csv_path = os.path.join(summary_dir, f"adipocyte_summary_{timestamp}.csv")
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Write header exactly as in original
        csvfile.write(
            "Image Name,Total Adipocytes,Median Adipocyte Size (microns squared),"
            "Average Adipocyte Size (microns squared),Processing Time (HH:MM:SS),Tumour Count\n"
        )

        # Write data for each image
        for result in batch_results:
            # Format processing time as HH:MM:SS
            hours, remainder = divmod(result['total_time'], 3600)
            minutes, seconds = divmod(remainder, 60)
            processing_time_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

            # Get adipocyte size statistics from individual result files
            median_size, average_size = _get_adipocyte_size_stats(result['output_dir'], result['image_name'])

            csvfile.write(
                f"{result['image_name']},{result['total_adipocytes']},"
                f"{median_size},{average_size},"
                f"{processing_time_formatted},{result['num_tumors']}\n"
            )

    # 2. Save configuration parameters (matching original format)
    config_json_path = os.path.join(summary_dir, f"configuration_parameters_{timestamp}.json")
    config_dict = _get_configuration_parameters(args)
    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, default=str)

    # 3. Create batch statistics summary
    batch_stats_path = os.path.join(summary_dir, f"batch_statistics_{timestamp}.txt")
    _create_batch_statistics_file(batch_stats_path, batch_results, total_batch_time)

    # 4. Create processing log summary
    log_summary_path = os.path.join(summary_dir, f"processing_log_summary_{timestamp}.txt")
    _create_processing_log_summary(log_summary_path, batch_results, args)

    print(f"\U0001F4CB Comprehensive summary created in: {summary_dir}")
    print(f"   \u2022 Detailed CSV: adipocyte_summary_{timestamp}.csv")
    print(f"   \u2022 Configuration: configuration_parameters_{timestamp}.json")
    print(f"   \u2022 Statistics: batch_statistics_{timestamp}.txt")
    print(f"   \u2022 Log summary: processing_log_summary_{timestamp}.txt")


def _get_adipocyte_size_stats(output_dir, image_name):
    """Get median and average adipocyte size from CSV files."""
    try:
        csv_path = os.path.join(output_dir, f"{image_name}_adipocyte_information.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            if 'Area_Microns_Squared' in df.columns and len(df) > 0:
                median_size = df['Area_Microns_Squared'].median()
                average_size = df['Area_Microns_Squared'].mean()
                return median_size, average_size
    except Exception:
        pass
    return 0, 0


def _get_configuration_parameters(args):
    """Get all configuration parameters for saving."""
    from config import config, paths

    config_dict = {
        # Analysis Parameters
        'MIN_ADIPOCYTE_AREA_MICRONS': config.MIN_ADIPOCYTE_AREA_MICRONS,
        'MAX_ADIPOCYTE_AREA_MICRONS': config.MAX_ADIPOCYTE_AREA_MICRONS,
        'GRID_CELL_SIZE_MICRONS': config.GRID_CELL_SIZE_MICRONS,
        'IOU_THRESHOLD': config.IOU_THRESHOLD,
        'MERGE_IOU_THRESHOLD': config.MERGE_IOU_THRESHOLD,
        'CONFIDENCE_THRESHOLD': config.CONFIDENCE_THRESHOLD,
        'SCALING_FACTOR': config.SCALING_FACTOR,
        'DEFAULT_MPP': config.DEFAULT_MPP,

        # Processing Parameters
        'DESIRED_RESOLUTION_LEVEL': config.DESIRED_RESOLUTION_LEVEL,
        'MAX_IO_WORKERS': config.MAX_IO_WORKERS,
        'WINDOW_SIZE': args.window_size,
        'STRIDE': args.stride,

        # Image Processing
        'APPLY_IMAGE_INVERSION': config.APPLY_IMAGE_INVERSION,
        'APPLY_SOBEL_FILTER': config.APPLY_SOBEL_FILTER,
        'APPLY_BILATERAL_FILTER': config.APPLY_BILATERAL_FILTER,

        # Model Paths
        'ADIPOCYTE_MODEL_DIR': paths.ADIPOCYTE_MODEL_DIR,
        'ADIPOCYTE_MODEL_CHECKPOINT': paths.ADIPOCYTE_MODEL_CHECKPOINT,
        'TUMOR_MODEL_DIR': paths.TUMOR_MODEL_DIR,
        'TUMOR_MODEL_CHECKPOINT': paths.TUMOR_MODEL_CHECKPOINT,

        # Feature Toggles
        'ENABLE_TUMOR_SEGMENTATION': config.ENABLE_TUMOR_SEGMENTATION,
        'ENABLE_TISSUE_GUIDANCE': config.ENABLE_TISSUE_GUIDANCE,
        'USE_GPU_INFERENCE': config.USE_GPU_INFERENCE,
        'USE_CUPY': config.USE_CUPY,
        'USE_GPU_PREPROCESSING': config.USE_GPU_PREPROCESSING,
        'ENABLE_GPU_LABEL_MAPPING': config.ENABLE_GPU_LABEL_MAPPING,
        'DEBUG_MODE': config.DEBUG_MODE,
        'DEBUG_SAVE_UNPROCESSED_WINDOWS': getattr(config, 'DEBUG_SAVE_UNPROCESSED_WINDOWS', False),

        # Output Options
        'SAVE_ANNOTATED_IMAGE': config.SAVE_ANNOTATED_IMAGE,
        'SAVE_DISTANCE_COLORED_IMAGE': config.SAVE_DISTANCE_COLORED_IMAGE,
        'SAVE_QUPATH_GEOJSON': config.SAVE_QUPATH_GEOJSON,
        'SAVE_QUPATH_SCRIPT': config.SAVE_QUPATH_SCRIPT,

        # Command Line Arguments
        'input_path': args.image_path,
        'output_directory': args.output_dir,
        'gpu_id': args.gpu_id,
    }

    return config_dict


def _create_batch_statistics_file(file_path, batch_results, total_batch_time):
    """Create batch statistics summary file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("AdiFind Batch Processing Statistics\n")
        f.write("=" * 50 + "\n\n")

        # Overall statistics
        total_images = len(batch_results)
        total_adipocytes = sum(r['total_adipocytes'] for r in batch_results)
        total_tumors = sum(r['num_tumors'] for r in batch_results)
        avg_time = total_batch_time / total_images if total_images > 0 else 0

        f.write(f"Total Images Processed: {total_images}\n")
        f.write(f"Total Adipocytes Detected: {total_adipocytes}\n")
        f.write(f"Total Tumor Regions Found: {total_tumors}\n")
        f.write(f"Total Processing Time: {total_batch_time:.1f} seconds\n")
        f.write(f"Average Time per Image: {avg_time:.1f} seconds\n\n")

        # Per-image breakdown
        f.write("Per-Image Breakdown:\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(batch_results, 1):
            f.write(f"{i:2d}. {result['image_name']}\n")
            f.write(f"    Adipocytes: {result['total_adipocytes']}\n")
            f.write(f"    Tumors: {result['num_tumors']}\n")
            f.write(f"    Time: {result['total_time']:.1f}s\n")
            f.write(f"    Windows: {result['total_windows']}\n\n")


def _create_processing_log_summary(file_path, batch_results, args):
    """Create processing log summary file."""
    from datetime import datetime

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("AdiFind Processing Log Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Processing Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Path: {args.image_path}\n")
        f.write(f"Processing Mode: Batch Processing\n")
        f.write(f"Images Found: {len(batch_results)}\n\n")

        f.write("Configuration Summary:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Window Size: {args.window_size}\n")
        f.write(f"Stride: {args.stride}\n")
        f.write(f"GPU ID: {args.gpu_id}\n")
        f.write(f"Tumor Segmentation: {'Enabled' if config.ENABLE_TUMOR_SEGMENTATION else 'Disabled'}\n")
        f.write(f"Tissue Guidance: {'Enabled' if config.ENABLE_TISSUE_GUIDANCE else 'Disabled'}\n")
        f.write(f"Debug Mode: {'Enabled' if config.DEBUG_MODE else 'Disabled'}\n\n")
        if config.DEBUG_MODE:
            f.write(f"Debug Unprocessed Windows: {'Enabled' if getattr(config, 'DEBUG_SAVE_UNPROCESSED_WINDOWS', False) else 'Disabled'}\n\n")

        f.write("Processing Results:\n")
        f.write("-" * 20 + "\n")
        for result in batch_results:
            f.write(f"\u2705 {result['image_name']}: {result['total_adipocytes']} adipocytes\n")


# ================================================================
# SINGLE IMAGE PROCESSING FUNCTION
# ================================================================

def process_single_image(image_path, args, output_dir):
    """
    Process a single image file.

    Args:
        image_path: Path to the image file
        args: Parsed command line arguments
        output_dir: Specific output directory for this image

    Returns:
        dict: Processing results and statistics
    """
    _load_runtime_dependencies()
    image_name = Path(image_path).stem

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Update the temporary output dir for logging
    temp_output_dir = args.output_dir
    args.output_dir = output_dir

    image_handler = None
    full_mask = None
    mask_cleanup = None
    execution_diagnostics = {}
    phase_timings = {}
    cleanup_done = [False]
    processing_warnings = []

    def _cleanup_memmap():
        nonlocal full_mask, mask_cleanup
        if cleanup_done[0]:
            return
        try:
            # Drop references before cleanup to release Windows file handles
            full_mask = None
            gc.collect()
            if mask_cleanup is not None:
                mask_cleanup()
            logging.info("Memmap cleanup completed")
        except Exception as e:
            logging.warning(f"Memmap cleanup failed: {e}")
        finally:
            cleanup_done[0] = True
            mask_cleanup = None
            _delete_memmap_files()
            log_resource_snapshot("after_image_cleanup", execution_diagnostics)

    def _delete_memmap_files():
        """Best-effort deletion of any AdiFind memmap files in known locations."""
        candidates = []
        for base_dir in (output_dir, tempfile.gettempdir()):
            if base_dir and os.path.isdir(base_dir):
                candidates.extend(glob.glob(os.path.join(base_dir, "adifind_mask_*.memmap")))
        for path in candidates:
            try:
                os.remove(path)
                logging.debug(f"\U0001F5D1\uFE0F Deleted memmap file: {path}")
            except FileNotFoundError:
                continue
            except Exception as e:
                logging.warning(f"\u26A0\uFE0F Could not delete memmap file {path}: {e}")

    def _record_phase(name, seconds):
        phase_timings[name] = float(seconds)
        logging.info("PHASE TIMING [%s] %.3fs", name, seconds)
    try:
        logging.debug(f"\U0001F4C1 Processing: {image_path}")
        logging.debug(f"\U0001F4C2 Output: {output_dir}")
        logging.debug(f"\U0001F527 Window size: {args.window_size}")
        logging.debug(f"\U0001F463 Stride: {args.stride}")

        # ================================================================
        # STEP 1: LOAD AND INITIALIZE IMAGE
        # ================================================================

        _emit_runtime_stage("runtime_image_load")
        logging.debug("\U0001F4D6 Loading whole slide image...")
        start_time = time.time()

        image_handler = ImageHandler(image_path)

        # Get the MPP and calculate optimal window parameters
        mpp = get_mpp(image_path)

        # Safety check: ensure MPP is not zero to prevent division by zero
        if mpp <= 0:
            logging.warning(f"\u26A0\uFE0F  Invalid MPP value ({mpp}), using default: {config.DEFAULT_MPP}")
            mpp = config.DEFAULT_MPP

        final_window_size, final_stride = calculate_optimal_window_params(
            mpp, args.window_size, args.stride
        )

        logging.debug(f"\U0001F4CF Image dimensions: {image_handler.width} x {image_handler.height}")
        logging.debug(f"\U0001F52C Microns per pixel (MPP): {mpp:.4f}")
        logging.debug(f"\U0001F4D0 Dynamic window size: {final_window_size}")
        logging.debug(f"\U0001F463 Dynamic stride: {final_stride}")

        # Debug: Log the exact MPP value and calculations
        logging.debug(f"\U0001F50D DEBUG - MPP value: {mpp}, type: {type(mpp)}")
        logging.debug(f"\U0001F50D DEBUG - MPP squared: {mpp ** 2}")
        logging.debug(f"\U0001F50D DEBUG - MIN_ADIPOCYTE_AREA_MICRONS: {config.MIN_ADIPOCYTE_AREA_MICRONS}")
        logging.debug(f"\U0001F50D DEBUG - MAX_ADIPOCYTE_AREA_MICRONS: {config.MAX_ADIPOCYTE_AREA_MICRONS}")
        log_resource_snapshot("slide_loaded", execution_diagnostics)

        # Log level information if available
        if hasattr(image_handler, 'slide') and hasattr(image_handler.slide, 'level_dimensions'):
            levels_available = len(image_handler.slide.level_dimensions)
            logging.debug(f"\U0001F50D Levels available: {levels_available}")
        else:
            logging.debug("\U0001F50D Single level image (non-pyramidal)")

        # ================================================================
        # STEP 2: CONFIGURE MODELS
        # ================================================================

        _emit_runtime_stage("runtime_model_setup")
        logging.debug("\U0001F916 Configuring detection model...")
        predictor = configure_adipocyte_model()

        monitor.log_gpu_status("Adipocyte model loaded")

        # Configure tumor model if enabled
        tumor_predictor = None
        configure_tumour_model = None
        optimized_segment_tumour_on_thumbnail = None
        save_tumor_csv = None
        save_tumor_thumbnail_overlay = None
        optimized_compute_adipocyte_distance_metrics = None
        optimized_compute_adipocyte_distances = None
        save_distance_colored_image = None

        if config.ENABLE_TUMOR_SEGMENTATION:
            logging.debug("\U0001F52C Configuring tumor detection model...")
            from tumor_detection import (
                configure_tumour_model,
                optimized_segment_tumour_on_thumbnail,
                optimized_compute_adipocyte_distance_metrics,
                optimized_compute_adipocyte_distances,
                save_tumor_csv,
                save_tumor_thumbnail_overlay,
                save_distance_colored_image
            )
            tumor_predictor = configure_tumour_model(paths.TUMOR_MODEL_DIR)

            monitor.log_gpu_status("Tumor model loaded")
            logging.info("\u2705 Tumor detection enabled")
        else:
            logging.debug("\u2139\uFE0F  Tumor detection disabled")

        # ================================================================
        # STEP 3: TUMOR SEGMENTATION (IF ENABLED)
        # ================================================================

        tumor_mask_fullres = None
        tumor_mask_analysis = None
        num_tumors = 0
        tumor_areas = []
        tumor_centroids = []
        thumbnail_size_actual = None

        if config.ENABLE_TUMOR_SEGMENTATION and tumor_predictor:
            logging.info("\U0001F52C Performing tumor segmentation...")

            # Create thumbnail for tumor detection
            tumor_start = time.time()
            thumbnail_size = (1024, 1024)  # Standard size for tumor detection

            if hasattr(image_handler, 'slide') and hasattr(image_handler.slide, 'get_thumbnail'):
                thumbnail = image_handler.slide.get_thumbnail(thumbnail_size)
                thumbnail = thumbnail.convert("RGB")
            else:
                # For non-slide images, read and resize
                full_img = image_handler.read_region((0, 0), 0, (image_handler.width, image_handler.height))
                thumbnail = full_img.copy()
                thumbnail.thumbnail(thumbnail_size, Image.Resampling.BILINEAR)

            thumbnail_np = np.array(thumbnail)
            thumbnail_size_actual = thumbnail.size

            # Perform tumor segmentation
            results = optimized_segment_tumour_on_thumbnail(
                thumbnail_np,
                tumor_predictor,
                (image_handler.height, image_handler.width)
            )

            tumor_mask_thumbnail, tumor_mask_analysis, tumor_mask_fullres, num_tumors, tumor_areas, tumor_centroids = results

            tumor_time = time.time() - tumor_start
            _record_phase("tumor_segmentation", tumor_time)
            logging.info(f"\U0001F52C Tumor segmentation completed in {tumor_time:.1f} seconds")
            logging.info(f"\U0001F3AF Found {num_tumors} tumor regions")

            # Save reviewable tumor thumbnail overlay
            save_tumor_thumbnail_overlay(thumbnail_np, tumor_mask_thumbnail, output_dir, image_name)

            # Save tumor results
            if num_tumors > 0:
                save_tumor_csv(output_dir, image_name, num_tumors, tumor_areas, tumor_centroids, mpp)

            # Free tumor model from GPU — only the adipocyte model should remain
            del tumor_predictor
            tumor_predictor = None
            flush_gpu_memory()
            gc.collect()
            logging.info("\U0001F9F9 Tumor model freed from GPU")
            log_resource_snapshot("after_tumor_segmentation", execution_diagnostics)
        else:
            _record_phase("tumor_segmentation", 0.0)

        # ================================================================
        # STEP 4: GENERATE PROCESSING WINDOWS
        # ================================================================

        _emit_runtime_stage("runtime_window_generation")
        logging.debug("\U0001FA9F Generating processing windows...")
        window_coords = list(generate_sliding_windows(
            image_handler.width,
            image_handler.height,
            final_window_size,
            final_stride
        ))

        total_windows = len(window_coords)
        logging.debug(f"\U0001F4CA Total windows to process: {total_windows}")

        all_window_coords = list(window_coords)
        roi_requested = getattr(args, 'roi_freehand', False) or bool(getattr(args, 'roi_polygon_file', None))
        roi_active = False
        roi_cancelled = False

        # ================================================================
        # STEP 4a: APPLY ROI GUIDANCE (OPTIONAL)
        # ================================================================

        if roi_requested:
            try:
                from roi_guidance import select_freehand_roi, filter_windows_by_roi

                roi_polygon_file = getattr(args, 'roi_polygon_file', None)
                roi_result = select_freehand_roi(
                    image_handler,
                    output_dir=output_dir,
                    roi_max_dim=getattr(args, 'roi_max_dim', 2048),
                    load_polygon_file=roi_polygon_file,
                )

                if roi_result is None:
                    logging.warning("ROI selection canceled; falling back to full image processing.")
                    roi_cancelled = True
                    window_coords = all_window_coords
                else:
                    roi_mask, scale_x, scale_y, _full_res_polygons = roi_result
                    pre_roi = len(window_coords)
                    window_coords = filter_windows_by_roi(
                        window_coords,
                        final_window_size,
                        roi_mask,
                        scale_x,
                        scale_y,
                        min_coverage=getattr(args, 'roi_min_coverage', 0.2),
                    )
                    guided_windows = len(window_coords)
                    reduction = (pre_roi - guided_windows) / pre_roi if pre_roi > 0 else 0
                    logging.info(f"ROI guidance: {guided_windows}/{pre_roi} windows ({reduction:.1%} reduction)")

                    if guided_windows == 0:
                        logging.warning("ROI selection produced no windows; falling back to full image processing.")
                        roi_cancelled = True
                        window_coords = all_window_coords
                    else:
                        roi_active = True

            except Exception as e:
                logging.warning(f"ROI selection failed; falling back to full image processing: {e}")
                roi_cancelled = True
                roi_active = False
                window_coords = all_window_coords

        # ================================================================
        # STEP 4b: APPLY TISSUE GUIDANCE (OPTIONAL — composable with ROI)
        # ================================================================

        if config.ENABLE_TISSUE_GUIDANCE:
            pre_tissue = len(window_coords)
            tissue_guidance_start = time.time()
            logging.info("\U0001F9EC Applying tissue guidance...")
            window_coords, _ = apply_tissue_guidance(
                image_handler, window_coords, final_window_size, output_dir
            )
            guided_windows = len(window_coords)
            reduction = (pre_tissue - guided_windows) / pre_tissue if pre_tissue > 0 else 0
            logging.info(f"\U0001F3AF Tissue guidance: {guided_windows}/{pre_tissue} windows ({reduction:.1%} reduction)")
            if roi_active:
                logging.info(f"\U0001F3AF Combined ROI + tissue guidance: {guided_windows}/{total_windows} total windows")

            # Flush any remaining GPU fragments from tissue detection
            flush_gpu_memory()
            tissue_guidance_time = time.time() - tissue_guidance_start
            _record_phase("tissue_guidance", tissue_guidance_time)
            log_resource_snapshot("after_tissue_guidance", execution_diagnostics)
        else:
            _record_phase("tissue_guidance", 0.0)

        # ================================================================
        # STEP 5: CALCULATE AREA THRESHOLDS
        # ================================================================

        # Additional safety check before area calculations
        if mpp <= 0:
            error_msg = f"Invalid MPP value ({mpp}) would cause division by zero"
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Convert area thresholds from \u00B5m\u00B2 to pixels using the actual MPP
        min_area_pixels = int(config.MIN_ADIPOCYTE_AREA_MICRONS / (mpp ** 2))
        max_area_pixels = int(config.MAX_ADIPOCYTE_AREA_MICRONS / (mpp ** 2))

        logging.debug(f"\U0001F4D0 Area thresholds: {min_area_pixels} - {max_area_pixels} pixels")
        logging.debug(f"\U0001F4D0 Area thresholds: {config.MIN_ADIPOCYTE_AREA_MICRONS} - {config.MAX_ADIPOCYTE_AREA_MICRONS} \u00B5m\u00B2")
        estimated_mask_gb = (image_handler.width * image_handler.height * 4) / (1024 ** 3)
        selected_window_count = len(window_coords)
        logging.info(
            "SLIDE CONFIG image=%s dims=%dx%d mpp=%.4f window=%s stride=%s total_windows=%d selected_windows=%d "
            "mask_estimate_gb=%.2f batch_size=%d io_workers=%d async_io=%s memmap_mask=%s",
            image_name,
            image_handler.width,
            image_handler.height,
            mpp,
            final_window_size,
            final_stride,
            total_windows,
            selected_window_count,
            estimated_mask_gb,
            config.BATCH_INFERENCE_SIZE,
            config.MAX_IO_WORKERS,
            config.ENABLE_ASYNC_IO,
            getattr(config, 'USE_MEMMAP_MASK', False),
        )

        # ================================================================
        # STEP 6: MAIN PROCESSING WITH TISSUE GUIDANCE
        # ================================================================

        _emit_runtime_stage("runtime_window_execution")
        logging.debug("\U0001F52C Starting main processing pipeline...")
        processing_start = time.time()
        execution_path = "generic_batched"

        # Choose processing method based on guidance settings
        # ROI-active takes priority: Steps 4a+4b already filtered window_coords
        # by both ROI and tissue, so pass them directly to process_all_windows.
        if roi_active:
            mode = "ROI + tissue-guided" if config.ENABLE_TISSUE_GUIDANCE else "ROI-guided"
            execution_path = "selected_window_threaded"
            logging.info(f"\U0001F9EC Using {mode} processing ({len(window_coords)} windows)...")
            logging.info("EXECUTION PATH: %s", execution_path)
            results = process_selected_windows_threaded(
                image_handler=image_handler,
                predictor=predictor,
                window_size=final_window_size,
                stride=final_stride,
                min_area_threshold_pixels=min_area_pixels,
                max_area_threshold_pixels=max_area_pixels,
                output_dir=output_dir,
                window_coords=window_coords
            )
        elif config.ENABLE_TISSUE_GUIDANCE:
            execution_path = "selected_window_threaded"
            logging.info(f"\U0001F9E0 Processing tissue-guided windows ({len(window_coords)} windows after guidance)...")
            print(f"\U0001F9E0 Tissue guidance: processing {len(window_coords)} windows (reduced from {total_windows})")
            logging.info("EXECUTION PATH: %s", execution_path)
            results = process_selected_windows_threaded(
                image_handler=image_handler,
                predictor=predictor,
                window_size=final_window_size,
                stride=final_stride,
                min_area_threshold_pixels=min_area_pixels,
                max_area_threshold_pixels=max_area_pixels,
                output_dir=output_dir,
                window_coords=window_coords
            )
        else:
            logging.debug("\U0001F504 Using standard full-image processing...")
            logging.info("EXECUTION PATH: %s", execution_path)
            results = process_all_windows(
                image_handler=image_handler,
                predictor=predictor,
                window_size=final_window_size,
                stride=final_stride,
                min_area_threshold_pixels=min_area_pixels,
                max_area_threshold_pixels=max_area_pixels,
                output_dir=output_dir
            )

        full_mask, mask_areas, adipocyte_ids, postprocessed_windows, processed_window_coords, final_properties, mask_cleanup = results
        execution_diagnostics = get_last_execution_diagnostics()
        execution_diagnostics.setdefault('execution_path', execution_path)

        # Handle backward compatibility if final_properties is not returned
        if final_properties is None:
            final_properties = {}

        processing_time = time.time() - processing_start
        _record_phase("window_execution", execution_diagnostics.get('window_execution_seconds', processing_time))
        _record_phase("finalization", execution_diagnostics.get('finalization_seconds', 0.0))
        logging.info(f"\u23F1\uFE0F Processing completed in {processing_time:.1f} seconds")

        # ================================================================
        # STEP 6b: POST-INFERENCE ROI FILTERING (BOUNDARY CLEANUP)
        # ================================================================

        if roi_active and final_properties:
            try:
                from roi_guidance import filter_detections_by_roi
                pre_filter = len(adipocyte_ids)
                full_mask, mask_areas, adipocyte_ids, final_properties = filter_detections_by_roi(
                    final_properties, full_mask, mask_areas, adipocyte_ids,
                    roi_mask, scale_x, scale_y,
                )
                removed = pre_filter - len(adipocyte_ids)
                if removed > 0:
                    logging.info(f"\U0001F5D1 ROI post-filter: removed {removed} detections outside ROI boundary")
                    execution_diagnostics['final_adipocytes'] = len(adipocyte_ids)
            except Exception as e:
                logging.warning(f"ROI post-filter failed (results unaffected): {e}")

        # ================================================================
        # STEP 7: RESULTS AND VISUALIZATION
        # ================================================================

        total_adipocytes = len(adipocyte_ids)
        logging.info(f"\U0001F389 Detection complete: {total_adipocytes} adipocytes found")

        # ================================================================
        # STEP 7.5: TUMOR DISTANCE COMPUTATION (IF ENABLED)
        # ================================================================

        adipocyte_distances = {}
        adipocyte_closest_tumor_ids = {}
        if config.ENABLE_TUMOR_SEGMENTATION and tumor_mask_analysis is not None and total_adipocytes > 0:
            logging.info("\U0001F4CF Computing adipocyte distances to tumor...")

            # Import is guaranteed to be available here since we checked ENABLE_TUMOR_SEGMENTATION
            adipocyte_distances, adipocyte_closest_tumor_ids, distance_warning = _compute_optional_tumor_distances(
                mask_areas=mask_areas,
                full_mask=full_mask,
                tumor_mask_analysis=tumor_mask_analysis,
                mpp=mpp,
                full_shape=(image_handler.height, image_handler.width),
                image_name=image_name,
                distance_func=optimized_compute_adipocyte_distance_metrics,
                stats_func=calculate_distance_statistics,
                analysis_downsample=160,
                return_closest_tumor_ids=True,
            )
            if distance_warning:
                processing_warnings.append(distance_warning)

        export_time = 0.0
        annotated_image_time = 0.0
        if total_adipocytes > 0:
            # Calculate statistics using the actual MPP
            areas_um2, avg_area, total_area = calculate_area_statistics(mask_areas, mpp)

            logging.info("\U0001F4CA Exporting results...")
            export_start = time.time()

            # CSV export - pass actual slide MPP for correct area conversion
            export_results_csv(
                mask_areas, full_mask, output_dir, image_name, adipocyte_distances,
                precomputed_properties=final_properties, mpp=mpp,
                adipocyte_closest_tumor_ids=adipocyte_closest_tumor_ids
            )

            # QuPath GeoJSON export — only save if enabled in config or via CLI override
            _maybe_export_qupath_geojson(mask_areas, full_mask, output_dir, image_name, final_properties)
            export_time = time.time() - export_start
            _record_phase("export", export_time)

            # ================================================================
            # VISUALIZATION PROFILING: IMAGE CREATION AND ANNOTATION
            # ================================================================

            visualization_start = time.time()
            logging.info("\U0001F3A8 Starting visualization creation with profiling...")

            # PHASE 1: EXTRACT ADIPOCYTE PROPERTIES
            props_extraction_start = time.time()
            logging.info("\U0001F4CA Extracting adipocyte properties for visualization...")

            # Use pre-computed properties from incremental collection (avoids expensive regionprops scan)
            if final_properties is not None and len(final_properties) > 0:
                logging.info("\U0001F9E0 Using pre-computed properties (skipping regionprops scan)")
                adipocyte_props = final_properties
            else:
                # Fallback to regionprops if incremental collection was disabled
                logging.warning("\u26A0\uFE0F No pre-computed properties available, falling back to regionprops scan")
                from skimage.measure import regionprops
                props = regionprops(full_mask)
                adipocyte_props = {}
                for prop in props:
                    if prop.label in mask_areas:
                        adipocyte_props[prop.label] = {
                            'centroid_y': prop.centroid[0],
                            'centroid_x': prop.centroid[1],
                            'area': mask_areas[prop.label]
                        }

            props_extraction_time = time.time() - props_extraction_start
            logging.info(
                f"\U0001F4CA Properties extraction completed: {props_extraction_time:.3f}s "
                f"for {len(adipocyte_props)} adipocytes"
            )

            # PHASE 2: ANNOTATED IMAGE CREATION (DEFAULT/MAIN OUTPUT)
            if _should_run_annotated_image_phase():
                annotated_image_start = time.time()
                logging.info("\U0001F5BC\uFE0F Creating annotated image outputs...")
                log_resource_snapshot("before_annotated_image", execution_diagnostics)

                annotation_outputs = annotate_image_with_adipocytes(
                    image_handler, mask_areas, full_mask, output_dir, tuple(args.stride),
                    tumor_mask_fullres, image_name,
                    tumor_centroids=tumor_centroids,
                    thumbnail_size=thumbnail_size_actual,
                    precomputed_properties=final_properties
                )

                annotated_image_time = time.time() - annotated_image_start
                _record_phase("annotated_image", annotated_image_time)
                if annotation_outputs.get('saved_annotated_image'):
                    logging.info(f"\U0001F5BC\uFE0F Main annotated image created: {annotated_image_time:.3f}s")
                elif annotation_outputs.get('saved_tumor_zone_overlay'):
                    logging.info(
                        f"\U0001F5BC\uFE0F Annotated image phase completed without base TIFF output: "
                        f"{annotated_image_time:.3f}s"
                    )
                else:
                    logging.info(
                        f"\U0001F5BC\uFE0F Annotated image phase completed without saved annotated outputs: "
                        f"{annotated_image_time:.3f}s"
                    )
                log_resource_snapshot("after_annotated_image", execution_diagnostics)
            else:
                annotated_image_time = 0.0
                _record_phase("annotated_image", annotated_image_time)
                logging.info(
                    "\U0001F5BC\uFE0F Annotated image phase skipped "
                    "(SAVE_ANNOTATED_IMAGE = False and no tumor-zone overlay output is enabled)"
                )

            # PHASE 3: DISTANCE-COLORED IMAGE (OPTIONAL)
            distance_image_time = 0.0
            if config.ENABLE_TUMOR_SEGMENTATION and config.SAVE_DISTANCE_COLORED_IMAGE and adipocyte_distances:
                distance_image_start = time.time()
                logging.info("\U0001F308 Creating distance-colored visualization (optional)...")

                # Import is guaranteed to be available here since we checked ENABLE_TUMOR_SEGMENTATION
                save_distance_colored_image(
                    image_handler, mask_areas, adipocyte_distances, output_dir, image_name,
                    full_mask, adipocyte_props, tumor_mask_fullres
                )

                distance_image_time = time.time() - distance_image_start
                logging.info(f"\U0001F308 Distance-colored image created: {distance_image_time:.3f}s")
            else:
                if config.ENABLE_TUMOR_SEGMENTATION and not config.SAVE_DISTANCE_COLORED_IMAGE:
                    logging.info("\U0001F308 Distance-colored image skipped (SAVE_DISTANCE_COLORED_IMAGE = False)")
                elif not config.ENABLE_TUMOR_SEGMENTATION:
                    logging.info("\U0001F308 Distance-colored image skipped (tumor detection disabled)")
                else:
                    logging.info("\U0001F308 Distance-colored image skipped (no adipocyte distances computed)")

            # PHASE 4: PLAIN (UNANNOTATED) IMAGE (OPTIONAL — DEMO)
            plain_image_time = 0.0
            if getattr(config, 'SAVE_UNPROCESSED_IMAGE', False):
                plain_image_start = time.time()
                logging.info("\U0001F5BC\uFE0F Creating plain (unannotated) slide image (demo)...")
                save_plain_image(image_handler, output_dir, image_name)
                plain_image_time = time.time() - plain_image_start
                logging.info(f"\U0001F5BC\uFE0F Plain image created: {plain_image_time:.3f}s")
            else:
                logging.info("\U0001F5BC\uFE0F Plain image skipped (SAVE_UNPROCESSED_IMAGE = False)")

            # PHASE 5: SOBEL + INVERTED IMAGE (OPTIONAL — DEMO)
            sobel_image_time = 0.0
            if getattr(config, 'SAVE_POSTPROCESSED_IMAGE', False):
                sobel_image_start = time.time()
                logging.info("\U0001F9EA Creating Sobel+inverted slide image (demo)...")
                save_sobel_inverted_image(image_handler, output_dir, image_name)
                sobel_image_time = time.time() - sobel_image_start
                logging.info(f"\U0001F9EA Sobel+inverted image created: {sobel_image_time:.3f}s")
            else:
                logging.info("\U0001F9EA Sobel+inverted image skipped (SAVE_POSTPROCESSED_IMAGE = False)")

            total_visualization_time = time.time() - visualization_start
            _record_phase("visualization_total", total_visualization_time)

            # ================================================================
            # VISUALIZATION PROFILING SUMMARY
            # ================================================================
            _log_visualization_summary(
                props_extraction_time=props_extraction_time,
                annotated_image_time=annotated_image_time,
                distance_image_time=distance_image_time,
                plain_image_time=plain_image_time,
                sobel_image_time=sobel_image_time,
                total_visualization_time=total_visualization_time,
                adipocyte_count=len(adipocyte_props),
                image_width=image_handler.width,
                image_height=image_handler.height,
            )
        else:
            _record_phase("export", 0.0)
            _record_phase("annotated_image", 0.0)
            _record_phase("visualization_total", 0.0)

        # Final statistics
        total_time = time.time() - start_time
        _record_phase("total_slide", total_time)
        logging.info(f"\u23F1\uFE0F Total execution time: {total_time:.1f} seconds")
        if total_time > 0 and total_windows > 0:
            logging.info(f"\u26A1 Processing rate: {total_windows/total_time:.1f} windows/second")
        else:
            logging.info("\u26A1 Processing rate: N/A (instant completion or no windows)")

        logging.info("PERFORMANCE DIAGNOSIS [%s]", image_name)
        logging.info(
            "  execution_path=%s | total_windows=%d | selected_windows=%d | processed_windows=%s | ready_for_inference=%s",
            execution_diagnostics.get('execution_path', 'unknown'),
            total_windows,
            len(window_coords),
            execution_diagnostics.get('window_count_processed', 'n/a'),
            execution_diagnostics.get('window_count_ready_for_inference', 'n/a'),
        )
        logging.info(
            "  window_execution=%.3fs | finalization=%.3fs | export=%.3fs | visualization=%.3fs | total=%.3fs",
            phase_timings.get('window_execution', 0.0),
            phase_timings.get('finalization', 0.0),
            phase_timings.get('export', 0.0),
            phase_timings.get('visualization_total', 0.0),
            total_time,
        )
        logging.info(
            "  windows_per_sec=%.2f | peak_ram=%.1f%% | peak_gpu_reserved=%.1fGB",
            execution_diagnostics.get('mean_windows_per_second', (len(window_coords) / processing_time) if processing_time > 0 else 0.0),
            execution_diagnostics.get('peak_ram_percent', 0.0),
            execution_diagnostics.get('peak_gpu_reserved_gb', 0.0),
        )
        if execution_diagnostics.get('execution_path') == 'selected_window_threaded':
            logging.info(
                "  threaded_samples_every=%s windows | async_io=%s | workers=%s",
                execution_diagnostics.get('window_sample_interval', 'n/a'),
                execution_diagnostics.get('async_io_enabled', 'n/a'),
                execution_diagnostics.get('max_workers', 'n/a'),
            )
        else:
            logging.info(
                "  batches=%s | avg_prep=%.4fs | avg_gpu_batch=%.4fs",
                execution_diagnostics.get('batches_completed', 'n/a'),
                execution_diagnostics.get('mean_prep_seconds', 0.0),
                execution_diagnostics.get('mean_gpu_batch_seconds', 0.0),
            )
        logging.info(
            "  adipocytes=%d | initial_labels=%s | mask_backend=%s | mask_estimate_gb=%.2f",
            total_adipocytes,
            execution_diagnostics.get('initial_unique_labels', 'n/a'),
            execution_diagnostics.get('mask_backend', 'unknown'),
            estimated_mask_gb,
        )

        # Success message for single image
        print(f"\u2705 {image_name}: {total_adipocytes} adipocytes detected in {total_time:.1f}s")

        # Memmap cleanup after all outputs/visualizations are complete
        _cleanup_memmap()

        return {
            'image_name': image_name,
            'image_path': image_path,
            'output_dir': output_dir,
            'total_adipocytes': total_adipocytes,
            'total_time': total_time,
            'total_windows': total_windows,
            'num_tumors': num_tumors,
            'median_size_microns': 0,  # Will be calculated from CSV if needed
            'average_size_microns': 0,  # Will be calculated from CSV if needed
            'warning_message': " ".join(processing_warnings)
        }

    finally:
        # Ensure memmap cleanup happens even on failure
        _cleanup_memmap()

        # Restore original output_dir and close any open image handles
        args.output_dir = temp_output_dir
        if image_handler is not None:
            try:
                image_handler.close()
            except Exception as e:
                logging.warning(f"Could not close image handler for {image_path}: {e}")


# ================================================================
# MAIN PROCESSING PIPELINE
# ================================================================

def main():
    """
    Main function for AdiFind WSI analysis.
    Orchestrates the complete pipeline from image loading to result export.
    Supports both single image and batch processing of directories with resume capability.
    """
    # Parse command line arguments
    args = parse_arguments()

    if getattr(args, 'gpu_probe_only', False):
        _setup_probe_logging(bool(getattr(args, 'verbose', False)))
        update_config_from_args(args)
        return _run_gpu_probe(args)

    # Handle special commands first
    if args.list_resumable:
        _load_batch_processing_dependencies()
        print("\U0001F50D Searching for resumable batch jobs...")
        current_dir = os.getcwd()
        resumable_batches = find_resumable_batches(current_dir)

        if not resumable_batches:
            print("\u274C No resumable batch jobs found in current directory")
            return

        print(f"\U0001F4CB Found {len(resumable_batches)} resumable batch job(s):")
        print("-" * 80)
        for i, batch in enumerate(resumable_batches, 1):
            print(f"{i}. Batch ID: {batch['batch_id']}")
            print(f"   Started: {batch['start_time']}")
            print(f"   Progress: {batch['processed_images']}/{batch['total_images']} images processed")
            print(f"   Remaining: {batch['remaining_images']} images")
            print(f"   Output Dir: {batch['base_output_dir']}")
            print(f"   Resume Command: --resume_batch {batch['state_file']}")
            print("-" * 80)
        return

    # Validate that image_path is provided for normal operations
    if not args.image_path and not args.resume_batch:
        print("\u274C Error: image_path is required unless using --list_resumable or --resume_batch")
        return

    # Handle resume batch processing
    startup_messages = []

    if args.resume_batch:
        _load_batch_processing_dependencies()
        if not os.path.exists(args.resume_batch):
            print(f"Resume state file not found: {args.resume_batch}")
            return

        startup_messages.append(f"Resuming batch from: {args.resume_batch}")
        # Load batch state to get image files and configuration
        with open(args.resume_batch, 'r', encoding='utf-8') as f:
            batch_state = json.load(f)

        startup_messages.append(
            f"DEBUG - Loaded JSON with {batch_state.get('total_images', 'UNKNOWN')} total images"
        )
        startup_messages.append(
            f"DEBUG - JSON has {len(batch_state.get('image_results', []))} image_results entries"
        )

        # Count statuses in loaded JSON
        status_counts = {}
        for result in batch_state.get('image_results', []):
            status = result.get('processing_status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        startup_messages.append(f"DEBUG - Status distribution in loaded JSON: {status_counts}")

        # Get image files that still need processing
        # Only include truly pending images (not failed, unless retry_failed is enabled)
        retry_failed = args.resume_failed or config.RETRY_FAILED_IMAGES
        if retry_failed:
            # Retry mode: include both pending and failed
            pending_images = [
                result['image_path'] for result in batch_state['image_results']
                if result['processing_status'] in ['pending', 'failed']
            ]
            startup_messages.append(
                f"DEBUG - Retry mode: Including {len(pending_images)} images (pending + failed)"
            )
        else:
            # Safe mode: only include pending images, skip failed
            pending_images = [
                result['image_path'] for result in batch_state['image_results']
                if result['processing_status'] == 'pending'
            ]
            startup_messages.append(
                f"DEBUG - Safe mode: Including {len(pending_images)} pending images only"
            )

        startup_messages.append(f"DEBUG - Found {len(pending_images)} images to process")

        if not pending_images:
            print("All images in this batch have already been processed!")
            return

        # Override args with batch configuration
        base_output_dir = batch_state['base_output_dir']
        image_files = pending_images

        # Update args from saved state
        saved_args = batch_state.get('args_snapshot', {})
        _apply_saved_batch_args(args, saved_args)

        startup_messages.append(
            f"DEBUG - About to create BatchProcessor with resume_batch={args.resume_batch}"
        )

        # Initialize batch processor with resume
        batch_processor = BatchProcessor(base_output_dir, image_files, args, args.resume_batch)

        # Setup logging for resumed batch processing
        setup_logging(base_output_dir, console=False)

    else:
        # Normal processing - get list of images to process
        image_files = get_image_files(args.image_path)

        # Process by on-disk size (smallest-first) if enabled.
        if SORT_BY_FILE_SIZE:
            try:
                image_files = sorted(image_files, key=lambda p: os.stat(p).st_size, reverse=not SIZE_SORT_ASCENDING)
            except OSError:
                pass

            # Optionally refine ordering by resolution (pixel area) using OpenSlide headers
            image_files = maybe_sort_by_resolution(
                image_files,
                cache_path=RESOLUTION_CACHE_PATH,
                ascending=RES_SORT_ASCENDING
            )

        # Check if we're processing multiple images
        is_batch_processing = len(image_files) > 1

        if is_batch_processing:
            _load_batch_processing_dependencies()
            startup_messages.append(f"\U0001F4C1 Batch processing mode: {len(image_files)} images found")

            # Generate base output directory for batch processing
            if args.output_dir is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_output_dir = f"adifind_batch_results_{timestamp}"
            else:
                # Add date suffix to user-specified output directory
                date_suffix = time.strftime("%Y%m%d_%H%M%S")
                base_output_dir = f"{args.output_dir}_{date_suffix}"

            # Create base output directory
            os.makedirs(base_output_dir, exist_ok=True)

            # Initialize batch processor
            batch_processor = BatchProcessor(base_output_dir, image_files, args)

            # Setup logging for batch processing
            setup_logging(base_output_dir, console=False)
        else:
            batch_processor = None

            # Generate output directory if not specified for single image
            if args.output_dir is None:
                image_name = Path(args.image_path).stem
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                args.output_dir = f"adifind_results_{image_name}_{timestamp}"
            else:
                # Add date suffix to user-specified output directory
                date_suffix = time.strftime("%Y%m%d_%H%M%S")
                args.output_dir = f"{args.output_dir}_{date_suffix}"

            # Create dedicated folder for the image within the output directory
            image_name = Path(args.image_path).stem
            date_suffix = time.strftime("%Y%m%d_%H%M%S")
            single_image_dir = os.path.join(args.output_dir, f"{image_name}_{date_suffix}")

            # Create output directory structure
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(single_image_dir, exist_ok=True)

            # Update args to point to the image-specific directory
            args.output_dir = single_image_dir

            # Setup logging
            setup_logging(args.output_dir, console=False)

    try:
        # Validate inputs
        validate_inputs(args)

        # Update configuration from arguments
        update_config_from_args(args)

        _emit_runtime_stage("runtime_startup")
        _load_runtime_dependencies()
        _log_gpu_runtime_stack(args)

        _print_startup_banner()

        for message in startup_messages:
            print(message)

        enable_console_logging(verbose=config.VERBOSE_LOGGING)

        # Override RETRY_FAILED_IMAGES if --resume_failed flag is provided
        if args.resume_failed:
            config.RETRY_FAILED_IMAGES = True
            print("\U0001F504 --resume_failed enabled: Will retry previously failed images")
        else:
            print(f"\U0001F6E1\uFE0F  Safe mode: RETRY_FAILED_IMAGES = {config.RETRY_FAILED_IMAGES}")

        # Import tumor detection functions if enabled (after config update)
        if config.ENABLE_TUMOR_SEGMENTATION:
            from tumor_detection import (
                configure_tumour_model,
                optimized_segment_tumour_on_thumbnail,
                optimized_compute_adipocyte_distance_metrics,
                optimized_compute_adipocyte_distances,
                save_tumor_csv,
                save_tumor_thumbnail_overlay,
                save_distance_colored_image
            )

        # Set GPU device
        if config.USE_GPU_INFERENCE:
            setup_gpu_device(args.gpu_id)
        else:
            logging.info("\u00F0\u0178\u2013\u00A5\u00EF\u00B8\u008F GPU inference disabled; skipping CUDA device setup")

        # Initialize system monitoring
        monitor.log_system_info()
        logging.debug("\U0001F5A5\uFE0F System monitoring initialized")
        logging.debug("\uFFFD Memory management initialized")

        # Process all images
        batch_start_time = time.time()

        # Determine which images to process
        if batch_processor:
            # Get pending images from batch processor with retry_failed setting
            retry_failed_effective = args.resume_failed or config.RETRY_FAILED_IMAGES
            pending_images = batch_processor.get_pending_images(retry_failed=retry_failed_effective)
            images_to_process = [result.image_path for result in pending_images]

            # Maintain size-based order on resume
            if SORT_BY_FILE_SIZE:
                try:
                    images_to_process.sort(key=lambda p: os.stat(p).st_size, reverse=not SIZE_SORT_ASCENDING)
                except OSError:
                    pass

            # Optionally refine resume ordering by resolution (pixel area)
            images_to_process = maybe_sort_by_resolution(
                images_to_process,
                cache_path=RESOLUTION_CACHE_PATH,
                ascending=RES_SORT_ASCENDING
            )

            total_images = batch_processor.batch_state.total_images
            print(f"\U0001F4CA Processing {len(images_to_process)} remaining images out of {total_images} total")
        else:
            # Single image processing
            images_to_process = image_files
            total_images = len(image_files)

        # Handle dry run mode
        if args.dry_run:
            print(f"\n\U0001F50D DRY RUN MODE - No actual processing will occur")
            print(f"\U0001F4CA Would process {len(images_to_process)} images:")

            for i, image_path in enumerate(images_to_process, 1):
                image_name = Path(image_path).name
                if batch_processor:
                    # Find the result object for this image to show status
                    image_result = next(
                        (r for r in batch_processor.batch_state.image_results if r.image_path == image_path),
                        None
                    )
                    if image_result:
                        status = image_result.processing_status
                        error_msg = image_result.error_message
                        print(f"  {i:3d}. {image_name} (status: {status})" + (f" - Error: {error_msg}" if error_msg else ""))
                    else:
                        print(f"  {i:3d}. {image_name} (status: unknown)")
                else:
                    print(f"  {i:3d}. {image_name}")

            print(f"\n\u2705 Dry run complete. {len(images_to_process)} images would be processed.")
            return

        for i, image_path in enumerate(images_to_process, 1):
            try:
                # Update batch processor if in batch mode
                if batch_processor:
                    batch_processor.mark_image_started(image_path)
                    print(f"\n\U0001F4F8 Processing image {i}/{len(images_to_process)}: {Path(image_path).name}")
                else:
                    print(f"\n\U0001F4F8 Processing: {Path(image_path).name}")

                # Determine output directory
                if batch_processor:
                    # Find the result object for this image
                    image_result = next(
                        (r for r in batch_processor.batch_state.image_results if r.image_path == image_path),
                        None
                    )
                    if image_result:
                        single_output_dir = image_result.output_dir
                    else:
                        # Fallback
                        image_name = Path(image_path).stem
                        single_output_dir = os.path.join(batch_processor.base_output_dir, image_name)
                else:
                    single_output_dir = args.output_dir

                # Process single image
                result = process_single_image(image_path, args, single_output_dir)

                # Update batch processor or store result
                if batch_processor:
                    batch_processor.mark_image_completed(image_path, result)
                else:
                    # For single image, keep compatibility format
                    batch_results = [result]

                # GPU cleanup between images
                flush_gpu_memory()

            except Exception as e:
                friendly_message = _friendly_processing_error(e)
                error_msg = f"\u274C Error processing {Path(image_path).name}: {friendly_message}"
                logging.exception("Error processing %s: %s", Path(image_path).name, friendly_message)
                print(error_msg)

                # Update batch processor or handle error
                if batch_processor:
                    batch_processor.mark_image_failed(image_path, friendly_message)
                    continue
                else:
                    raise

        # ================================================================
        # BATCH PROCESSING SUMMARY
        # ================================================================

        total_batch_time = time.time() - batch_start_time

        if batch_processor:
            # Use batch processor for comprehensive summary
            print("\n" + "=" * 60)
            print("\U0001F4CA BATCH PROCESSING SUMMARY")
            print("=" * 60)

            state = batch_processor.batch_state
            print(f"\u2705 Successfully processed: {state.successful_images}/{state.total_images} images")
            if state.failed_images > 0:
                print(f"\u274C Failed: {state.failed_images} images")
            print(f"\U0001F52C Total adipocytes detected: {state.total_adipocytes}")
            if config.ENABLE_TUMOR_SEGMENTATION:
                print(f"\U0001F3AF Total tumor regions found: {state.total_tumors}")
            print(f"\u23F1\uFE0F Total processing time: {total_batch_time:.1f} seconds")
            if state.successful_images > 0:
                print(f"\u26A1 Average time per image: {state.total_processing_time/state.successful_images:.1f} seconds")
            print(f"\U0001F4C2 Results saved to: {batch_processor.base_output_dir}")

            # Create final summary
            summary_files = batch_processor.create_final_summary()

            # Show resume info if there are remaining images
            remaining_images = len([r for r in state.image_results if r.processing_status == 'failed'])
            if remaining_images > 0:
                print(f"\n\U0001F4A1 {remaining_images} images failed and can be retried.")
                print(f"   {batch_processor.get_resume_info()}")

        elif not batch_processor and 'batch_results' in locals() and batch_results:
            # Single image processing
            result = batch_results[0]
            print("\n" + "=" * 60)
            print(f"\U0001F389 SUCCESS: {result['total_adipocytes']} adipocytes detected!")
            print(f"\U0001F4C2 Results saved to: {result['output_dir']}")
            print(f"\u23F1\uFE0F Completed in {result['total_time']:.1f} seconds")
            print("=" * 60)

        # Memory summary
        final_memory = memory_manager.get_system_memory_usage()
        logging.info(f"\U0001F4BE Final memory usage: {final_memory:.1f} GB")
        logging.info("\U0001F3C1 AdiFind analysis completed successfully")
        return 0

    except KeyboardInterrupt:
        logging.info("\U0001F6D1 Processing interrupted by user")
        print("\n\U0001F6D1 Processing interrupted by user")
        return 130

    except Exception as e:
        logging.error(f"\u274C Error in main processing: {e}")
        print(f"\n\u274C Error: {e}")
        raise

    finally:
        # Ensure cleanup happens
        try:
            flush_gpu_memory()
        except Exception:
            pass


# Import legacy compatibility functions
from legacy_compatibility import adifind_main, run_adipocyte_detection

def _load_resolution_cache(cache_path):
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_resolution_cache(cache, cache_path):
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _get_slide_pixel_area(path):
    """Return level-0 pixel area (W*H) for a slide, or None on failure."""
    openslide_module = _get_openslide_module()
    if openslide_module is None:
        return None
    try:
        slide = openslide_module.OpenSlide(path)
        w, h = slide.dimensions  # header-only
        slide.close()
        return int(w) * int(h)
    except Exception:
        return None


def maybe_sort_by_resolution(paths, cache_path, ascending=True, verbose=False):
    """Return a new list sorted by pixel area (smallest-first if ascending).

    Uses a JSON cache to avoid re-opening slides repeatedly. Only header metadata
    is read; no pixel data is decoded. If OpenSlide is unavailable, this is a no-op.
    """
    if not paths:
        return paths
    try:
        from config import SORT_BY_RESOLUTION
    except Exception:
        SORT_BY_RESOLUTION = False
    if not SORT_BY_RESOLUTION or _get_openslide_module() is None:
        return paths
    cache = _load_resolution_cache(cache_path)
    updated = False
    areas = {}
    for p in paths:
        sp = str(p)
        a = cache.get(sp)
        if a is None:
            a = _get_slide_pixel_area(sp)
            cache[sp] = a
            updated = True
        areas[sp] = a
    if updated:
        _save_resolution_cache(cache, cache_path)

    def _key(spath):
        a = areas.get(str(spath))
        # Place unknown/None at the end, keep stable order among them
        return (0, a if ascending else -a) if isinstance(a, (int, float)) else (1, 0)

    # Return a new list to avoid side-effects
    return sorted(paths, key=_key)
# --- End helpers ---


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    raise SystemExit(main() or 0)


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'main',
    'adifind_main',
    'run_adipocyte_detection'
]
