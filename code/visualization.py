#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization and Output Module (auto-scaled, tumor-safe overlay)
================================================================

Handles image visualization, annotation, QuPath export, and results output.
Adds:
- Automatic scale capping to keep annotated images under a pixel budget
- Tumor overlay is downsampled safely; micron-boundaries are skipped if too large
- Clear warnings that specify *which* mask is implicated (label vs tumour)
- Robust TIFF saving with tifffile (BigTIFF)
"""

import os
import csv
import json
import logging
import numpy as np
import typing as t
import cv2
import tifffile
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import regionprops, regionprops_table

# Import from other modules
from config import config
from tumor_colors import build_tumor_instance_overlay, get_tumor_instance_alpha

CSV_BASE_FIELDS = [
    'Adipocyte_ID',
    'Area_Pixels',
    'Area_Microns_Squared',
    'Centroid_X',
    'Centroid_Y',
    'Grid_ID',
    'MPP_Used',
]
CSV_DISTANCE_FIELDS = [
    'Closest_Tumour_ID',
    'Distance_To_Closest_Tumour',
    'Distance_Bin',
]
CSV_PRECOMPUTED_EXTENDED_FIELDS = [
    'bbox_min_x',
    'bbox_min_y',
    'bbox_max_x',
    'bbox_max_y',
]
CSV_STANDARD_EXTENDED_FIELDS = [
    *CSV_PRECOMPUTED_EXTENDED_FIELDS,
    'eccentricity',
    'solidity',
    'extent',
    'perimeter',
    'equivalent_diameter',
]


# ================================================================
# HELPERS (auto-scaling & preview capping)
# ================================================================

def _compute_safe_scale(width: int, height: int, requested_scale: float) -> float:
    """Downscale automatically so target pixel count never exceeds MAX_ANNOTATED_PIXELS."""
    max_px = int(getattr(config, 'MAX_ANNOTATED_PIXELS', 250_000_000))  # ~250 MP default
    base_px = max(1, int(width) * int(height))
    if requested_scale <= 0:
        return 0.0
    # pixels at scale s = base_px * s^2  ->  s_safe = min(requested_scale, sqrt(max_px / base_px))
    s_safe = min(float(requested_scale), (max_px / float(base_px)) ** 0.5)
    if s_safe < requested_scale:
        logging.warning(
            f"\u26A0\uFE0F Annotated image scale reduced from {requested_scale:.5f} to {s_safe:.5f} "
            f"to respect MAX_ANNOTATED_PIXELS={max_px:,} (base={base_px:,} px). "
            f"This affects all overlays (adipocyte label mask + tumour mask)."
        )
    return s_safe


def _cap_preview_by_pixels(image_rgb: np.ndarray) -> np.ndarray:
    """Cap preview image by a pixel budget to keep OpenCV writers safe and fast."""
    if image_rgb is None:
        return image_rgb
    max_preview_px = int(getattr(config, 'ANNOTATED_PREVIEW_MAX_PIXELS', 30_000_000))  # ~30 MP
    h, w = image_rgb.shape[:2]
    cur_px = h * w
    if cur_px <= max_preview_px:
        return image_rgb
    scale = (max_preview_px / float(cur_px)) ** 0.5
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ================================================================
# ANNOTATION AND VISUALIZATION FUNCTIONS
# ================================================================

def _get_annotation_output_settings():
    """Resolve which annotated-image-derived outputs are enabled."""
    save_annotated_image = bool(getattr(config, 'SAVE_ANNOTATED_IMAGE', True))
    request_tumor_zone_overlay = bool(getattr(config, 'SAVE_TUMOR_ZONE_OVERLAY_IMAGE', False))
    save_tumor_zone_overlay = request_tumor_zone_overlay and bool(
        getattr(config, 'ENABLE_TUMOR_SEGMENTATION', False)
    )
    return {
        'save_annotated_image': save_annotated_image,
        'request_tumor_zone_overlay': request_tumor_zone_overlay,
        'save_tumor_zone_overlay': save_tumor_zone_overlay,
        'should_generate': save_annotated_image or save_tumor_zone_overlay,
    }


def annotate_image_with_adipocytes(
    image_handler,
    mask_areas,
    full_mask,
    output_dir,
    stride,
    tumor_mask_fullres=None,
    image_name=None,
    tumor_centroids=None,
    thumbnail_size=None,
    precomputed_properties=None
):
    """
    Create optimized annotated image with adipocyte overlays.

    Args:
        image_handler: ImageHandler object for slide access
        mask_areas: Dict mapping adipocyte IDs to their areas
        full_mask: Full-resolution label mask with adipocyte IDs
        output_dir: Directory for saving annotated image
        stride: (stride_x, stride_y) tuple (kept for API compatibility)
        tumor_mask_fullres: Optional tumour mask for overlay
        image_name: Optional base name for outputs
        tumor_centroids: Optional list of tumor centroids (from thumbnail)
        thumbnail_size: Optional tuple of (width, height) of the thumbnail used for detection
        precomputed_properties: Optional dict of pre-computed adipocyte properties
                               {id: {'centroid_x': float, 'centroid_y': float, 'area': int, 'bbox': tuple}}
                               If provided, avoids regionprops() scan on gigapixel mask
    """
    annotation_outputs = {
        'generated': False,
        'saved_annotated_image': False,
        'saved_tumor_zone_overlay': False,
    }
    output_settings = _get_annotation_output_settings()
    if not output_settings['should_generate']:
        logging.info(
            "Annotated image generation skipped "
            "(SAVE_ANNOTATED_IMAGE = False and no tumor-zone overlay output is enabled)"
        )
        return annotation_outputs

    logging.info("\uD83D\uDCF8 Creating annotated image...")

    try:
        from system_utils import get_mpp

        # Extract adipocyte properties - use precomputed if available
        if precomputed_properties is not None and len(precomputed_properties) > 0:
            logging.info("\uD83E\uDDE0 Using pre-computed adipocyte properties (memory-efficient)")
            adipocyte_props = {}
            adipocyte_ids = []
            for aid, props in precomputed_properties.items():
                adipocyte_props[aid] = {
                    'centroid_y': props['centroid_y'],
                    'centroid_x': props['centroid_x'],
                    'area': props['area']
                }
                adipocyte_ids.append(aid)
        else:
            # Fall back to regionprops for standard path
            props = regionprops(full_mask)
            adipocyte_props = {}
            adipocyte_ids = []
            for prop in props:
                if prop.label in mask_areas:
                    adipocyte_props[prop.label] = {
                        'centroid_y': prop.centroid[0],
                        'centroid_x': prop.centroid[1],
                        'area': mask_areas[prop.label]
                    }
                    adipocyte_ids.append(prop.label)

        # Get image dimensions and set parameters
        width, height = image_handler.width, image_handler.height
        requested_scale = float(getattr(config, 'ANNOTATED_IMAGE_SCALE', 1.0))
        mpp = get_mpp(image_handler)
        desired_level = getattr(config, 'DESIRED_RESOLUTION_LEVEL', 0)

        # Auto-cap the scale to avoid gigantic overlays
        scaling_factor = _compute_safe_scale(width, height, requested_scale)

        # Optionally skip if the safe scale is too tiny to be useful
        min_scale = float(getattr(config, 'MIN_ANNOTATED_SCALE', 0.01))
        if scaling_factor < min_scale:
            logging.warning(
                f"\uD83D\uDED1 Annotated image skipped: safe scale {scaling_factor:.5f} < MIN_ANNOTATED_SCALE={min_scale}. "
                f"Reduce ANNOTATED_IMAGE_SCALE or increase MAX_ANNOTATED_PIXELS if you really need it."
            )
            return annotation_outputs

        # Create optimized annotated image
        annotated_image = create_optimized_annotated_image(
            image_handler, full_mask, mask_areas, adipocyte_props,
            adipocyte_ids, width, height, scaling_factor, mpp,
            tumor_mask_fullres, desired_level,
            tumor_centroids, thumbnail_size
        )
        annotation_outputs['generated'] = True

        # --- Save TIFF robustly with BigTIFF ---
        annotated_image = np.ascontiguousarray(annotated_image, dtype=np.uint8)
        if output_settings['save_annotated_image']:
            if image_name:
                output_path = os.path.join(output_dir, f"{image_name}_adifind_annotated.tiff")
            else:
                output_path = os.path.join(output_dir, "annotated_image.tiff")

            try:
                tifffile.imwrite(output_path, annotated_image, bigtiff=True)
                annotation_outputs['saved_annotated_image'] = True
                logging.info(f"\uD83D\uDCC1 Annotated image saved (tifffile BigTIFF): {output_path}")
            except Exception as e:
                logging.exception(f"Failed to save annotated image via tifffile: {e}")
                raise
        else:
            logging.info("Annotated image save skipped (SAVE_ANNOTATED_IMAGE = False)")

        # --- Optional additional tumor-zone overlay TIFF ---
        try:
            if output_settings['request_tumor_zone_overlay']:
                if not output_settings['save_tumor_zone_overlay']:
                    logging.info("Tumor-zone overlay export skipped (ENABLE_TUMOR_SEGMENTATION = False)")
                elif tumor_mask_fullres is None:
                    logging.info("Tumor-zone overlay export skipped (no tumor mask available)")
                elif not mpp or mpp <= 0:
                    logging.warning("Tumor-zone overlay export skipped (invalid mpp)")
                else:
                    zoned_image, zone_stats = add_tumor_distance_zones_overlay(
                        np.copy(annotated_image), tumor_mask_fullres, scaling_factor, mpp
                    )
                    if zoned_image is not None:
                        zone_path = os.path.join(
                            output_dir,
                            f"{image_name}_adifind_annotated_tumor_zones.tiff"
                            if image_name else "annotated_tumor_zones.tiff"
                        )
                        zoned_image = np.ascontiguousarray(zoned_image, dtype=np.uint8)
                        tifffile.imwrite(zone_path, zoned_image, bigtiff=True)
                        annotation_outputs['saved_tumor_zone_overlay'] = True
                        logging.info(f"\uD83D\uDCC1 Tumor-zone overlay image saved: {zone_path}")
                        if zone_stats:
                            logging.info(
                                "Tumor-zone thresholds: near<=%.1f\u00b5m (%dpx), intermediate<=%.1f\u00b5m (%dpx), "
                                "distal>%.1f\u00b5m | pixels near=%d intermediate=%d distal=%d",
                                zone_stats.get('near_um', 0.0),
                                zone_stats.get('near_px', 0),
                                zone_stats.get('intermediate_um', 0.0),
                                zone_stats.get('intermediate_px', 0),
                                zone_stats.get('intermediate_um', 0.0),
                                zone_stats.get('near_count', 0),
                                zone_stats.get('intermediate_count', 0),
                                zone_stats.get('distal_count', 0),
                            )
                    else:
                        logging.info("No tumor detected — tumor-zone overlay not generated")
        except Exception as e:
            logging.exception(f"Failed to save tumor-zone overlay image: {e}")

        # --- Optional small preview (PNG) ---
        try:
            if bool(getattr(config, 'SAVE_ANNOTATED_PREVIEW', False)):
                preview = _cap_preview_by_pixels(annotated_image)
                preview_path = os.path.join(
                    output_dir,
                    f"{image_name}_annotated_preview.png" if image_name else "annotated_preview.png"
                )
                cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                logging.info(f"\uD83D\uDDBC\uFE0F Preview saved: {preview_path} ({preview.shape[1]}x{preview.shape[0]})")
        except cv2.error as e:
            logging.exception(f"OpenCV preview save error (ignored): {e}")

    except cv2.error as e:
        logging.exception(f"OpenCV error during annotation: {e}")
    except Exception as e:
        logging.error(f"\u274C Error creating annotated image: {e}")

    return annotation_outputs


def create_optimized_annotated_image(
    image_handler,
    full_mask,
    mask_areas,
    adipocyte_props,
    adipocyte_ids,
    width,
    height,
    scaling_factor,
    mpp,
    tumour_mask_fullres,
    desired_level,
    tumor_centroids=None,
    thumbnail_size=None
):
    """\uD83D\uDE80 Create annotated image with optimized rendering."""

    # Hard clamp scale to pixel budget to avoid huge allocations
    max_px = int(getattr(config, 'MAX_ANNOTATED_PIXELS', 250_000_000))
    target_width = int(width * scaling_factor)
    target_height = int(height * scaling_factor)
    target_pixels = max(1, target_width * target_height)
    if target_pixels > max_px and target_pixels > 0:
        clamp_factor = (max_px / float(target_pixels)) ** 0.5
        scaling_factor *= clamp_factor
        target_width = max(1, int(width * scaling_factor))
        target_height = max(1, int(height * scaling_factor))
        logging.warning(
            f"\u26A0\uFE0F Annotated scale clamped to {scaling_factor:.5f} so output stays within "
            f"MAX_ANNOTATED_PIXELS={max_px:,} (target={target_width}x{target_height})."
        )

    # === OPTIMIZED IMAGE READING ===
    full_image = read_optimal_image(
        image_handler, width, height, scaling_factor, desired_level
    )

    # === OPTIMIZED MASK OVERLAY (adipocyte label mask) ===
    mask_overlay = create_optimized_mask_overlay(
        full_mask, mask_areas, adipocyte_ids, target_width, target_height, scaling_factor
    )

    # Ensure dtype/shape parity & contiguity before blending
    mask_overlay = np.ascontiguousarray(mask_overlay, dtype=np.uint8)
    full_image = np.ascontiguousarray(full_image, dtype=np.uint8)
    if mask_overlay.shape != full_image.shape:
        raise ValueError(f"Shape mismatch for addWeighted: mask {mask_overlay.shape}, image {full_image.shape}")

    # Blend images efficiently
    alpha = 0.2
    annotated_image = cv2.addWeighted(full_image, 1 - alpha, mask_overlay, alpha, 0)

    # === OPTIONAL OVERLAYS ===
    if getattr(config, 'SHOW_GRID', False) or getattr(config, 'SHOW_GRID_LABELS', False):
        annotated_image = add_optimized_grid(annotated_image, width, height, scaling_factor, mpp)

    if getattr(config, 'SHOW_ADIPOCYTE_IDS', False):
        annotated_image = add_optimized_adipocyte_ids(annotated_image, adipocyte_props, adipocyte_ids, scaling_factor)

    if tumour_mask_fullres is not None:
        annotated_image = add_optimized_tumor_overlay(
            annotated_image, tumour_mask_fullres, scaling_factor, mpp,
            tumor_centroids, thumbnail_size
        )

    return annotated_image


def read_optimal_image(image_handler, width, height, scaling_factor, desired_level):
    """
    \uD83D\uDE80 OPTIMIZED image reading helper function (read near target level; resize once).
    """
    target_width = int(width * scaling_factor)
    target_height = int(height * scaling_factor)

    # Find the best level that's closest to our target resolution
    if hasattr(image_handler, 'slide') and hasattr(image_handler.slide, 'level_downsamples'):
        downsamples = image_handler.slide.level_downsamples
        level_factors = [1 / ds for ds in downsamples]
        level_diffs = [abs(factor - scaling_factor) for factor in level_factors]
        best_level = level_diffs.index(min(level_diffs))
        level_dims = image_handler.slide.level_dimensions[best_level]
        full_image_pil = image_handler.read_region((0, 0), best_level, level_dims)
    else:
        # For non-slide images, read full resolution at desired level
        full_image_pil = image_handler.read_region((0, 0), desired_level, (width, height))

    # Convert to RGB and resize to target
    full_image_pil = full_image_pil.convert('RGB')
    if full_image_pil.size != (target_width, target_height):
        full_image_pil = full_image_pil.resize((target_width, target_height), Image.Resampling.BILINEAR)

    return np.array(full_image_pil)


def create_optimized_mask_overlay(full_mask, mask_areas, adipocyte_ids, target_width, target_height, scaling_factor):
    """\uD83D\uDE80 Create mask overlay with vectorized operations (adipocyte label mask)."""
    import matplotlib.pyplot as plt

    target_pixels = int(target_width) * int(target_height)
    max_px = int(getattr(config, 'MAX_ANNOTATED_PIXELS', 250_000_000))
    if target_pixels > max_px:
        logging.warning(
            f"\u26A0\uFE0F Adipocyte label mask overlay target is large ({target_pixels:,} px) at scale {scaling_factor:.5f}. "
            f"Check MAX_ANNOTATED_PIXELS if memory becomes an issue."
        )

    # Resize mask efficiently; keep labels intact (NEAREST)
    if scaling_factor != 1.0:
        try:
            full_mask_resized = cv2.resize(
                full_mask.astype(np.int32),
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST
            )
        except cv2.error:
            # Fallback across OpenCV builds: resize as float then cast back
            full_mask_resized = cv2.resize(
                full_mask.astype(np.float32),
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST
            ).astype(full_mask.dtype)
    else:
        full_mask_resized = full_mask

    # Vectorized color mapping by area
    valid_ids = np.array(adipocyte_ids)
    if len(valid_ids) > 0:
        areas = np.array([mask_areas[aid] for aid in valid_ids])
        area_min, area_max = areas.min(), areas.max()
        if area_max > area_min:
            areas_norm = (areas - area_min) / (area_max - area_min)
        else:
            areas_norm = np.zeros_like(areas)
        colormap = plt.get_cmap('jet')
        colors = (colormap(areas_norm)[:, :3] * 255).astype(np.uint8)
    else:
        colors = np.zeros((0, 3), dtype=np.uint8)

    max_label = int(full_mask_resized.max()) if full_mask_resized.size else 0
    color_mapping = np.zeros((max_label + 1, 3), dtype=np.uint8)

    for i, aid in enumerate(valid_ids):
        if aid <= max_label:
            color_mapping[aid] = colors[i]

    # Vectorized color assignment
    mask_overlay = color_mapping[full_mask_resized]

    return mask_overlay


def add_optimized_adipocyte_ids(image, adipocyte_props, adipocyte_ids, scaling_factor):
    """Add adipocyte IDs with optimized text rendering."""
    font_scale = max(0.05 * 10, 0.5)
    thickness = max(int(0.05 * 10), 1)

    for aid in adipocyte_ids:
        if aid in adipocyte_props:
            cx = int(adipocyte_props[aid]['centroid_x'] * scaling_factor)
            cy = int(adipocyte_props[aid]['centroid_y'] * scaling_factor)

            text = str(aid)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2

            if 0 <= text_x < image.shape[1] and 0 <= text_y < image.shape[0]:
                # Black outline + white text
                cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (255, 255, 255), thickness)

    return image


def add_optimized_grid(image, width, height, scaling_factor, mpp):
    """Add analysis grid overlay."""
    target_width = int(width * scaling_factor)
    target_height = int(height * scaling_factor)

    # Grid parameters from config
    grid_size_microns = getattr(config, 'GRID_CELL_SIZE_MICRONS', 1100)
    if not mpp or mpp <= 0:
        logging.warning("\u26A0\uFE0F mpp is invalid; skipping grid overlay.")
        return image

    grid_size_pixels = int(grid_size_microns / mpp)
    grid_size_pixels = max(1, grid_size_pixels)  # avoid zero/negative step in range()

    grid_color = (128, 128, 128)  # Gray
    thickness = 1

    # Vertical lines
    for x in range(0, width, grid_size_pixels):
        x_scaled = int(x * scaling_factor)
        if x_scaled < target_width:
            cv2.line(image, (x_scaled, 0), (x_scaled, target_height), grid_color, thickness)

    # Horizontal lines
    for y in range(0, height, grid_size_pixels):
        y_scaled = int(y * scaling_factor)
        if y_scaled < target_height:
            cv2.line(image, (0, y_scaled), (target_width, y_scaled), grid_color, thickness)

    # Grid labels if enabled
    if getattr(config, 'SHOW_GRID_LABELS', False):
        font_scale = 0.5
        font_thickness = 1
        label_color = (255, 255, 255)  # White

        grid_num = 0
        for y in range(0, height, grid_size_pixels):
            for x in range(0, width, grid_size_pixels):
                x_scaled = int(x * scaling_factor)
                y_scaled = int(y * scaling_factor)
                if x_scaled < target_width and y_scaled < target_height:
                    label = str(grid_num)
                    cv2.putText(image, label, (x_scaled + 5, y_scaled + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 1)
                    cv2.putText(image, label, (x_scaled + 5, y_scaled + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, font_thickness)
                    grid_num += 1

    return image


def save_tissue_window_grid_thumbnail(
    image_handler,
    guided_windows,
    window_size,
    tissue_regions,
    output_dir,
    image_name,
    max_dim
):
    """
    Save a low-res thumbnail with tissue boundaries and guided window grid overlay.

    Args:
        image_handler: ImageHandler object for slide access
        guided_windows: List of (x, y) window coordinates in full-res
        window_size: (width, height) of processing windows in full-res
        tissue_regions: List of tissue regions (x, y, w, h) in full-res
        output_dir: Output directory
        image_name: Base image name for output file
        max_dim: Max dimension (pixels) for thumbnail (preserve aspect)
    """
    try:
        if image_handler is None:
            logging.warning("No image handler available; skipping tissue window grid thumbnail")
            return

        # Build thumbnail (RGB)
        if hasattr(image_handler, 'slide') and hasattr(image_handler.slide, 'get_thumbnail'):
            thumb_pil = image_handler.slide.get_thumbnail((max_dim, max_dim)).convert("RGB")
        else:
            # Fallback for non-slide images
            full_img = image_handler.read_region((0, 0), 0, (image_handler.width, image_handler.height)).convert("RGB")
            w, h = full_img.size
            scale = min(float(max_dim) / float(w), float(max_dim) / float(h), 1.0)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            if (new_w, new_h) != (w, h):
                try:
                    resample = Image.Resampling.BILINEAR
                except AttributeError:
                    resample = Image.BILINEAR
                full_img = full_img.resize((new_w, new_h), resample=resample)
            thumb_pil = full_img

        thumb_rgb = np.array(thumb_pil)
        thumb_bgr = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR)

        # Scale factors from full-res to thumbnail
        full_w = float(image_handler.width)
        full_h = float(image_handler.height)
        scale_x = thumb_bgr.shape[1] / full_w if full_w > 0 else 1.0
        scale_y = thumb_bgr.shape[0] / full_h if full_h > 0 else 1.0

        # Colors (BGR) - high contrast for visibility at a distance
        tissue_color = (118, 230, 0)    # #00E676 (neon green)
        grid_color = (255, 255, 0)      # #00FFFF (cyan)
        tissue_thickness = 3
        grid_thickness = 1

        # Draw tissue boundaries (regions are x, y, w, h)
        if tissue_regions:
            for region in tissue_regions:
                if len(region) != 4:
                    continue
                x, y, w, h = region
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)
                x1 = max(0, min(x1, thumb_bgr.shape[1] - 1))
                y1 = max(0, min(y1, thumb_bgr.shape[0] - 1))
                x2 = max(0, min(x2, thumb_bgr.shape[1] - 1))
                y2 = max(0, min(y2, thumb_bgr.shape[0] - 1))
                cv2.rectangle(thumb_bgr, (x1, y1), (x2, y2), tissue_color, tissue_thickness)

        # Draw guided windows
        win_w, win_h = window_size
        if guided_windows:
            for x, y in guided_windows:
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + win_w) * scale_x)
                y2 = int((y + win_h) * scale_y)
                x1 = max(0, min(x1, thumb_bgr.shape[1] - 1))
                y1 = max(0, min(y1, thumb_bgr.shape[0] - 1))
                x2 = max(0, min(x2, thumb_bgr.shape[1] - 1))
                y2 = max(0, min(y2, thumb_bgr.shape[0] - 1))
                cv2.rectangle(thumb_bgr, (x1, y1), (x2, y2), grid_color, grid_thickness)

        out_path = os.path.join(output_dir, f"{image_name}_tissue_window_grid.png")
        cv2.imwrite(out_path, thumb_bgr)
        logging.info(f"\u2705 Tissue window grid thumbnail saved: {out_path}")
    except Exception as e:
        logging.error(f"\u274C Error saving tissue window grid thumbnail: {e}")


def add_optimized_tumor_overlay(image, tumour_mask_fullres, scaling_factor, mpp, tumor_centroids=None, thumbnail_size=None):
    """Add tumor overlay with optimized processing and boundary visualization.

    - Always resizes tumour mask to annotated image size (NEAREST)
    - Fills tumour areas with deterministic per-instance colors at any size
    - Optionally draws micron boundaries, but only if under a safe pixel budget
    - Optionally draws tumor labels if centroids and thumbnail size are provided
    """
    # Target shape (W, H) for cv2.resize
    target_shape = (image.shape[1], image.shape[0])
    target_px = int(target_shape[0]) * int(target_shape[1])

    # Downsample labelled tumour mask with IDs intact.
    tumour_mask_vis = cv2.resize(
        np.asarray(tumour_mask_fullres).astype(np.uint16, copy=False),
        target_shape,
        interpolation=cv2.INTER_NEAREST
    )
    tumour_binary_vis = ((tumour_mask_vis > 0).astype(np.uint8) * 255)

    # Log explicitly which mask is being handled
    if target_px > int(getattr(config, 'MAX_ANNOTATED_PIXELS', 250_000_000)):
        logging.warning(
            f"\u26A0\uFE0F Tumour mask overlay target is large ({target_px:,} px). Mask: tumour. "
            f"Using downsampled overlay at scale {scaling_factor:.5f}."
        )

    # Create instance-colored overlay efficiently.
    tumour_overlay = build_tumor_instance_overlay(tumour_mask_vis)

    # Blend only tumour pixels so non-tumour tissue is not darkened.
    alpha_tumor = get_tumor_instance_alpha()
    tumor_pixels = tumour_mask_vis > 0
    if np.any(tumor_pixels):
        image = image.copy()
        image[tumor_pixels] = (
            (1.0 - alpha_tumor) * image[tumor_pixels]
            + alpha_tumor * tumour_overlay[tumor_pixels]
        ).astype(np.uint8)

    # Draw tumor labels if centroids are provided
    if tumor_centroids and thumbnail_size:
        # Calculate scale from thumbnail to annotated image
        # thumbnail_size is (width, height)
        # image.shape is (height, width)
        
        scale_x = image.shape[1] / thumbnail_size[0]
        scale_y = image.shape[0] / thumbnail_size[1]
        
        for i, (cx, cy) in enumerate(tumor_centroids):
            # Project centroid to annotated image coordinates
            text_x = int(cx * scale_x)
            text_y = int(cy * scale_y)
            
            label = f"T{i+1}"
            
            # Draw label with outline
            cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 0, 0), 3)
            cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (255, 255, 255), 1)

    # Micron boundaries (optional, gated by size)
    show_boundaries = bool(getattr(config, 'SHOW_TUMOR_BOUNDARIES', getattr(config, 'SHOW_TUMOUR_MICRON_BOUNDARIES', False)))
    if show_boundaries and np.any(tumour_binary_vis):
        # Avoid enormous distance transforms
        max_dt_px = int(getattr(config, 'MAX_DIST_TRANSFORM_PIXELS', 300_000_000))  # ~300 MP default
        if target_px > max_dt_px:
            logging.warning(
                f"\u26D4 Skipping tumour micron boundaries: target {target_shape[0]}x{target_shape[1]} ({target_px:,} px) "
                f"exceeds MAX_DIST_TRANSFORM_PIXELS={max_dt_px:,}. Mask: tumour. "
                f"The tumour region is still filled (downsampled)."
            )
            return image

        # Colors & distances
        micron_distances = [50, 100, 200, 300, 400, 500, 1000, 1500]
        boundary_colors = {
            50: (255, 255, 0),      # Yellow
            100: (0, 255, 255),     # Cyan
            200: (255, 0, 255),     # Magenta
            300: (0, 128, 255),     # Orange
            400: (0, 255, 0),       # Green
            500: (0, 0, 255),       # Red
            1000: (0, 0, 128),      # Blue
            1500: (255, 255, 255),  # White
        }

        # Distance transform expects distance to nearest zero; invert mask
        try:
            dist = cv2.distanceTransform(255 - tumour_binary_vis, cv2.DIST_L2, 3)
        except cv2.error as e:
            logging.exception(f"distanceTransform failed for tumour mask (size {target_shape[0]}x{target_shape[1]}). Skipping boundaries. Mask: tumour. Error: {e}")
            return image

        # Draw each boundary contour
        if not mpp or mpp <= 0:
            logging.warning("\u26A0\uFE0F mpp invalid; skipping tumour micron boundaries.")
            return image

        for micron in micron_distances:
            px = int(round(micron / mpp * scaling_factor))
            if px > 0:
                boundary = ((dist >= px - 1) & (dist <= px + 1)).astype(np.uint8)
                contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                color = boundary_colors.get(micron, (255, 255, 255))
                cv2.drawContours(image, contours, -1, color, 2)

    return image


def _build_tumor_zone_masks(tumour_mask_vis, mpp, scaling_factor):
    """Build near/intermediate/distal boolean masks from a resized tumour mask."""
    if tumour_mask_vis is None or tumour_mask_vis.size == 0:
        raise ValueError("Tumour mask for zone construction is empty")
    if not mpp or mpp <= 0:
        raise ValueError("mpp must be positive to build tumor zones")

    near_um = float(getattr(config, 'TUMOR_ZONE_NEAR_UM', 1500.0))
    intermediate_um = float(getattr(config, 'TUMOR_ZONE_INTERMEDIATE_UM', 5000.0))
    if near_um <= 0:
        logging.warning("Invalid TUMOR_ZONE_NEAR_UM=%.3f; falling back to 1500.0", near_um)
        near_um = 1500.0
    if intermediate_um <= near_um:
        logging.warning(
            "TUMOR_ZONE_INTERMEDIATE_UM=%.3f is not greater than TUMOR_ZONE_NEAR_UM=%.3f; adjusting.",
            intermediate_um, near_um
        )
        intermediate_um = near_um + 1.0

    near_px = int(round(near_um / float(mpp) * float(scaling_factor)))
    intermediate_px = int(round(intermediate_um / float(mpp) * float(scaling_factor)))
    if near_px < 1:
        logging.warning(
            "Tumor-zone near threshold collapsed to %d px at scale %.5f; clamping to 1 px.",
            near_px, scaling_factor
        )
        near_px = 1
    if intermediate_px <= near_px:
        logging.warning(
            "Tumor-zone intermediate threshold collapsed to %d px (near=%d px); clamping to near+1.",
            intermediate_px, near_px
        )
        intermediate_px = near_px + 1

    tumour_binary = ((tumour_mask_vis > 0).astype(np.uint8) * 255)
    dist = cv2.distanceTransform(255 - tumour_binary, cv2.DIST_L2, 3)

    near_mask = (dist > 0) & (dist <= near_px)
    intermediate_mask = (dist > near_px) & (dist <= intermediate_px)
    distal_mask = dist > intermediate_px

    stats = {
        'near_um': near_um,
        'intermediate_um': intermediate_um,
        'near_px': near_px,
        'intermediate_px': intermediate_px,
        'near_count': int(np.count_nonzero(near_mask)),
        'intermediate_count': int(np.count_nonzero(intermediate_mask)),
        'distal_count': int(np.count_nonzero(distal_mask)),
    }
    return near_mask, intermediate_mask, distal_mask, stats


def add_tumor_distance_zones_overlay(image, tumour_mask_fullres, scaling_factor, mpp):
    """
    Add near/intermediate/distal tumor-distance zones on top of an existing annotated image.

    Distances are measured outward from tumor boundaries and applied only outside tumor pixels.
    """
    if image is None:
        return None, {'reason': 'image_none'}
    if tumour_mask_fullres is None:
        return None, {'reason': 'tumour_mask_none'}
    if not mpp or mpp <= 0:
        return None, {'reason': 'invalid_mpp'}

    target_shape = (image.shape[1], image.shape[0])  # (W, H)
    target_px = int(target_shape[0]) * int(target_shape[1])

    max_dt_px = int(getattr(config, 'MAX_DIST_TRANSFORM_PIXELS', 300_000_000))
    if target_px > max_dt_px:
        logging.warning(
            "Skipping tumor-zone overlay: target %dx%d (%d px) exceeds MAX_DIST_TRANSFORM_PIXELS=%d",
            target_shape[0], target_shape[1], target_px, max_dt_px
        )
        return None, {'reason': 'too_large', 'target_px': target_px, 'max_dt_px': max_dt_px}

    tumour_mask_vis = cv2.resize(
        (tumour_mask_fullres > 0).astype(np.uint8) * 255,
        target_shape,
        interpolation=cv2.INTER_NEAREST,
    )
    if not np.any(tumour_mask_vis):
        logging.info("No tumor detected — skipping tumor-zone overlay")
        return None, {'reason': 'empty_tumour_mask'}

    try:
        near_mask, intermediate_mask, distal_mask, stats = _build_tumor_zone_masks(
            tumour_mask_vis, mpp, scaling_factor
        )
    except Exception as e:
        logging.warning(f"Skipping tumor-zone overlay: zone mask build failed: {e}")
        return None, {'reason': 'zone_build_failed'}

    def _color_from_config(name, default):
        color = getattr(config, name, default)
        if not isinstance(color, (tuple, list)) or len(color) != 3:
            return np.array(default, dtype=np.uint8)
        return np.array([int(np.clip(v, 0, 255)) for v in color], dtype=np.uint8)

    near_color = _color_from_config('TUMOR_ZONE_NEAR_COLOR_RGB', (255, 255, 0))
    intermediate_color = _color_from_config('TUMOR_ZONE_INTERMEDIATE_COLOR_RGB', (0, 255, 0))
    distal_color = _color_from_config('TUMOR_ZONE_DISTAL_COLOR_RGB', (0, 128, 255))

    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[near_mask] = near_color
    overlay[intermediate_mask] = intermediate_color
    overlay[distal_mask] = distal_color

    alpha = float(getattr(config, 'TUMOR_ZONE_ALPHA', 0.25))
    alpha = max(0.0, min(1.0, alpha))

    # Blend only on zone pixels so tumor interior stays unchanged.
    zone_union = near_mask | intermediate_mask | distal_mask
    if np.any(zone_union):
        blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        zoned_image = image.copy()
        zoned_image[zone_union] = blended[zone_union]
    else:
        zoned_image = image.copy()

    stats.update({
        'alpha': alpha,
        'target_width': int(target_shape[0]),
        'target_height': int(target_shape[1]),
    })
    return zoned_image, stats


# ================================================================
# HEATMAP (optional)
# ================================================================

def create_heatmap_visualization(full_mask, mask_areas, output_dir):
    """
    Create a heatmap visualization showing adipocyte density.
    (Note: this builds a float map as large as full_mask; be mindful of RAM.)
    """
    logging.info("\uD83D\uDD25 Creating density heatmap...")

    try:
        density_map = np.zeros_like(full_mask, dtype=np.float32)
        for adipocyte_id, area in mask_areas.items():
            mask_region = (full_mask == adipocyte_id)
            density_map[mask_region] = area

        if density_map.max() > 0:
            density_map = (density_map / density_map.max()) * 255

        density_map_uint8 = density_map.astype(np.uint8)
        heatmap = cv2.applyColorMap(density_map_uint8, cv2.COLORMAP_JET)

        heatmap_path = os.path.join(output_dir, "adipocyte_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap)
        logging.info(f"\u2705 Heatmap saved: {heatmap_path}")

    except Exception as e:
        logging.error(f"Error creating heatmap: {e}")


# ================================================================
# QUPATH EXPORT FUNCTIONS
# ================================================================

def _qupath_ring_area(coords):
    """Return the absolute shoelace area for a closed QuPath coordinate ring."""
    if len(coords) < 4:
        return 0.0

    area = 0.0
    for p1, p2 in zip(coords, coords[1:]):
        area += (float(p1[0]) * float(p2[1])) - (float(p2[0]) * float(p1[1]))
    return abs(area) / 2.0


def _qupath_orientation(a, b, c):
    cross = ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]))
    if abs(cross) <= 1e-9:
        return 0
    return 1 if cross > 0 else -1


def _qupath_on_segment(a, b, c):
    return (
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
    )


def _qupath_segments_intersect(a, b, c, d):
    o1 = _qupath_orientation(a, b, c)
    o2 = _qupath_orientation(a, b, d)
    o3 = _qupath_orientation(c, d, a)
    o4 = _qupath_orientation(c, d, b)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _qupath_on_segment(a, c, b):
        return True
    if o2 == 0 and _qupath_on_segment(a, d, b):
        return True
    if o3 == 0 and _qupath_on_segment(c, a, d):
        return True
    if o4 == 0 and _qupath_on_segment(c, b, d):
        return True
    return False


def _qupath_ring_has_self_touch(coords):
    """Detect self-intersections and non-adjacent self-touches in a closed ring."""
    vertices = [(float(p[0]), float(p[1])) for p in coords[:-1]]

    seen = set()
    for vertex in vertices:
        if vertex in seen:
            return True
        seen.add(vertex)

    segment_count = len(vertices)
    for i in range(segment_count):
        a = vertices[i]
        b = vertices[(i + 1) % segment_count]
        for j in range(i + 1, segment_count):
            if j == i + 1:
                continue
            if i == 0 and j == segment_count - 1:
                continue
            c = vertices[j]
            d = vertices[(j + 1) % segment_count]
            if _qupath_segments_intersect(a, b, c, d):
                return True
    return False


def _qupath_contour_to_ring(contour, row_offset, col_offset):
    """Convert an OpenCV contour to a validated closed QuPath ring."""
    coords = []
    for point in contour:
        col, row = point[0]
        full_col = float(col + col_offset)
        full_row = float(row + row_offset)
        if not (np.isfinite(full_col) and np.isfinite(full_row)):
            return None, "non-finite coordinate"
        coord = (full_col, full_row)
        if not coords or coords[-1] != coord:
            coords.append(coord)

    if len(coords) > 1 and coords[0] == coords[-1]:
        coords.pop()

    if len(set(coords)) < 3:
        return None, "fewer than 3 unique vertices"

    ring = [[x, y] for x, y in coords]
    ring.append([coords[0][0], coords[0][1]])

    if _qupath_ring_area(ring) <= 1e-6:
        return None, "zero-area contour"

    if _qupath_ring_has_self_touch(ring):
        return None, "self-intersecting or self-touching contour"

    return ring, None


def export_qupath_annotations(mask_areas, full_mask, output_dir, image_name, precomputed_properties=None):
    """Export adipocyte annotations in QuPath-compatible GeoJSON format.

    Defensive: respect the global config flags controlling QuPath GeoJSON export
    so callers that don't check the config are still protected.
    
    Args:
        mask_areas: Dict mapping adipocyte IDs to their areas
        full_mask: Full-resolution label mask
        output_dir: Output directory path
        image_name: Base name for output file
        precomputed_properties: Optional pre-computed properties dict to avoid regionprops scan
                               {id: {'centroid_x': float, 'centroid_y': float, 'area': int, 'bbox': tuple}}
    """
    # Defensive guard: respect config flags
    if not (
        getattr(config, 'ENABLE_QUPATH_EXPORT', False)
        and getattr(config, 'SAVE_QUPATH_GEOJSON', False)
    ):
        logging.info("\uD83D\uDCC1 QuPath GeoJSON export disabled in config; skipping export_qupath_annotations")
        return

    logging.info("\uD83D\uDDFA\uFE0F Exporting QuPath GeoJSON annotations...")

    try:
        # Use pre-computed properties if available (memory-efficient path)
        if precomputed_properties is not None and len(precomputed_properties) > 0:
            logging.info("\uD83E\uDDE0 Using pre-computed properties for QuPath export")
            props = None  # Not using regionprops
            prop_iter = precomputed_properties.items()
        else:
            props = regionprops(full_mask)
            prop_iter = (
                (
                    p.label,
                    {
                        'bbox': p.bbox,
                        'centroid_x': p.centroid[1],
                        'centroid_y': p.centroid[0],
                    },
                )
                for p in props
            )
        
        geojson_data = {"type": "FeatureCollection", "features": []}
        skipped_adipocyte_reasons = {}
        invalid_contour_reasons = []
        area_mismatch_labels = []

        for label, prop_dict in prop_iter:
            if label in mask_areas:
                from skimage import filters

                min_row, min_col, max_row, max_col = prop_dict['bbox']
                pad = 2
                padded_min_row = max(0, min_row - pad)
                padded_min_col = max(0, min_col - pad)
                padded_max_row = min(full_mask.shape[0], max_row + pad)
                padded_max_col = min(full_mask.shape[1], max_col + pad)

                mask_region = full_mask[padded_min_row:padded_max_row, padded_min_col:padded_max_col]
                adipocyte_mask = (mask_region == label).astype(np.uint8)
                if np.sum(adipocyte_mask) == 0:
                    skipped_adipocyte_reasons[int(label)] = "empty mask region"
                    continue

                if adipocyte_mask.shape[0] > 5 and adipocyte_mask.shape[1] > 5:
                    adipocyte_mask_smooth = filters.gaussian(adipocyte_mask.astype(float), sigma=0.5)
                    adipocyte_mask = (adipocyte_mask_smooth > 0.5).astype(np.uint8)

                contours_cv, _ = cv2.findContours(adipocyte_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours_cv) == 0:
                    skipped_adipocyte_reasons[int(label)] = "no contours"
                    continue

                valid_parts = []
                contour_skip_reasons = []
                contours_cv = sorted(contours_cv, key=cv2.contourArea, reverse=True)
                for contour in contours_cv:
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    epsilon = max(epsilon, 0.5)
                    epsilon = min(epsilon, 2.0)
                    simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
                    qupath_coords, skip_reason = _qupath_contour_to_ring(
                        simplified_contour,
                        padded_min_row,
                        padded_min_col,
                    )
                    if qupath_coords is None:
                        contour_skip_reasons.append(skip_reason)
                        invalid_contour_reasons.append((int(label), skip_reason))
                        continue
                    valid_parts.append({
                        "coords": qupath_coords,
                        "area": _qupath_ring_area(qupath_coords),
                    })

                if not valid_parts:
                    unique_reasons = sorted(set(contour_skip_reasons))
                    skipped_adipocyte_reasons[int(label)] = (
                        ", ".join(unique_reasons) if unique_reasons else "invalid contours"
                    )
                    continue

                area_pixels = float(mask_areas[label])
                exported_area = sum(part["area"] for part in valid_parts)
                if area_pixels > 0 and exported_area < area_pixels * 0.5:
                    area_mismatch_labels.append((int(label), exported_area / area_pixels))

                part_count = len(valid_parts)
                for part_index, part in enumerate(valid_parts, start=1):
                    name = f"Adipocyte_{label}"
                    properties = {
                        "name": name,
                        "objectType": "annotation",
                        "classification": {"name": "Adipocyte", "colorRGB": -65536},
                        "area_pixels": int(mask_areas[label]),
                        "centroid_x": float(prop_dict['centroid_x']),
                        "centroid_y": float(prop_dict['centroid_y']),
                        "adipocyte_id": int(label)
                    }
                    if part_count > 1:
                        name = f"{name}_part_{part_index}"
                        properties["name"] = name
                        properties["part_index"] = part_index
                        properties["part_count"] = part_count

                    feature = {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [part["coords"]]},
                        "properties": properties
                    }
                    geojson_data["features"].append(feature)

        qupath_path = os.path.join(output_dir, f"{image_name}_qupath_annotations.geojson")
        with open(qupath_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2)

        logging.info(f"\u2705 QuPath GeoJSON exported: {qupath_path}")
        logging.info(f"\uD83D\uDCCA Exported {len(geojson_data['features'])} adipocyte annotations to GeoJSON")
        if skipped_adipocyte_reasons:
            skipped_adipocyte_ids = sorted(skipped_adipocyte_reasons)
            preview = skipped_adipocyte_ids[:20]
            suffix = "..." if len(skipped_adipocyte_ids) > len(preview) else ""
            logging.warning(
                "QuPath GeoJSON skipped %d/%d adipocytes with invalid contours: %s%s",
                len(skipped_adipocyte_reasons),
                len(mask_areas),
                preview,
                suffix,
            )
            reason_preview = {label: skipped_adipocyte_reasons[label] for label in preview}
            logging.warning("QuPath GeoJSON skip reasons: %s%s", reason_preview, suffix)
        if invalid_contour_reasons:
            logging.debug(
                "QuPath GeoJSON rejected %d contour parts during validation",
                len(invalid_contour_reasons),
            )
        if area_mismatch_labels:
            preview = [(label, round(ratio, 3)) for label, ratio in area_mismatch_labels[:20]]
            suffix = "..." if len(area_mismatch_labels) > len(preview) else ""
            logging.warning(
                "QuPath GeoJSON contour area was <50%% of area_pixels for %d adipocytes; "
                "possible fragmented labels: %s%s",
                len(area_mismatch_labels),
                preview,
                suffix,
            )

    except Exception as e:
        logging.error(f"Error exporting QuPath GeoJSON: {e}")


# ================================================================
# CSV EXPORT FUNCTIONS
# ================================================================

EXTENDED_PROPERTY_COLUMNS = (
    'Eccentricity',
    'Solidity',
    'Extent',
    'Perimeter',
    'Equivalent_Diameter',
)

_REGIONPROPS_EXTENDED_PROPERTIES = (
    ('Eccentricity', 'eccentricity'),
    ('Solidity', 'solidity'),
    ('Extent', 'extent'),
    ('Perimeter', 'perimeter'),
    ('Equivalent_Diameter', 'equivalent_diameter'),
)


def _calculate_extended_properties_for_csv(full_mask, mask_areas):
    """Calculate optional morphometric CSV fields from the final labeled mask."""
    if full_mask is None:
        logging.warning("Extended properties requested, but no full mask is available")
        return {}

    try:
        props_table = regionprops_table(
            full_mask,
            properties=['label'] + [prop_name for _, prop_name in _REGIONPROPS_EXTENDED_PROPERTIES],
        )
    except Exception as e:
        logging.error(f"Extended property calculation failed: {e}")
        return {}

    valid_labels = {int(label) for label in mask_areas.keys()}
    extended_properties = {}
    for i, label in enumerate(props_table.get('label', [])):
        label_id = int(label)
        if label_id not in valid_labels:
            continue
        extended_properties[label_id] = {
            csv_name: float(props_table[prop_name][i])
            for csv_name, prop_name in _REGIONPROPS_EXTENDED_PROPERTIES
        }

    return extended_properties


def export_results_csv(
    mask_areas,
    full_mask,
    output_dir,
    image_name,
    adipocyte_distances=None,
    precomputed_properties=None,
    mpp=None,
    adipocyte_closest_tumor_ids=None,
):
    """Export adipocyte detection results to CSV format.
    
    Args:
        mask_areas: Dict mapping adipocyte IDs to their areas
        full_mask: Full-resolution label mask (may be unused if precomputed_properties provided)
        output_dir: Output directory path
        image_name: Base name for output file
        adipocyte_distances: Optional dict of distances to tumor
        precomputed_properties: Optional pre-computed properties dict to avoid regionprops scan
                               {id: {'centroid_x': float, 'centroid_y': float, 'area': int, 'bbox': tuple}}
        mpp: Microns per pixel value from slide metadata. If None, falls back to config.DEFAULT_MPP
        adipocyte_closest_tumor_ids: Optional dict mapping adipocyte IDs to nearest tumour IDs
    """
    # Use actual MPP from slide metadata, fallback to config default
    if mpp is None:
        mpp = getattr(config, 'DEFAULT_MPP', 0.5)
        logging.warning(f"\u26A0\uFE0F No MPP provided to CSV export, using default: {mpp} \u00B5m/pixel")
    
    logging.info(f"\uD83D\uDCCA Exporting results to CSV (MPP: {mpp:.4f} \u00B5m/pixel)...")

    try:
        # Grid ID mapping (match add_optimized_grid numbering: row-major by grid cell)
        grid_size_microns = getattr(config, 'GRID_CELL_SIZE_MICRONS', 1100)
        grid_size_pixels = max(1, int(grid_size_microns / mpp))
        mask_h, mask_w = full_mask.shape[:2]
        num_cols = max(1, (mask_w + grid_size_pixels - 1) // grid_size_pixels)
        num_rows = max(1, (mask_h + grid_size_pixels - 1) // grid_size_pixels)

        def _grid_id_for_centroid(cx, cy):
            col = int(cx // grid_size_pixels)
            row = int(cy // grid_size_pixels)
            if col < 0:
                col = 0
            elif col >= num_cols:
                col = num_cols - 1
            if row < 0:
                row = 0
            elif row >= num_rows:
                row = num_rows - 1
            return int(row * num_cols + col)

        csv_data = []

        # Check if there are any valid (non-negative) tumor distances
        has_valid_tumor_distances = (
            adipocyte_distances is not None
            and any(d is not None and d >= 0 for d in adipocyte_distances.values())
        )

        def _closest_tumor_id_for_adipocyte(adipocyte_id):
            if adipocyte_closest_tumor_ids and adipocyte_id in adipocyte_closest_tumor_ids:
                try:
                    closest_id = int(adipocyte_closest_tumor_ids[adipocyte_id])
                except (TypeError, ValueError):
                    closest_id = 0
                if closest_id > 0:
                    return closest_id
            return 1

        extended_properties_requested = bool(getattr(config, 'CALCULATE_EXTENDED_PROPERTIES', False))
        extended_properties_written = False

        if precomputed_properties is not None and len(precomputed_properties) > 0:
            # Use pre-computed properties (memory-efficient path)
            logging.info("\uD83E\uDDE0 Using pre-computed properties for CSV export")
            if extended_properties_requested:
                precomputed_by_label = {
                    int(aid): props for aid, props in precomputed_properties.items()
                }
                for aid, extended_props in _calculate_extended_properties_for_csv(full_mask, mask_areas).items():
                    if aid in precomputed_by_label:
                        precomputed_by_label[aid].update(extended_props)

            for aid, props in precomputed_properties.items():
                row = {
                    'Adipocyte_ID': aid,
                    'Area_Pixels': props['area'],
                    'Area_Microns_Squared': props['area'] * (mpp ** 2),
                    'Centroid_X': props['centroid_x'],
                    'Centroid_Y': props['centroid_y'],
                    'Grid_ID': _grid_id_for_centroid(props['centroid_x'], props['centroid_y']),
                    'MPP_Used': mpp,
                }

                if has_valid_tumor_distances and adipocyte_distances and aid in adipocyte_distances:
                    dist_value = adipocyte_distances[aid]
                    if dist_value is not None and dist_value >= 0:
                        row['Closest_Tumour_ID'] = _closest_tumor_id_for_adipocyte(aid)
                        row['Distance_To_Closest_Tumour'] = dist_value
                        if dist_value <= 100:
                            row['Distance_Bin'] = 'Close'
                        elif dist_value <= 500:
                            row['Distance_Bin'] = 'Medium'
                        else:
                            row['Distance_Bin'] = 'Far'

                if extended_properties_requested:
                    if 'bbox' in props:
                        bbox = props['bbox']
                        row.update({
                            'bbox_min_x': bbox[1],
                            'bbox_min_y': bbox[0],
                            'bbox_max_x': bbox[3],
                            'bbox_max_y': bbox[2],
                        })

                    extended_values = {
                        column: props[column]
                        for column in EXTENDED_PROPERTY_COLUMNS
                        if column in props
                    }
                    if len(extended_values) == len(EXTENDED_PROPERTY_COLUMNS):
                        row.update(extended_values)
                        extended_properties_written = True

                csv_data.append(row)
        else:
            # Fall back to regionprops (standard path)
            props = regionprops(full_mask)
            for prop in props:
                if prop.label in mask_areas:
                    row = {
                        'Adipocyte_ID': prop.label,
                        'Area_Pixels': mask_areas[prop.label],
                        'Area_Microns_Squared': mask_areas[prop.label] * (mpp ** 2),
                        'Centroid_X': prop.centroid[1],
                        'Centroid_Y': prop.centroid[0],
                        'Grid_ID': _grid_id_for_centroid(prop.centroid[1], prop.centroid[0]),
                        'MPP_Used': mpp,
                    }

                    if has_valid_tumor_distances and adipocyte_distances and prop.label in adipocyte_distances:
                        dist_value = adipocyte_distances[prop.label]
                        if dist_value is not None and dist_value >= 0:
                            row['Closest_Tumour_ID'] = _closest_tumor_id_for_adipocyte(prop.label)
                            row['Distance_To_Closest_Tumour'] = dist_value
                            if dist_value <= 100:
                                row['Distance_Bin'] = 'Close'
                            elif dist_value <= 500:
                                row['Distance_Bin'] = 'Medium'
                            else:
                                row['Distance_Bin'] = 'Far'

                    if extended_properties_requested:
                        row.update({
                            'bbox_min_x': prop.bbox[1],
                            'bbox_min_y': prop.bbox[0],
                            'bbox_max_x': prop.bbox[3],
                            'bbox_max_y': prop.bbox[2],
                            'Eccentricity': prop.eccentricity,
                            'Solidity': prop.solidity,
                            'Extent': prop.extent,
                            'Perimeter': prop.perimeter,
                            'Equivalent_Diameter': prop.equivalent_diameter
                        })
                        extended_properties_written = True

                    csv_data.append(row)

        csv_path = os.path.join(output_dir, f"adipocyte_information_{image_name}.csv")
        if csv_data:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = list(csv_data[0].keys())
                for row in csv_data[1:]:
                    for fieldname in row.keys():
                        if fieldname not in fieldnames:
                            fieldnames.append(fieldname)
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)

        logging.info(f"\u2705 CSV results exported: {csv_path}")
        logging.info(f"\uD83D\uDCC8 Exported data for {len(csv_data)} adipocytes")

        if extended_properties_requested and extended_properties_written:
            logging.info("\uD83D\uDCCA Extended properties calculated: eccentricity, solidity, extent, perimeter, equivalent_diameter")
        elif extended_properties_requested:
            logging.warning("\u26A0\uFE0F Extended properties requested, but no extended measurements were written")
        else:
            logging.info("\uD83D\uDCCA BIBLE default properties only (use --extended_properties for additional morphological properties)")

        if csv_data:
            areas_um2 = [row['Area_Microns_Squared'] for row in csv_data]
            summary_stats = {
                'image_name': image_name,
                'total_adipocytes': len(csv_data),
                'mean_area_um2': float(np.mean(areas_um2)),
                'median_area_um2': float(np.median(areas_um2)),
                'std_area_um2': float(np.std(areas_um2)),
                'min_area_um2': float(np.min(areas_um2)),
                'max_area_um2': float(np.max(areas_um2)),
                'total_area_um2': float(np.sum(areas_um2)),
                'extended_properties_enabled': bool(extended_properties_requested)
            }
            summary_path = os.path.join(output_dir, f"{image_name}_summary_stats.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_stats, f, indent=2)
            logging.info(f"\u2705 Summary statistics saved: {summary_path}")

    except Exception as e:
        logging.exception(f"Error exporting CSV results: {e}")
        raise


# ================================================================
# IMAGE STITCHING AND RECONSTRUCTION
# ================================================================

def stitch_postprocessed_windows(postprocessed_windows, window_coords, window_size, output_dir, image_name):
    """Stitch post-processed windows back into a full image."""
    logging.info("\uD83E\uDDE9 Stitching post-processed windows...")

    try:
        if not postprocessed_windows:
            logging.warning("No post-processed windows to stitch")
            return

        max_x = max(x + window_size[0] for x, y in window_coords)
        max_y = max(y + window_size[1] for x, y in window_coords)

        stitched = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        for window, (x, y) in zip(postprocessed_windows, window_coords):
            h, w = window.shape[:2]
            stitched[y:y + h, x:x + w] = window

        stitched_path = os.path.join(output_dir, f"{image_name}_stitched_postprocessed.png")
        cv2.imwrite(stitched_path, cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR))
        logging.info(f"\u2705 Stitched image saved: {stitched_path}")

    except Exception as e:
        logging.error(f"Error stitching windows: {e}")


# ================================================================
# MASK VISUALIZATION
# ================================================================

def save_mask_visualization(full_mask, output_dir, image_name):
    """Save visualization of the detection mask (random-color labels)."""
    logging.info("\uD83C\uDFA8 Creating mask visualization...")

    try:
        unique_labels = np.unique(full_mask)
        unique_labels = unique_labels[unique_labels != 0]

        mask_vis = np.zeros((*full_mask.shape, 3), dtype=np.uint8)
        np.random.seed(42)  # reproducible colors
        for label in unique_labels:
            color = np.random.randint(0, 255, 3)
            mask_vis[full_mask == label] = color

        mask_path = os.path.join(output_dir, f"{image_name}_mask_visualization.png")
        cv2.imwrite(mask_path, cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))

        raw_mask_path = os.path.join(output_dir, f"{image_name}_raw_mask.npy")
        np.save(raw_mask_path, full_mask)

        logging.info(f"\u2705 Mask visualization saved: {mask_path}")
        logging.info(f"\uD83D\uDCBE Raw mask saved: {raw_mask_path}")

    except Exception as e:
        logging.error(f"Error creating mask visualization: {e}")


# ================================================================
# DEMO / ONE-OFF IMAGE EXPORTS
# ================================================================

def save_plain_image(image_handler, output_dir, image_name):
    """
    Save the slide as a plain (unannotated) TIFF at the configured scale.

    Uses the same ANNOTATED_IMAGE_SCALE / MAX_ANNOTATED_PIXELS safety caps
    as the annotated image so the output stays memory-safe.
    Gated by config.SAVE_UNPROCESSED_IMAGE.
    """
    logging.info("\U0001F5BC\uFE0F Saving plain (unannotated) slide image...")
    try:
        width, height = image_handler.width, image_handler.height
        requested_scale = float(getattr(config, 'ANNOTATED_IMAGE_SCALE', 1.0))
        scaling_factor = _compute_safe_scale(width, height, requested_scale)
        desired_level = getattr(config, 'DESIRED_RESOLUTION_LEVEL', 0)

        plain_image = read_optimal_image(
            image_handler, width, height, scaling_factor, desired_level
        )
        plain_image = np.ascontiguousarray(plain_image, dtype=np.uint8)

        output_path = os.path.join(output_dir, f"{image_name}_plain.tiff")
        tifffile.imwrite(output_path, plain_image, bigtiff=True)
        logging.info(
            f"\u2705 Plain image saved: {output_path} "
            f"({plain_image.shape[1]}x{plain_image.shape[0]}, scale={scaling_factor:.4f})"
        )

        # Also save an inverted-only version (no Sobel)
        inverted_plain = (255 - plain_image).astype(np.uint8)
        inverted_path = os.path.join(output_dir, f"{image_name}_inverted.tiff")
        tifffile.imwrite(inverted_path, inverted_plain, bigtiff=True)
        logging.info(
            f"\u2705 Inverted image saved: {inverted_path} "
            f"({inverted_plain.shape[1]}x{inverted_plain.shape[0]}, scale={scaling_factor:.4f})"
        )
    except Exception as e:
        logging.error(f"\u274C Error saving plain/inverted image: {e}")


def save_sobel_inverted_image(image_handler, output_dir, image_name):
    """
    Save the slide with inversion + Sobel edge detection applied, as a TIFF.

    Pipeline: read slide → invert (255 - img) → per-channel Sobel magnitude → save.
    Matches the per-window preprocessing used during inference.
    Gated by config.SAVE_POSTPROCESSED_IMAGE.
    """
    logging.info("\U0001F9EA Saving Sobel + inverted slide image...")
    try:
        width, height = image_handler.width, image_handler.height
        requested_scale = float(getattr(config, 'ANNOTATED_IMAGE_SCALE', 1.0))
        scaling_factor = _compute_safe_scale(width, height, requested_scale)
        desired_level = getattr(config, 'DESIRED_RESOLUTION_LEVEL', 0)

        image_rgb = read_optimal_image(
            image_handler, width, height, scaling_factor, desired_level
        )

        # --- Inversion (same as config.APPLY_IMAGE_INVERSION) ---
        inverted = (255 - image_rgb).astype(np.uint8)

        # --- Per-channel Sobel magnitude (same kernel as core_processing) ---
        sobel_out = np.zeros_like(inverted, dtype=np.uint8)
        for ch in range(inverted.shape[2]):
            sobelx = cv2.Sobel(inverted[:, :, ch], cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(inverted[:, :, ch], cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(sobelx ** 2 + sobely ** 2)
            sobel_out[:, :, ch] = np.clip(mag, 0, 255).astype(np.uint8)

        sobel_out = np.ascontiguousarray(sobel_out, dtype=np.uint8)

        output_path = os.path.join(output_dir, f"{image_name}_sobel_inverted.tiff")
        tifffile.imwrite(output_path, sobel_out, bigtiff=True)
        logging.info(
            f"\u2705 Sobel+inverted image saved: {output_path} "
            f"({sobel_out.shape[1]}x{sobel_out.shape[0]}, scale={scaling_factor:.4f})"
        )
    except Exception as e:
        logging.error(f"\u274C Error saving Sobel+inverted image: {e}")


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'annotate_image_with_adipocytes',
    'create_heatmap_visualization',
    'export_qupath_annotations',
    'export_results_csv',
    'stitch_postprocessed_windows',
    'save_mask_visualization',
    'save_plain_image',
    'save_sobel_inverted_image',
    'add_tumor_distance_zones_overlay',
]
