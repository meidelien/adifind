#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tumor Detection Module
======================

Handles tumor segmentation, distance computation, and analysis for AdiFind WSI analysis.
"""

import os
import logging
import numpy as np
import cv2
import time
from scipy.ndimage import distance_transform_edt, label
from PIL import Image

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Import from other modules
from config import config, paths
from image_processing import CustomBatchPredictor
from model_registry import merge_detectron2_builtin_config, resolve_model_path
from system_utils import memory_manager
from tumor_colors import build_tumor_instance_overlay, get_tumor_instance_alpha

# Try GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cp_distance_transform_edt
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available, using CPU for distance transforms")


# ================================================================
# TUMOR MODEL CONFIGURATION
# ================================================================

def configure_tumour_model(model_dir=None):
    """
    Configure tumor detection model (from original BIBLE).
    
    Args:
        model_dir: Directory containing tumor model. If None, the model
                   is auto-downloaded from HuggingFace Hub.
        
    Returns:
        Configured tumor predictor
    """
    model_path, model_dir = resolve_model_path(
        "tumor",
        model_dir=model_dir or paths.TUMOR_MODEL_DIR,
        model_checkpoint=paths.TUMOR_MODEL_CHECKPOINT,
    )

    logging.info("Configuring tumor model...")
    
    cfg = get_cfg()
    cfg.OUTPUT_DIR = model_dir
    merge_detectron2_builtin_config(cfg)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Tumor
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.SOLVER.AMP.ENABLED = True
    cfg.MODEL.DEVICE = 'cuda' if config.USE_GPU_INFERENCE else 'cpu'
    
    logging.info("? Tumor model configured")
    return CustomBatchPredictor(cfg)


# ================================================================
# TUMOR SEGMENTATION FUNCTIONS
# ================================================================

def _extract_instance_scores(instances, mask_count):
    """Return Detectron scores when available, otherwise None."""
    if not hasattr(instances, "scores"):
        return None
    try:
        scores = instances.scores.cpu().numpy()
    except Exception:
        return None
    if len(scores) != mask_count:
        return None
    return scores


def _build_labelled_tumor_mask(masks, scores=None):
    """Build a one-based labelled tumour mask from filtered instance masks."""
    if len(masks) == 0:
        return None, [], []

    ordered_records = []
    for index, mask in enumerate(masks):
        mask_bool = mask.astype(bool)
        area = int(np.count_nonzero(mask_bool))
        score = float(scores[index]) if scores is not None else None
        ordered_records.append((index, score, area, mask_bool))

    if scores is not None:
        ordered_records.sort(key=lambda record: (-record[1], record[0]))

    label_mask = np.zeros(masks[0].shape, dtype=np.uint16)
    tumor_areas = []
    tumor_centroids = []
    next_label = 1
    skipped_overlapping = 0

    for _index, _score, _area, mask_bool in ordered_records:
        assign_mask = mask_bool & (label_mask == 0)
        assigned_area = int(np.count_nonzero(assign_mask))
        if assigned_area == 0:
            skipped_overlapping += 1
            continue

        label_mask[assign_mask] = next_label
        mask_np = assign_mask.astype(np.uint8)
        moments = cv2.moments(mask_np)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            ys, xs = np.nonzero(assign_mask)
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))

        tumor_areas.append(assigned_area)
        tumor_centroids.append((cx, cy))
        next_label += 1

    if skipped_overlapping:
        logging.info("Skipped %d fully overlapping tumor detections after label assignment", skipped_overlapping)

    return label_mask, tumor_areas, tumor_centroids

def optimized_segment_tumour_on_thumbnail(thumbnail_np, tumour_predictor, full_shape):
    """
    OPTIMIZED tumor segmentation on thumbnail (from original BIBLE).
    
    Args:
        thumbnail_np: Thumbnail image for tumor detection
        tumour_predictor: Configured tumor predictor
        full_shape: Full image dimensions
        
    Returns:
        tuple: (tumor_mask_thumbnail, tumor_mask_analysis, tumor_mask_fullres,
                num_tumors, tumor_areas, tumor_centroids)
    """
    start_time = time.time()
    logging.info("Starting OPTIMIZED tumor segmentation...")
    
    # Step 1: Inference on thumbnail
    outputs = tumour_predictor([thumbnail_np])
    output = outputs[0]
    instances = output["instances"]
    masks = instances.pred_masks.cpu().numpy() if len(instances) > 0 else []
    scores = _extract_instance_scores(instances, len(masks)) if len(masks) > 0 else None

    # Step 2: Labelled mask construction with minimum area filter
    if len(masks) > 0:
        # Filter out tiny false-positive detections (e.g. 1-2 pixel artifacts)
        min_tumor_pixels = getattr(config, 'MIN_TUMOR_AREA_PIXELS', 100)
        keep_indices = [
            i for i, mask in enumerate(masks)
            if int(np.count_nonzero(mask.astype(bool))) >= min_tumor_pixels
        ]
        filtered_masks = [masks[i] for i in keep_indices]
        filtered_scores = scores[keep_indices] if scores is not None and len(keep_indices) > 0 else None
        tumour_mask, tumor_areas, tumor_centroids = _build_labelled_tumor_mask(
            filtered_masks,
            filtered_scores,
        )
        if tumour_mask is None:
            tumour_mask = np.zeros(thumbnail_np.shape[:2], dtype=np.uint16)
        num_tumors = len(tumor_areas)
        if len(filtered_masks) < len(masks):
            logging.info(f"Filtered {len(masks) - len(filtered_masks)} small tumor detections below {min_tumor_pixels}px threshold")
    else:
        tumour_mask = np.zeros(thumbnail_np.shape[:2], dtype=np.uint16)
        num_tumors = 0
        tumor_areas = []
        tumor_centroids = []

    logging.info(f"Found {num_tumors} tumor regions")
    
    # Step 3: Smart upscaling to analysis resolution
    analysis_downsample = 160  # Analysis resolution
    analysis_shape = (
        max(1, full_shape[0] // analysis_downsample),
        max(1, full_shape[1] // analysis_downsample)
    )
    
    # Resize to analysis resolution
    tumor_mask_analysis = cv2.resize(
        tumour_mask.astype(np.uint16, copy=False),
        (analysis_shape[1], analysis_shape[0]), 
        interpolation=cv2.INTER_NEAREST
    ).astype(np.uint16)
    
    # Step 4: Full resolution upscaling
    tumor_mask_fullres = cv2.resize(
        tumour_mask.astype(np.uint16, copy=False),
        (full_shape[1], full_shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.uint16)
    
    duration = time.time() - start_time
    logging.info(f"? Tumor segmentation completed in {duration:.2f} seconds")
    
    return tumour_mask, tumor_mask_analysis, tumor_mask_fullres, num_tumors, tumor_areas, tumor_centroids


def save_tumor_thumbnail_overlay(thumbnail_np, tumor_mask_thumbnail, output_dir, image_name):
    """
    Save a reviewable tumor segmentation thumbnail with instance-colored overlay.
    
    Args:
        thumbnail_np: RGB thumbnail image used for tumor detection (H, W, 3)
        tumor_mask_thumbnail: Labelled tumor mask at thumbnail resolution (H, W)
        output_dir: Output directory for the image
        image_name: Base image name for output file
    """
    try:
        if thumbnail_np is None or tumor_mask_thumbnail is None:
            logging.warning("No tumor thumbnail or mask available; skipping thumbnail overlay save")
            return

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Ensure shapes match (resize mask if needed)
        thumb_h, thumb_w = thumbnail_np.shape[:2]
        mask = tumor_mask_thumbnail
        if mask.shape[0] != thumb_h or mask.shape[1] != thumb_w:
            mask = cv2.resize(mask.astype(np.uint16), (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)

        # Prepare overlay
        base = thumbnail_np.astype(np.uint8, copy=False)
        overlay = base.copy()
        mask_bool = mask > 0
        color_overlay = build_tumor_instance_overlay(mask)
        alpha = get_tumor_instance_alpha()

        if mask_bool.any():
            overlay[mask_bool] = (
                (1.0 - alpha) * base[mask_bool] + alpha * color_overlay[mask_bool]
            ).astype(np.uint8)

            font_scale = max(0.45, min(1.2, max(thumb_w, thumb_h) / 1200.0))
            thickness = max(1, int(round(font_scale * 2)))
            for tumor_id in np.unique(mask):
                tumor_id = int(tumor_id)
                if tumor_id <= 0:
                    continue
                ys, xs = np.nonzero(mask == tumor_id)
                if len(xs) == 0:
                    continue
                text = f"T{tumor_id}"
                text_x = int(np.mean(xs))
                text_y = int(np.mean(ys))
                cv2.putText(
                    overlay,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    thickness + 2,
                )
                cv2.putText(
                    overlay,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )

        # Save as PNG (convert RGB -> BGR for OpenCV)
        out_path = os.path.join(output_dir, f"{image_name}_tumor_thumbnail.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        logging.info(f"? Tumor thumbnail overlay saved: {out_path}")
    except Exception as e:
        logging.error(f"? Error saving tumor thumbnail overlay: {e}")


# ================================================================
# DISTANCE COMPUTATION FUNCTIONS
# ================================================================

def _format_safe_rate(count, seconds, units):
    """Return a human-readable rate without raising on zero-duration timings."""
    if seconds <= 0:
        return f"instant ({count} {units.split('/')[0]})" if count else "N/A"
    return f"{count / seconds:.1f} {units}"


def _format_safe_percent(part, total):
    """Return a profiling percentage without raising on zero-duration totals."""
    if total <= 0:
        return "N/A"
    return f"{part / total * 100:5.1f}%"


def optimized_compute_adipocyte_distance_metrics(mask_areas, full_mask, tumor_mask_analysis, mpp,
                                                full_shape, analysis_downsample=160):
    """Compute adipocyte distances and nearest labelled tumour IDs."""
    start_time = time.time()
    logging.info("Computing adipocyte distances to tumor...")

    tumor_labels = None if tumor_mask_analysis is None else np.asarray(tumor_mask_analysis)
    tumor_pixels = None if tumor_labels is None else (tumor_labels > 0)
    tumor_pixel_count = 0 if tumor_pixels is None else int(np.count_nonzero(tumor_pixels))
    min_significant_pixels = getattr(config, 'MIN_TUMOR_PIXELS_FOR_DISTANCE', 50)
    if tumor_pixels is None or tumor_pixel_count < min_significant_pixels:
        logging.info(
            f"No significant tumor found (tumor pixels={tumor_pixel_count}, "
            f"threshold={min_significant_pixels}). Setting all distances to -1."
        )
        return {aid: -1 for aid in mask_areas.keys()}, {}
    
    # Analysis resolution shape
    analysis_shape = (
        max(1, full_shape[0] // analysis_downsample),
        max(1, full_shape[1] // analysis_downsample)
    )
    
    # Downsample adipocyte mask to analysis resolution
    adipocyte_mask_analysis = cv2.resize(
        full_mask.astype(np.uint16),
        (analysis_shape[1], analysis_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Compute distance transform from tumor boundaries
    distance_start = time.time()
    logging.info("Computing distance transform...")

    positive_labels = np.unique(tumor_labels[tumor_pixels])
    has_instance_labels = len(positive_labels) > 1 or (
        len(positive_labels) == 1 and int(positive_labels[0]) != 1
    )
    nearest_tumor_labels = None

    if has_instance_labels:
        distance_map, nearest_indices = distance_transform_edt(
            ~tumor_pixels,
            return_indices=True,
        )
        nearest_tumor_labels = tumor_labels[tuple(nearest_indices)]
        logging.info("Used CPU distance transform with labelled tumour nearest-ID extraction")
    elif CUPY_AVAILABLE and config.USE_CUPY:
        try:
            # GPU-accelerated distance transform
            gpu_transfer_start = time.time()
            tumor_mask_gpu = cp.asarray(~tumor_pixels)  # Invert for distance from tumor
            gpu_transfer_time = time.time() - gpu_transfer_start
            
            gpu_compute_start = time.time()
            distance_map_gpu = cp_distance_transform_edt(tumor_mask_gpu)
            gpu_compute_time = time.time() - gpu_compute_start
            
            gpu_download_start = time.time()
            distance_map = cp.asnumpy(distance_map_gpu)
            gpu_download_time = time.time() - gpu_download_start
            
            total_distance_time = time.time() - distance_start
            logging.info("? Used GPU-accelerated distance transform")
            logging.info(f"   GPU data transfer: {gpu_transfer_time:.3f}s")
            logging.info(f"   ? GPU computation: {gpu_compute_time:.3f}s") 
            logging.info(f"   GPU download: {gpu_download_time:.3f}s")
            logging.info(f"   Total GPU distance transform: {total_distance_time:.3f}s")
        except Exception as e:
            logging.warning(f"GPU distance transform failed, using CPU: {e}")
            cpu_fallback_start = time.time()
            distance_map = distance_transform_edt(~tumor_pixels)
            cpu_fallback_time = time.time() - cpu_fallback_start
            logging.info(f"   CPU fallback completed: {cpu_fallback_time:.3f}s")
    else:
        # CPU distance transform
        cpu_start = time.time()
        distance_map = distance_transform_edt(~tumor_pixels)
        cpu_time = time.time() - cpu_start
        if not CUPY_AVAILABLE:
            logging.info("Used CPU distance transform (CuPy not available)")
        else:
            logging.info("Used CPU distance transform (CuPy acceleration disabled)")
        logging.info(f"   CPU distance transform: {cpu_time:.3f}s")
    
    # Convert pixel distances to microns
    analysis_mpp = mpp * analysis_downsample
    distance_map_microns = distance_map * analysis_mpp
    
    # Calculate distances for each adipocyte with profiling
    adipocyte_extraction_start = time.time()
    logging.info(f"Extracting distances for {len(mask_areas)} adipocytes...")
    
    adipocyte_distances = {}
    adipocyte_closest_tumor_ids = {}
    processed_count = 0
    
    for adipocyte_id in mask_areas.keys():
        # Find adipocyte pixels in analysis mask
        adipocyte_pixels = (adipocyte_mask_analysis == adipocyte_id)
        
        if np.sum(adipocyte_pixels) > 0:
            # Get minimum distance for this adipocyte
            distances = distance_map_microns[adipocyte_pixels]
            min_index = int(np.argmin(distances))
            min_distance = distances[min_index]
            adipocyte_distances[adipocyte_id] = float(min_distance)
            if nearest_tumor_labels is not None:
                nearest_labels_for_adipocyte = nearest_tumor_labels[adipocyte_pixels]
                closest_tumor_id = int(nearest_labels_for_adipocyte[min_index])
                if closest_tumor_id > 0:
                    adipocyte_closest_tumor_ids[adipocyte_id] = closest_tumor_id
            elif tumor_pixel_count > 0:
                adipocyte_closest_tumor_ids[adipocyte_id] = 1
        else:
            adipocyte_distances[adipocyte_id] = -1
        
        processed_count += 1
        
        # Log progress for large numbers of adipocytes
        if processed_count % 1000 == 0:
            progress = (processed_count / len(mask_areas) * 100) if mask_areas else 0.0
            logging.info(
                "   Distance extraction progress: %.1f%% (%d/%d)",
                progress,
                processed_count,
                len(mask_areas),
            )
    
    adipocyte_extraction_time = time.time() - adipocyte_extraction_start
    logging.info(f"Adipocyte distance extraction completed: {adipocyte_extraction_time:.3f}s")
    logging.info(
        "   Extraction rate: %s",
        _format_safe_rate(len(mask_areas), adipocyte_extraction_time, "adipocytes/second")
    )
    
    duration = time.time() - start_time
    distance_transform_time = time.time() - distance_start
    
    # Comprehensive profiling summary
    logging.info("TUMOR DISTANCE COMPUTATION PROFILING:")
    logging.info("=" * 50)
    logging.info(
        "Distance Transform:     %8.3fs (%s)",
        distance_transform_time,
        _format_safe_percent(distance_transform_time, duration),
    )
    logging.info(
        "Adipocyte Extraction:   %8.3fs (%s)",
        adipocyte_extraction_time,
        _format_safe_percent(adipocyte_extraction_time, duration),
    )
    logging.info("=" * 50)
    logging.info(f"TOTAL DISTANCE TIME:    {duration:8.3f}s")
    logging.info(f"CuPy Distance Ops:     {'ENABLED' if CUPY_AVAILABLE and config.USE_CUPY else 'DISABLED'}")
    logging.info(
        "Processing Efficiency:  %s",
        _format_safe_rate(len(mask_areas), duration, "adipocytes/second")
    )
    
    return adipocyte_distances, adipocyte_closest_tumor_ids


def optimized_compute_adipocyte_distances(mask_areas, full_mask, tumor_mask_analysis, mpp,
                                        full_shape, analysis_downsample=160):
    """
    OPTIMIZED adipocyte distance computation (from original BIBLE).

    Args:
        mask_areas: Dictionary of adipocyte areas
        full_mask: Full resolution adipocyte mask
        tumor_mask_analysis: Tumor mask at analysis resolution
        mpp: Microns per pixel
        full_shape: Full image shape
        analysis_downsample: Downsample factor for analysis

    Returns:
        dict: Adipocyte distances to tumor in microns
    """
    adipocyte_distances, _closest_tumor_ids = optimized_compute_adipocyte_distance_metrics(
        mask_areas=mask_areas,
        full_mask=full_mask,
        tumor_mask_analysis=tumor_mask_analysis,
        mpp=mpp,
        full_shape=full_shape,
        analysis_downsample=analysis_downsample,
    )
    return adipocyte_distances


# ================================================================
# TUMOR ANALYSIS FUNCTIONS
# ================================================================

def save_tumor_csv(output_dir, image_name, num_tumors, tumor_areas, tumor_centroids, mpp):
    """
    Save tumor detection results to CSV.
    
    Args:
        output_dir: Output directory
        image_name: Name of the image
        num_tumors: Number of detected tumors
        tumor_areas: List of tumor areas
        tumor_centroids: List of tumor centroids
        mpp: Microns per pixel
    """
    import csv
    
    if num_tumors == 0:
        logging.info("No tumors detected, skipping tumor CSV")
        return
    
    csv_path = os.path.join(output_dir, f"{image_name}_tumor_results.csv")
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['tumor_id', 'area_pixels', 'area_um2', 'centroid_x', 'centroid_y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, (area, centroid) in enumerate(zip(tumor_areas, tumor_centroids)):
                row = {
                    'tumor_id': i + 1,
                    'area_pixels': area,
                    'area_um2': area * (mpp ** 2),
                    'centroid_x': centroid[0],
                    'centroid_y': centroid[1]
                }
                writer.writerow(row)
        
        logging.info(f"? Tumor CSV saved: {csv_path}")
        
    except Exception as e:
        logging.error(f"? Error saving tumor CSV: {e}")


def save_distance_colored_image(image_handler, mask_areas, adipocyte_distances, output_dir, image_name, 
                               full_mask=None, adipocyte_props=None, tumor_mask_fullres=None):
    """
    Save image with adipocytes colored by distance to tumor (from original BIBLE).
    
    Args:
        image_handler: Image handler object
        mask_areas: Dictionary of adipocyte areas
        adipocyte_distances: Dictionary of adipocyte distances  
        output_dir: Output directory
        image_name: Name of the image
        full_mask: Full resolution mask with adipocyte labels
        adipocyte_props: Adipocyte properties
        tumor_mask_fullres: Tumor mask for overlay
    """
    logging.info("Creating distance-colored image...")
    
    try:
        import matplotlib.pyplot as plt
        import tifffile
        from system_utils import get_mpp
        
        # Extract adipocyte IDs and distances
        adipocyte_ids = list(adipocyte_distances.keys())
        min_dists = [adipocyte_distances[aid] for aid in adipocyte_ids]
        
        # If no distances available, return
        if min_dists is None or all(d is None or d < 0 for d in min_dists):
            logging.warning("No distance data available for distance coloring.")
            return
        
        # Get image dimensions and scaling
        width = image_handler.width
        height = image_handler.height
        
        # Use reasonable scaling for memory management
        max_dimension = 4000
        scaling_factor = min(1.0, max_dimension / max(width, height))
        desired_level = 0
        
        mpp = get_mpp(image_handler)
        
        # Create distance-colored image using BIBLE method
        distance_colored_image = create_optimized_distance_colored_annotated_image_with_distances(
            image_handler, full_mask, adipocyte_props, adipocyte_ids, min_dists, 
            width, height, scaling_factor, mpp, tumor_mask_fullres, desired_level
        )
        
        if distance_colored_image is not None:
            output_path = os.path.join(output_dir, f"{image_name}_distance_colored.tiff")
            tifffile.imwrite(output_path, distance_colored_image, bigtiff=True)
            logging.info(f"Distance-colored image saved at {output_path}")
        else:
            logging.warning("Distance-colored image creation failed")
        
    except Exception as e:
        logging.error(f"? Error creating distance-colored image: {e}")


def create_optimized_distance_colored_annotated_image_with_distances(image_handler, full_mask, adipocyte_props, 
                                                                   adipocyte_ids, min_dists, width, height, 
                                                                   scaling_factor, mpp, tumour_mask_fullres, desired_level):
    """Create annotated image with adipocytes colored by pre-computed distances to tumor boundaries (from BIBLE)."""
    
    # If no distances available, return None
    if min_dists is None or all(d is None for d in min_dists):
        logging.warning("No distance data available for distance coloring.")
        return None
    
    # === OPTIMIZED IMAGE READING ===  
    from core_processing import MemoryManager
    full_image = MemoryManager.read_optimal_image(
        image_handler, width, height, scaling_factor, desired_level
    )
    
    target_width = int(width * scaling_factor)
    target_height = int(height * scaling_factor)
    
    # === CREATE DISTANCE-COLORED MASK OVERLAY USING PRE-COMPUTED DISTANCES ===
    mask_overlay = create_distance_colored_mask_overlay_with_distances(
        full_mask, adipocyte_ids, min_dists, target_width, target_height, scaling_factor
    )
    
    # Blend images efficiently
    alpha = 0.2
    annotated_image = cv2.addWeighted(full_image, 1 - alpha, mask_overlay, alpha, 0)
    
    # === ADD TUMOR OVERLAY ===
    if tumour_mask_fullres is not None:
        from visualization import add_optimized_tumor_overlay
        annotated_image = add_optimized_tumor_overlay(annotated_image, tumour_mask_fullres, scaling_factor, mpp)
    
    return annotated_image


def create_distance_colored_mask_overlay_with_distances(full_mask, adipocyte_ids, min_dists, target_width, target_height, scaling_factor):
    """Create mask overlay with adipocytes colored by pre-computed distances using reverse jet colormap (from BIBLE)."""
    import matplotlib.pyplot as plt
    
    # Resize mask efficiently
    if scaling_factor != 1.0:
        full_mask_resized = cv2.resize(
            full_mask.astype(np.int32), 
            (target_width, target_height), 
            interpolation=cv2.INTER_NEAREST
        )
    else:
        full_mask_resized = full_mask
    
    # Convert distances to array and handle None values
    valid_ids = np.array(adipocyte_ids)
    distances = np.array([d if d is not None else 0 for d in min_dists])
    
    # Log distance statistics for debugging
    valid_distances = distances[distances > 0]
    if len(valid_distances) > 0:
        logging.info(f"Distance range: {valid_distances.min():.1f} - {valid_distances.max():.1f} µm")
        logging.info(f"Mean distance: {valid_distances.mean():.1f} µm")
        logging.info(f"Median distance: {np.median(valid_distances):.1f} µm")
        logging.info(f"Adipocytes with valid distances: {len(valid_distances)}/{len(distances)}")
        
        # Show distribution of distances
        unique_distances = len(np.unique(valid_distances))
        logging.info(f"Unique distance values: {unique_distances}")
    else:
        logging.warning("No valid distances found for distance coloring")
        # Return a basic colored overlay if no distances available
        return create_fallback_colored_overlay(full_mask_resized, valid_ids)
    
    # Normalize distances for colormap
    # Use adaptive max distance based on actual data, but cap at reasonable values
    if len(valid_distances) > 0:
        # Use 95th percentile as max to avoid outliers skewing the colormap
        max_distance = min(np.percentile(valid_distances, 95), 2000.0)
        max_distance = max(max_distance, 100.0)  # Ensure minimum range
        
        # If all distances are very similar, expand the range slightly
        distance_range = valid_distances.max() - valid_distances.min()
        if distance_range < 10.0:  # If range is less than 10 microns
            max_distance = valid_distances.max() + 50.0  # Add some range
    else:
        max_distance = 1000.0  # Default fallback
    
    # Normalize distances (0 = close, 1 = far)
    distances_norm = np.clip(distances / max_distance, 0, 1)
    
    # Use reverse jet colormap as requested (red = close, blue = far)
    colormap = plt.get_cmap('jet_r')  # reverse jet: red=close(0), blue=far(1)
    colors = (colormap(distances_norm)[:, :3] * 255).astype(np.uint8)
    
    # Log colormap statistics
    unique_colors = len(np.unique(colors.view(np.void), axis=0))
    logging.info(f"Generated {unique_colors} unique colors from {len(distances)} adipocytes")
    logging.info(f"Using max distance: {max_distance:.1f} µm for normalization")
    
    # Show some example distance->color mappings for debugging
    if len(valid_distances) > 0:
        sample_indices = np.linspace(0, len(distances)-1, min(5, len(distances)), dtype=int)
        for i in sample_indices:
            logging.info(f"Adipocyte {valid_ids[i]}: {distances[i]:.1f}µm -> color({colors[i][0]}, {colors[i][1]}, {colors[i][2]})")
    
    # Create mapping array for vectorized lookup
    max_label = full_mask_resized.max()
    color_mapping = np.zeros((max_label + 1, 3), dtype=np.uint8)
    
    # Map adipocyte IDs to distance-based colors
    for i, aid in enumerate(valid_ids):
        if aid <= max_label:
            color_mapping[aid] = colors[i]
    
    # Vectorized color assignment
    mask_overlay = color_mapping[full_mask_resized]
    
    return mask_overlay


def create_fallback_colored_overlay(full_mask_resized, valid_ids):
    """Create a basic colored overlay when no valid distances are available."""
    import matplotlib.pyplot as plt
    
    # Use a simple colormap for basic visualization
    colormap = plt.get_cmap('tab10')
    max_label = full_mask_resized.max()
    color_mapping = np.zeros((max_label + 1, 3), dtype=np.uint8)
    
    # Assign random colors to adipocytes
    for i, aid in enumerate(valid_ids):
        if aid <= max_label:
            color = (colormap(i % 10)[:3] * 255).astype(np.uint8)
            color_mapping[aid] = color
    
    mask_overlay = color_mapping[full_mask_resized]
    return mask_overlay


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'configure_tumour_model',
    'optimized_segment_tumour_on_thumbnail', 
    'optimized_compute_adipocyte_distance_metrics',
    'optimized_compute_adipocyte_distances',
    'save_tumor_csv',
    'save_tumor_thumbnail_overlay',
    'save_distance_colored_image'
]
