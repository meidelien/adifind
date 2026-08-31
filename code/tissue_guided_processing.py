#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tissue-Guided Processing Module
==============================

Implements tissue-guided window generation that significantly reduces processing area
by targeting only tissue-containing regions. This is the key functionality that makes
the "tissue guidance" feature actually work.

Based on the BIBLE implementation that reduces processing from full image to tissue regions only.
"""

import os
import gc
import logging
import time
import numpy as np
import cv2
import torch
from PIL import Image

# Import from other modules
from config import config
from models import configure_tissue_model
from system_utils import monitor


class TissueGuidanceDetector:
    """Tissue detection using dedicated tissue guidance model."""
    
    def __init__(self, tissue_model_dir=None):
        """Initialize tissue guidance detector."""
        self.tissue_model_dir = tissue_model_dir
        self.tissue_predictor = None
        
        # Load tissue model (auto-downloads from HuggingFace if no local path)
        try:
            if tissue_model_dir and os.path.exists(tissue_model_dir):
                logging.info(f"Loading tissue guidance model from: {tissue_model_dir}")
                self.tissue_predictor = configure_tissue_model(tissue_model_dir)
            else:
                logging.info("Auto-loading tissue guidance model...")
                self.tissue_predictor = configure_tissue_model()
            logging.info("? Tissue guidance model loaded successfully")
        except Exception as e:
            logging.error(f"? Failed to load tissue guidance model: {e}")
            self.tissue_predictor = None
    
    def detect_tissue_regions_rough(self, image_handler):
        """
        Detect tissue regions using rough guidance approach.
        This is the KEY function that identifies where tissue is located.
        
        Args:
            image_handler: ImageHandler object
            
        Returns:
            tuple: (tissue_regions, tissue_info)
                - tissue_regions: List of (x, y, w, h) bounding boxes for tissue
                - tissue_info: Dictionary with detection statistics
        """
        if not self.tissue_predictor:
            logging.warning("No tissue predictor available, using full image")
            # Fallback: treat entire image as tissue
            return [(0, 0, image_handler.width, image_handler.height)], {"num_detections": 0}
        
        logging.info("Detecting tissue regions for guidance...")
        
        # Generate tissue thumbnail for faster processing
        thumbnail, scaling_factor = self.generate_tissue_thumbnail(image_handler)
        
        # Run tissue detection on thumbnail
        tissue_detections = self._run_tissue_detection(thumbnail)
        
        # Scale detections back to full resolution
        tissue_regions = []
        for detection in tissue_detections:
            # Scale coordinates back to full image
            x = int(detection['x'] / scaling_factor)
            y = int(detection['y'] / scaling_factor)
            w = int(detection['w'] / scaling_factor)
            h = int(detection['h'] / scaling_factor)
            
            # Ensure regions are within image bounds
            x = max(0, min(x, image_handler.width))
            y = max(0, min(y, image_handler.height))
            w = min(w, image_handler.width - x)
            h = min(h, image_handler.height - y)
            
            if w > 0 and h > 0:  # Valid region
                tissue_regions.append((x, y, w, h))
        
        tissue_info = {
            "num_detections": len(tissue_detections),
            "scaling_factor": scaling_factor,
            "thumbnail_size": thumbnail.shape[:2]
        }
        
        logging.info(f"Tissue detection complete: {len(tissue_regions)} regions found")
        return tissue_regions, tissue_info
    
    def generate_tissue_thumbnail(self, image_handler):
        """
        Generate downscaled thumbnail for tissue detection.
        Uses efficient pyramid level selection.
        """
        # Target thumbnail size for tissue detection
        target_max_dimension = config.TISSUE_THUMBNAIL_SIZE
        
        # Calculate scaling factor
        max_dimension = max(image_handler.width, image_handler.height)
        scaling_factor = min(1.0, target_max_dimension / max_dimension)
        
        target_width = int(image_handler.width * scaling_factor)
        target_height = int(image_handler.height * scaling_factor)
        
        # Read optimal resolution image
        thumbnail = self._read_optimal_image(image_handler, target_width, target_height, scaling_factor)
        
        logging.info(f"Generated tissue thumbnail: {thumbnail.shape} (scale: {scaling_factor:.3f})")
        return thumbnail, scaling_factor
    
    def _read_optimal_image(self, image_handler, target_width, target_height, scaling_factor):
        """Read image at optimal resolution level."""
        # Find the best level that's closest to our target resolution
        if hasattr(image_handler, 'slide') and hasattr(image_handler.slide, 'level_downsamples'):
            downsamples = image_handler.slide.level_downsamples
            level_factors = [1 / ds for ds in downsamples]
            level_diffs = [abs(factor - scaling_factor) for factor in level_factors]
            best_level = level_diffs.index(min(level_diffs))
            level_dims = image_handler.slide.level_dimensions[best_level]
            
            # Read at optimal level
            full_image_pil = image_handler.read_region((0, 0), best_level, level_dims)
            
            # Resize if necessary
            if level_dims != (target_width, target_height):
                full_image_pil = full_image_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            # Fallback for non-slide images
            full_image_pil = image_handler.read_region((0, 0), 0, (image_handler.width, image_handler.height))
            full_image_pil = full_image_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        return np.array(full_image_pil, dtype=np.uint8)[:, :, :3]
    
    def _run_tissue_detection(self, thumbnail):
        """Run tissue detection on thumbnail image."""
        try:
            # Run inference on thumbnail
            outputs = self.tissue_predictor([thumbnail])
            output = outputs[0]
            instances = output["instances"]
            
            if len(instances) == 0:
                logging.warning("No tissue detected in thumbnail")
                return []
            
            # Extract detection data
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            
            # Convert to detection format
            detections = []
            for i, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = box
                detection = {
                    'x': int(x1),
                    'y': int(y1),
                    'w': int(x2 - x1),
                    'h': int(y2 - y1),
                    'score': float(score)
                }
                detections.append(detection)
            
            logging.info(f"Detected {len(detections)} tissue regions (avg score: {np.mean(scores):.3f})")
            return detections
            
        except Exception as e:
            logging.error(f"? Error in tissue detection: {e}")
            return []


class TissueGuidedWindowGenerator:
    """Generate processing windows guided by tissue detection."""
    
    def __init__(self, tissue_detector):
        """Initialize window generator with tissue detector."""
        self.tissue_detector = tissue_detector
    
    def generate_tissue_guided_windows(self, image_handler, window_size, stride):
        """
        Generate windows that target tissue regions using rough guidance.
        This is the CORE function that reduces processing area.
        
        Args:
            image_handler: ImageHandler object
            window_size: (width, height) of processing windows
            stride: (stride_x, stride_y) for window generation
            
        Returns:
            tuple: (guided_windows, tissue_regions, processing_stats)
        """
        logging.info("Generating tissue-guided processing windows...")
        
        # Step 1: Get rough tissue regions
        tissue_regions, tissue_info = self.tissue_detector.detect_tissue_regions_rough(image_handler)
        
        # Step 2: Generate windows that overlap with tissue regions
        window_width, window_height = window_size
        stride_x, stride_y = stride
        
        guided_windows = []
        
        for tissue_x, tissue_y, tissue_w, tissue_h in tissue_regions:
            # Generate windows within this tissue region with some overlap
            region_x_start = max(0, tissue_x - window_width // 2)
            region_y_start = max(0, tissue_y - window_height // 2)
            region_x_end = min(image_handler.width, tissue_x + tissue_w + window_width // 2)
            region_y_end = min(image_handler.height, tissue_y + tissue_h + window_height // 2)
            
            # Generate sliding windows within this tissue region
            x_positions = list(range(region_x_start, region_x_end - window_width + 1, stride_x))
            y_positions = list(range(region_y_start, region_y_end - window_height + 1, stride_y))
            
            # Ensure coverage at region boundaries
            if x_positions and x_positions[-1] + window_width < region_x_end:
                x_positions.append(region_x_end - window_width)
            if y_positions and y_positions[-1] + window_height < region_y_end:
                y_positions.append(region_y_end - window_height)
            
            # Add windows for this tissue region
            for y in y_positions:
                for x in x_positions:
                    # Ensure window is within image bounds
                    if (x >= 0 and y >= 0 and 
                        x + window_width <= image_handler.width and 
                        y + window_height <= image_handler.height):
                        guided_windows.append((x, y))
        
        # Remove duplicates (in case tissue regions overlap)
        guided_windows = list(set(guided_windows))
        
        # Calculate what full processing would have been
        full_x_positions = list(range(0, image_handler.width - window_width + 1, stride_x))
        full_y_positions = list(range(0, image_handler.height - window_height + 1, stride_y))
        if full_x_positions and full_x_positions[-1] + window_width < image_handler.width:
            full_x_positions.append(image_handler.width - window_width)
        if full_y_positions and full_y_positions[-1] + window_height < image_handler.height:
            full_y_positions.append(image_handler.height - window_height)
        
        total_possible_windows = len(full_x_positions) * len(full_y_positions)
        
        # Step 3: Calculate processing statistics
        processing_stats = {
            "total_possible_windows": total_possible_windows,
            "tissue_guided_windows": len(guided_windows),
            "speedup_factor": total_possible_windows / len(guided_windows) if len(guided_windows) > 0 else 1.0,
            "num_tissue_regions": len(tissue_regions),
            "tissue_detections": tissue_info["num_detections"]
        }
        
        logging.info(f"Tissue guidance analysis complete:")
        logging.info(f"   . Total possible windows: {total_possible_windows:,}")
        logging.info(f"   . Tissue-guided windows: {len(guided_windows):,}")
        logging.info(f"   . Speedup factor: {processing_stats['speedup_factor']:.1f}x")
        logging.info(f"   . Tissue regions found: {len(tissue_regions)}")
        
        return guided_windows, tissue_regions, processing_stats


def process_with_tissue_guidance(image_handler, predictor, window_size, stride, 
                               min_area_threshold_pixels, max_area_threshold_pixels, 
                               output_dir, tissue_model_dir=None):
    """
    Main function to process image with tissue guidance.
    This replaces the standard process_all_windows when tissue guidance is enabled.
    
    Args:
        image_handler: ImageHandler object
        predictor: Configured Detectron2 predictor for adipocyte detection
        window_size: (width, height) tuple for processing windows
        stride: (stride_x, stride_y) tuple for window overlap
        min_area_threshold_pixels: Minimum adipocyte area in pixels
        max_area_threshold_pixels: Maximum adipocyte area in pixels
        output_dir: Directory for saving results
        tissue_model_dir: Path to tissue guidance model
        
    Returns:
        tuple: Same as process_all_windows but with tissue guidance stats
    """
    # Initialize tissue guidance (auto-downloads model if needed)
    if True:
        try:
            logging.info("Initializing tissue-guided processing...")
            tissue_detector = TissueGuidanceDetector(tissue_model_dir)
            window_generator = TissueGuidedWindowGenerator(tissue_detector)
            
            # Generate tissue-guided windows
            guided_windows, tissue_regions, stats = window_generator.generate_tissue_guided_windows(
                image_handler, window_size, stride
            )

            # Optional: save low-res preview of tissue-guided windows
            if getattr(config, 'SAVE_TISSUE_WINDOW_GRID_THUMBNAIL', False):
                try:
                    from pathlib import Path
                    from visualization import save_tissue_window_grid_thumbnail
                    image_name = Path(getattr(image_handler, 'file_path', 'image')).stem
                    max_dim = getattr(config, 'TISSUE_THUMBNAIL_SIZE', 2048)
                    save_tissue_window_grid_thumbnail(
                        image_handler=image_handler,
                        guided_windows=guided_windows,
                        window_size=window_size,
                        tissue_regions=tissue_regions,
                        output_dir=output_dir,
                        image_name=image_name,
                        max_dim=max_dim
                    )
                except Exception as e:
                    logging.warning(f"Failed to save tissue window grid thumbnail: {e}")
            
            # If tissue guidance produces windows, use them
            if len(guided_windows) > 0:
                return process_guided_windows(
                    image_handler, predictor, guided_windows, window_size,
                    min_area_threshold_pixels, max_area_threshold_pixels,
                    output_dir, tissue_regions, stats
                )
            else:
                logging.warning("No tissue-guided windows generated. Falling back to full processing.")
                
        except Exception as e:
            logging.warning(f"Tissue guidance failed: {e}. Falling back to full processing.")
    
    # Fallback to original processing
    logging.info("Using original full-image processing...")
    from core_processing import process_all_windows
    return process_all_windows(
        image_handler, predictor, window_size, stride,
        min_area_threshold_pixels, max_area_threshold_pixels, output_dir
    )


def process_guided_windows(image_handler, predictor, guided_windows, window_size,
                         min_area_threshold_pixels, max_area_threshold_pixels,
                         output_dir, tissue_regions, stats):
    """
    Process tissue-guided windows using existing pipeline logic.
    This is similar to process_all_windows but only processes guided windows.
    Enhanced with tissue-guidance specific profiling.
    """
    import concurrent.futures
    import threading
    import time
    from skimage.measure import regionprops_table
    from core_processing import inference_worker, apply_label_mapping_memory_efficient, find, create_mask_array
    from progress_utils import ColorChangingTqdm
    
    tissue_processing_start = time.time()
    logging.info(f"Processing {len(guided_windows):,} tissue-guided windows...")
    logging.info(f"Tissue Guidance Stats:")
    logging.info(f"   . Speedup factor: {stats['speedup_factor']:.1f}x")
    logging.info(f"   . Tissue regions: {stats['num_tissue_regions']}")
    logging.info(f"   . Guidance efficiency: {len(guided_windows)/stats['total_possible_windows']*100:.1f}% (guided/total windows)")
    
    # Initialize variables (identical to original)
    width, height = image_handler.width, image_handler.height
    
    # Use memory-mapped mask if low-memory mode enabled
    full_mask, mask_cleanup = create_mask_array(height, width, output_dir)
    
    parent = {}
    adipocyte_counter = [1]
    lock = threading.Lock()
    
    # Collect post-processed windows and coordinates
    postprocessed_windows = []
    window_coords = []
    processing_counter = [0]
    
    # Prepare worker arguments
    worker_args = []
    for x, y in guided_windows:
        worker_args.append((
            image_handler, x, y, window_size[0], window_size[1],
            min_area_threshold_pixels, max_area_threshold_pixels,
            adipocyte_counter, parent, full_mask, output_dir, lock
        ))
    
    # Process windows in parallel
    inference_start = time.time()
    max_workers = config.MAX_IO_WORKERS
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(ColorChangingTqdm(
            executor.map(lambda args: inference_worker(args, predictor), worker_args),
            total=len(guided_windows),
            desc="Processing Tissue-Guided Windows"
        ))
    print()  # Add line break after progress bar
    inference_time = time.time() - inference_start
    
    logging.info(f"Tissue-guided inference completed in {inference_time:.3f}s")
    logging.info(f"   . Windows per second: {len(guided_windows)/inference_time:.1f}")
    
    # Post-processing phase with detailed profiling
    postprocess_start = time.time()
    logging.info("Starting post-processing: Union-find merging, area filtering, and relabeling...")
    
    # Apply union-find mapping (same as original)
    union_find_start = time.time()
    unique_labels = np.unique(full_mask)
    unique_labels = unique_labels[unique_labels != 0]
    
    if config.ENABLE_PROFILING:
        logging.info(f"STEP 1: Union-find processing {len(unique_labels):,} labels...")
    else:
        logging.info(f"Merging overlapping detections ({len(unique_labels):,} initial labels)...")
    
    if len(unique_labels) > 0:
        label_mapping_cc = {}
        for label in unique_labels:
            root_label = find(parent, label)
            label_mapping_cc[label] = root_label
        
        full_mask = apply_label_mapping_memory_efficient(full_mask, label_mapping_cc)
        del label_mapping_cc
    
    union_find_time = time.time() - union_find_start
    
    if config.ENABLE_PROFILING:
        logging.info(f"Union-find completed in {union_find_time:.3f}s")
    
    # Apply area thresholds and relabel (same as original)
    if config.ENABLE_PROFILING:
        logging.info("STEP 2: Applying area thresholds and filtering...")
    else:
        logging.info("Filtering adipocytes by size criteria...")
        
    area_filter_start = time.time()
    props_table = regionprops_table(full_mask, properties=['label', 'area'])
    
    valid_mask = (
        (props_table['area'] >= min_area_threshold_pixels)
        & (props_table['area'] <= max_area_threshold_pixels)
    )
    valid_labels = props_table['label'][valid_mask]
    valid_labels_set = set(valid_labels)
    
    # Remove invalid adipocytes using memory-efficient approach
    # When max_label is very large (billions), creating dense lookup arrays fails
    max_label = full_mask.max()
    chunk_size = getattr(config, 'MASK_CHUNK_SIZE', 2048)
    
    # Use chunked filtering if lookup table would exceed 1GB
    lookup_size_gb = (max_label + 1) * 4 / (1024 ** 3)  # uint32 = 4 bytes
    
    if lookup_size_gb > 1.0:
        logging.info(f"Using chunked area filtering (max_label={max_label:,}, would need {lookup_size_gb:.1f}GB lookup)")
        height = full_mask.shape[0]
        for start_row in range(0, height, chunk_size):
            end_row = min(start_row + chunk_size, height)
            chunk = full_mask[start_row:end_row, :]
            
            # Use local lookup table instead of np.isin (which creates huge bool arrays)
            chunk_unique = np.unique(chunk)
            chunk_max = chunk_unique.max() if len(chunk_unique) > 0 else 0
            
            if chunk_max > 0:
                # Build local lookup: valid labels map to themselves, invalid to 0
                local_lookup = np.zeros(chunk_max + 1, dtype=np.uint32)
                for label in chunk_unique:
                    if label in valid_labels_set:
                        local_lookup[label] = label
                # Apply lookup to zero out invalid labels
                full_mask[start_row:end_row, :] = local_lookup[chunk]
    else:
        # Standard lookup table approach (fast when max_label is reasonable)
        mapping = np.zeros(max_label + 1, dtype=np.uint32)
        mapping[valid_labels] = valid_labels
        full_mask[:] = mapping[full_mask]
    
    area_filter_time = time.time() - area_filter_start
    
    if config.ENABLE_PROFILING:
        logging.info(f"Area filtering completed in {area_filter_time:.3f}s - {len(valid_labels):,} valid adipocytes")
        logging.info("STEP 3: Relabeling adipocytes with consecutive IDs...")
    else:
        logging.info(f"Size filtering complete: {len(valid_labels):,} valid adipocytes found")
        logging.info("Finalizing adipocyte labels...")
    
    # Relabel consecutively
    relabel_start = time.time()
    unique_final_labels = np.unique(full_mask)
    unique_final_labels = unique_final_labels[unique_final_labels != 0]
    
    if len(unique_final_labels) > 0:
        max_label = full_mask.max()
        lookup_size_gb = (max_label + 1) * 4 / (1024 ** 3)
        
        # Build old->new mapping dict
        relabel_dict = {old_id: new_id for new_id, old_id in enumerate(unique_final_labels, start=1)}
        
        if lookup_size_gb > 1.0:
            logging.info(f"Using chunked relabeling (max_label={max_label:,}, would need {lookup_size_gb:.1f}GB lookup)")
            # Process in chunks using dict-based remapping
            height = full_mask.shape[0]
            chunk_size = getattr(config, 'MASK_CHUNK_SIZE', 2048)
            for start_row in range(0, height, chunk_size):
                end_row = min(start_row + chunk_size, height)
                chunk = full_mask[start_row:end_row, :]
                # Get unique labels in this chunk and build local lookup
                chunk_unique = np.unique(chunk)
                chunk_max = chunk_unique.max() if len(chunk_unique) > 0 else 0
                if chunk_max > 0:
                    local_lookup = np.zeros(chunk_max + 1, dtype=np.uint32)
                    for old_id in chunk_unique:
                        if old_id in relabel_dict:
                            local_lookup[old_id] = relabel_dict[old_id]
                    full_mask[start_row:end_row, :] = local_lookup[chunk]
        else:
            # Standard lookup table approach
            relabel_mapping = np.zeros(max_label + 1, dtype=np.uint32)
            for old_id, new_id in relabel_dict.items():
                relabel_mapping[old_id] = new_id
            full_mask = relabel_mapping[full_mask]
        
        # Include centroid and bbox for downstream use (avoids redundant regionprops scans)
        props_table = regionprops_table(full_mask, properties=['label', 'area', 'centroid', 'bbox'])
    
    relabel_time = time.time() - relabel_start
    
    if config.ENABLE_PROFILING:
        logging.info(f"Relabeling completed in {relabel_time:.3f}s")
        logging.info("STEP 4: Creating final data structures...")
    else:
        logging.info("Creating final adipocyte data...")
    
    # Build final data structures including properties for downstream use
    final_data_start = time.time()
    mask_areas = {}
    adipocyte_ids = []
    final_properties = {}  # Populate for downstream use (avoids redundant regionprops scans)
    
    for i, label in enumerate(props_table['label']):
        area = props_table['area'][i]
        mask_areas[label] = area
        adipocyte_ids.append(label)
        # Build final_properties dict with centroid and bbox
        final_properties[label] = {
            'area': area,
            'centroid_x': props_table['centroid-1'][i],  # centroid returns (row, col)
            'centroid_y': props_table['centroid-0'][i],
            'bbox': (
                props_table['bbox-0'][i],  # min_row
                props_table['bbox-1'][i],  # min_col
                props_table['bbox-2'][i],  # max_row
                props_table['bbox-3'][i],  # max_col
            )
        }
    
    logging.info(f"Built final_properties for {len(final_properties)} adipocytes")
    final_data_time = time.time() - final_data_start
    postprocess_time = time.time() - postprocess_start
    total_tissue_time = time.time() - tissue_processing_start
    
    # Post-processing completion message
    logging.info(f"? Post-processing complete: {len(adipocyte_ids):,} final adipocytes")
    
    # Comprehensive tissue-guided profiling summary (only if profiling enabled)
    if config.ENABLE_PROFILING:
        logging.info("TISSUE-GUIDED PROCESSING PROFILING:")
    logging.info("=" * 60)
    logging.info(f"Inference Phase:          {inference_time:8.3f}s ({inference_time/total_tissue_time*100:5.1f}%)")
    logging.info(f"Union-Find Phase:         {union_find_time:8.3f}s ({union_find_time/total_tissue_time*100:5.1f}%)")  
    logging.info(f"\uFFFD Final Data Creation:      {final_data_time:8.3f}s ({final_data_time/total_tissue_time*100:5.1f}%)")
    logging.info(f"\uFFFD\uD83D\uDCCA Total Post-Processing:    {postprocess_time:8.3f}s ({postprocess_time/total_tissue_time*100:5.1f}%)")
    logging.info("=" * 60)
    logging.info(f"TOTAL TISSUE-GUIDED TIME: {total_tissue_time:8.3f}s")
    logging.info(f"Tissue-guided processing complete:")
    logging.info(f"   . Windows processed: {len(guided_windows):,}")
    logging.info(f"   . Speedup achieved: {stats['speedup_factor']:.1f}x")
    logging.info(f"   . Final adipocytes: {len(adipocyte_ids):,}")
    logging.info(f"   . Processing efficiency: {len(adipocyte_ids)/len(guided_windows):.2f} adipocytes/window")
    
    # Compare against theoretical full processing
    theoretical_full_time = total_tissue_time * stats['speedup_factor']
    logging.info(f"? Estimated time savings: {theoretical_full_time - total_tissue_time:.1f}s ({(1 - total_tissue_time/theoretical_full_time)*100:.1f}% faster)")
    
    # Return final_properties to avoid redundant regionprops scans in visualization
    # Include mask_cleanup so callers can remove memmap files after use
    return full_mask, mask_areas, adipocyte_ids, postprocessed_windows, window_coords, final_properties, mask_cleanup


# ================================================================
# STANDALONE TISSUE GUIDANCE FUNCTIONS
# ================================================================
# (Merged from tissue_guidance.py — used by main.py for composable
#  tissue filtering on pre-generated window lists)

def detect_tissue_regions(image_handler, output_dir):
    """
    Detect tissue regions in the whole slide image using tissue guidance model.
    
    Args:
        image_handler: ImageHandler object for slide access
        output_dir: Directory for saving tissue detection results
        
    Returns:
        tuple: (tissue_mask, tissue_regions) where tissue_mask is binary mask
               and tissue_regions is list of bounding boxes
    """
    logging.debug("\U0001f52c Starting tissue region detection...")
    
    try:
        # Configure tissue detection model
        tissue_predictor = configure_tissue_model()
        if tissue_predictor is None:
            logging.warning("Tissue guidance model not available, skipping tissue detection")
            return None, []
        
        # Get thumbnail for tissue detection
        thumbnail_level = image_handler.get_best_level_for_downsample(config.TISSUE_DETECTION_DOWNSAMPLE)
        thumbnail = image_handler.read_region((0, 0), thumbnail_level, image_handler.slide.level_dimensions[thumbnail_level])
        thumbnail = thumbnail.convert("RGB")
        thumbnail_np = np.array(thumbnail)
        
        logging.debug(f"Tissue detection on thumbnail: {thumbnail_np.shape}")
        
        # Perform tissue detection
        monitor.log_gpu_status("Starting tissue detection")
        outputs = tissue_predictor([thumbnail_np])
        output = outputs[0]

        # Free tissue model from GPU — only the adipocyte model should remain
        del tissue_predictor
        gc.collect()
        torch.cuda.empty_cache()
        monitor.log_gpu_status("Tissue model freed from GPU")
        
        if "instances" not in output:
            logging.warning("No instances found in tissue detection output")
            return None, []
        
        instances = output["instances"]
        
        if len(instances) == 0:
            logging.warning("No tissue regions detected")
            return None, []
        
        # Process tissue detections
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        
        # Filter by confidence threshold
        valid_detections = scores > config.TISSUE_CONFIDENCE_THRESHOLD
        boxes = boxes[valid_detections]
        scores = scores[valid_detections]
        
        if len(boxes) == 0:
            logging.warning("No tissue regions passed confidence threshold")
            return None, []
        
        # Create tissue mask
        tissue_mask = np.zeros(thumbnail_np.shape[:2], dtype=np.uint8)
        
        # Fill tissue regions in mask
        tissue_regions = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(thumbnail_np.shape[1], x2)
            y2 = min(thumbnail_np.shape[0], y2)
            
            # Fill tissue region in mask
            tissue_mask[y1:y2, x1:x2] = 255
            
            # Scale coordinates back to full resolution
            scale_factor = image_handler.slide.level_downsamples[thumbnail_level]
            full_res_box = [
                int(x1 * scale_factor),
                int(y1 * scale_factor),
                int(x2 * scale_factor),
                int(y2 * scale_factor)
            ]
            tissue_regions.append(full_res_box)
        
        # Save tissue detection results
        if config.SAVE_TISSUE_DETECTION:
            tissue_vis = thumbnail_np.copy()
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(tissue_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            tissue_path = os.path.join(output_dir, "tissue_detection.png")
            cv2.imwrite(tissue_path, cv2.cvtColor(tissue_vis, cv2.COLOR_RGB2BGR))
            
            mask_path = os.path.join(output_dir, "tissue_mask.png")
            cv2.imwrite(mask_path, tissue_mask)
            
            logging.info(f"\u2705 Tissue detection saved: {tissue_path}")
            logging.info(f"\u2705 Tissue mask saved: {mask_path}")
        
        logging.info(f"\U0001f9ec Detected {len(tissue_regions)} tissue regions")
        monitor.log_gpu_status("Completed tissue detection")
        
        return tissue_mask, tissue_regions
        
    except Exception as e:
        logging.error(f"Error in tissue detection: {e}")
        monitor.log_gpu_status("Tissue detection error")
        return None, []


def filter_windows_by_tissue(window_coords, tissue_regions, window_size):
    """
    Filter processing windows to only include those overlapping with tissue regions.
    
    Args:
        window_coords: List of (x, y) coordinates for processing windows
        tissue_regions: List of tissue bounding boxes [x1, y1, x2, y2]
        window_size: (width, height) tuple for window dimensions
        
    Returns:
        list: Filtered list of window coordinates that overlap with tissue
    """
    if not tissue_regions:
        logging.info("No tissue regions provided, processing all windows")
        return window_coords
    
    logging.info("\U0001f3af Filtering windows by tissue regions...")
    
    filtered_windows = []
    window_width, window_height = window_size
    
    for x, y in window_coords:
        window_box = [x, y, x + window_width, y + window_height]
        
        # Check if window overlaps with any tissue region
        overlaps_tissue = False
        for tissue_box in tissue_regions:
            if _boxes_overlap(window_box, tissue_box):
                overlaps_tissue = True
                break
        
        if overlaps_tissue:
            filtered_windows.append((x, y))
    
    reduction_ratio = len(filtered_windows) / len(window_coords) if window_coords else 0
    logging.info(f"\U0001f4ca Tissue filtering: {len(filtered_windows)}/{len(window_coords)} windows retained ({reduction_ratio:.1%})")
    
    return filtered_windows


def _boxes_overlap(box1, box2):
    """
    Check if two bounding boxes overlap.
    
    Args:
        box1: [x1, y1, x2, y2] coordinates
        box2: [x1, y1, x2, y2] coordinates
        
    Returns:
        bool: True if boxes overlap, False otherwise
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Check if boxes do NOT overlap
    if x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1:
        return False
    
    return True


def calculate_tissue_coverage(tissue_mask, window_coords, window_size, thumbnail_scale):
    """
    Calculate tissue coverage percentage for each processing window.
    """
    logging.info("\U0001f4ca Calculating tissue coverage for windows...")
    
    coverage_map = {}
    window_width, window_height = window_size
    
    for x, y in window_coords:
        thumb_x = int(x / thumbnail_scale)
        thumb_y = int(y / thumbnail_scale)
        thumb_w = int(window_width / thumbnail_scale)
        thumb_h = int(window_height / thumbnail_scale)
        
        window_region = tissue_mask[thumb_y:thumb_y+thumb_h, thumb_x:thumb_x+thumb_w]
        
        if window_region.size > 0:
            tissue_pixels = np.sum(window_region > 0)
            total_pixels = window_region.size
            coverage = tissue_pixels / total_pixels
        else:
            coverage = 0.0
        
        coverage_map[(x, y)] = coverage
    
    return coverage_map


def prioritize_windows_by_tissue(window_coords, tissue_coverage_map, min_coverage=0.1):
    """
    Prioritize processing windows based on tissue coverage.
    """
    logging.info(f"\u2b50 Prioritizing windows by tissue coverage (min: {min_coverage:.1%})...")
    
    high_priority = []
    low_priority = []
    
    for coords in window_coords:
        coverage = tissue_coverage_map.get(coords, 0.0)
        
        if coverage >= min_coverage:
            high_priority.append(coords)
        else:
            low_priority.append(coords)
    
    high_priority.sort(key=lambda coords: tissue_coverage_map.get(coords, 0.0), reverse=True)
    
    logging.info(f"\U0001f4ca Window prioritization: {len(high_priority)} high priority, {len(low_priority)} low priority")
    
    return high_priority, low_priority


def apply_tissue_guidance(image_handler, window_coords, window_size, output_dir):
    """
    Apply complete tissue guidance pipeline to optimize processing.
    
    Args:
        image_handler: ImageHandler object for slide access
        window_coords: List of (x, y) coordinates for processing windows
        window_size: (width, height) tuple for window dimensions
        output_dir: Directory for saving guidance results
        
    Returns:
        tuple: (filtered_windows, tissue_coverage_map) for optimized processing
    """
    if not config.ENABLE_TISSUE_GUIDANCE:
        logging.info("Tissue guidance disabled, processing all windows")
        return window_coords, {}
    
    logging.info("\U0001f9ec Applying tissue guidance pipeline...")
    
    try:
        # Step 1: Detect tissue regions
        tissue_mask, tissue_regions = detect_tissue_regions(image_handler, output_dir)
        
        if tissue_mask is None or not tissue_regions:
            logging.warning("No tissue regions detected, processing all windows")
            return window_coords, {}
        
        # Step 2: Filter windows by tissue overlap
        filtered_windows = filter_windows_by_tissue(window_coords, tissue_regions, window_size)
        
        if not filtered_windows:
            logging.warning("No windows overlap with tissue, processing all windows")
            return window_coords, {}
        
        # Step 3: Calculate tissue coverage for remaining windows
        thumbnail_level = image_handler.get_best_level_for_downsample(config.TISSUE_DETECTION_DOWNSAMPLE)
        thumbnail_scale = image_handler.slide.level_downsamples[thumbnail_level]
        
        coverage_map = calculate_tissue_coverage(tissue_mask, filtered_windows, window_size, thumbnail_scale)
        
        # Step 4: Prioritize windows by tissue coverage
        high_priority, low_priority = prioritize_windows_by_tissue(
            filtered_windows, coverage_map, config.MIN_TISSUE_COVERAGE
        )
        
        # Combine prioritized windows (high priority first)
        optimized_windows = high_priority + low_priority
        
        reduction_ratio = len(optimized_windows) / len(window_coords) if window_coords else 0
        logging.info(f"\U0001f9ec Tissue guidance complete: {len(optimized_windows)}/{len(window_coords)} windows ({reduction_ratio:.1%})")
        
        return optimized_windows, coverage_map
        
    except Exception as e:
        logging.error(f"Error in tissue guidance: {e}")
        import traceback
        traceback.print_exc()
        logging.warning("Falling back to processing all windows")
        return window_coords, {}


def refine_tissue_mask(tissue_mask, morphology_kernel_size=5):
    """
    Refine tissue mask using morphological operations.
    """
    if tissue_mask is None:
        return None
    
    logging.info("\U0001f9f9 Refining tissue mask...")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel_size, morphology_kernel_size))
    refined_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
    
    return refined_mask


def create_tissue_boundary_overlay(image_handler, tissue_regions, output_dir):
    """
    Create visualization overlay showing detected tissue boundaries.
    """
    if not tissue_regions:
        return
    
    logging.info("\U0001f5bc Creating tissue boundary overlay...")
    
    try:
        thumbnail_level = image_handler.get_best_level_for_downsample(config.TISSUE_DETECTION_DOWNSAMPLE)
        thumbnail = image_handler.read_region((0, 0), thumbnail_level, image_handler.slide.level_dimensions[thumbnail_level])
        thumbnail_np = np.array(thumbnail.convert("RGB"))
        
        scale_factor = image_handler.slide.level_downsamples[thumbnail_level]
        
        overlay = thumbnail_np.copy()
        for region in tissue_regions:
            x1, y1, x2, y2 = region
            
            thumb_x1 = int(x1 / scale_factor)
            thumb_y1 = int(y1 / scale_factor)
            thumb_x2 = int(x2 / scale_factor)
            thumb_y2 = int(y2 / scale_factor)
            
            cv2.rectangle(overlay, (thumb_x1, thumb_y1), (thumb_x2, thumb_y2), (0, 255, 0), 3)
            cv2.putText(overlay, "TISSUE", (thumb_x1, thumb_y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        overlay_path = os.path.join(output_dir, "tissue_boundary_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        logging.info(f"\u2705 Tissue boundary overlay saved: {overlay_path}")
        
    except Exception as e:
        logging.error(f"Error creating tissue boundary overlay: {e}")


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'TissueGuidanceDetector',
    'TissueGuidedWindowGenerator', 
    'process_with_tissue_guidance',
    'process_guided_windows',
    'detect_tissue_regions',
    'filter_windows_by_tissue',
    'calculate_tissue_coverage',
    'prioritize_windows_by_tissue',
    'apply_tissue_guidance',
    'refine_tissue_mask',
    'create_tissue_boundary_overlay',
]
