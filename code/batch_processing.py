#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch processing with summary updates and resume support."""

import os
import json
import csv
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
from dataclasses import dataclass, asdict
from config import config


@dataclass
class ImageProcessingResult:
    """Data class for individual image processing results."""
    image_name: str
    image_path: str
    output_dir: str
    total_adipocytes: int = 0
    num_tumors: int = 0
    total_time: float = 0.0
    processing_status: str = "pending"  # pending, processing, completed, failed
    error_message: str = ""
    median_size: float = 0.0
    average_size: float = 0.0
    area_stats: Dict = None
    distance_stats: Dict = None
    timestamp_start: str = ""
    timestamp_end: str = ""


@dataclass
class BatchState:
    """Data class for batch processing state."""
    batch_id: str
    start_time: str
    base_output_dir: str
    summary_dir: str
    total_images: int
    processed_images: int = 0
    successful_images: int = 0
    failed_images: int = 0
    total_adipocytes: int = 0
    total_tumors: int = 0
    total_processing_time: float = 0.0
    config_snapshot: Dict = None
    args_snapshot: Dict = None
    image_results: List[ImageProcessingResult] = None
    
    def __post_init__(self):
        if self.image_results is None:
            self.image_results = []


class BatchProcessor:
    """
    Advanced batch processing manager with continuous updates and resume capability.
    """
    
    def __init__(self, base_output_dir: str, image_files: List[str], args, resume_from: Optional[str] = None):
        """
        Initialize batch processor.
        
        Args:
            base_output_dir: Base directory for batch output
            image_files: List of image file paths
            args: Command line arguments
            resume_from: Path to previous batch state file to resume from
        """
        self.base_output_dir = base_output_dir
        self.image_files = image_files
        self.args = args
        
        print(f"DEBUG - BatchProcessor.__init__ called with:")
        print(f"DEBUG -   base_output_dir: {base_output_dir}")
        print(f"DEBUG -   image_files count: {len(image_files)}")
        print(f"DEBUG -   resume_from: {resume_from}")
        print(f"DEBUG -   resume_from exists: {os.path.exists(resume_from) if resume_from else 'N/A'}")
        
        # Initialize or resume batch state
        if resume_from and os.path.exists(resume_from):
            print(f"DEBUG - Loading existing batch state from: {resume_from}")
            self.batch_state = self._load_batch_state(resume_from)
            self.batch_id = self.batch_state.batch_id  # Preserve original batch ID
            self.summary_dir = self.batch_state.summary_dir  # Preserve original summary dir
            print(f"Resuming batch from: {resume_from}")
            print(f"Previously processed: {self.batch_state.processed_images}/{self.batch_state.total_images} images")
        else:
            print(f"DEBUG - Creating new batch (resume_from={resume_from}, exists={os.path.exists(resume_from) if resume_from else False})")
            # Create batch ID and summary directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.batch_id = f"batch_{timestamp}"
            self.summary_dir = os.path.join(base_output_dir, "batch_summary")
            os.makedirs(self.summary_dir, exist_ok=True)
            
            self.batch_state = self._initialize_new_batch(args)
            print(f"Starting new batch: {self.batch_id}")
        
        # Paths for persistent files
        self.state_file = os.path.join(self.summary_dir, f"{self.batch_id}_state.json")
        self.summary_csv = os.path.join(self.summary_dir, f"{self.batch_id}_summary.csv")
        self.config_file = os.path.join(self.summary_dir, f"{self.batch_id}_config.json")
        self.progress_log = os.path.join(self.summary_dir, f"{self.batch_id}_progress.log")
        
        # Save initial state (only for new batches)
        if not resume_from:
            self._save_batch_state()
            self._save_config_snapshot()
            self._initialize_summary_csv()
        
    def _initialize_new_batch(self, args) -> BatchState:
        """Initialize a new batch state."""
        from config import config
        
        # Create image results for all files
        # Use dict to ensure no duplicates from the start
        image_results_dict = {}
        date_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for image_path in self.image_files:
            # Skip if already processed (shouldn't happen in new batch, but safety first)
            if image_path in image_results_dict:
                print(f"WARNING - Duplicate image path in input: {image_path}")
                continue
                
            image_name = Path(image_path).stem
            output_dir = os.path.join(self.base_output_dir, f"{image_name}_{date_suffix}")
            
            result = ImageProcessingResult(
                image_name=image_name,
                image_path=image_path,
                output_dir=output_dir
            )
            image_results_dict[image_path] = result
        
        # Convert dict to list
        image_results = list(image_results_dict.values())
        
        return BatchState(
            batch_id=self.batch_id,
            start_time=datetime.now().isoformat(),
            base_output_dir=self.base_output_dir,
            summary_dir=self.summary_dir,
            total_images=len(image_results),
            config_snapshot=self._get_config_snapshot(),
            args_snapshot=vars(args),
            image_results=image_results
        )
    
    def _load_batch_state(self, state_file: str) -> BatchState:
        """Load batch state from file."""
        print(f"Loading batch state from: {state_file}")
        
        # Read using utf-8 and replace invalid bytes to be defensive
        try:
            with open(state_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                print(f"DEBUG - File size: {len(content)} characters")
                
                # Check if the file seems truncated (doesn't end with })
                if not content.strip().endswith('}'):
                    print(f"WARNING - File may be truncated, doesn't end with '}}'. Last 100 chars: {content[-100:]}")
                
                data = json.loads(content)
                print(f"DEBUG - JSON parsed successfully")
                
        except json.JSONDecodeError as e:
            print(f"ERROR - JSON decode error: {e}")
            print(f"DEBUG - Error position: {e.pos if hasattr(e, 'pos') else 'unknown'}")
            raise
        except Exception as e:
            print(f"ERROR - File reading error: {e}")
            raise
        
        print(f"DEBUG - Raw JSON data shows:")
        print(f"DEBUG - total_images: {data.get('total_images', 'NOT FOUND')}")
        print(f"DEBUG - processed_images: {data.get('processed_images', 'NOT FOUND')}")
        print(f"DEBUG - successful_images: {data.get('successful_images', 'NOT FOUND')}")
        print(f"DEBUG - failed_images: {data.get('failed_images', 'NOT FOUND')}")
        print(f"DEBUG - image_results array length IN JSON: {len(data.get('image_results', []))}")
        
        # Check status distribution in raw JSON before processing
        raw_statuses = {}
        failed_paths_from_json = set()
        completed_paths_from_json = set()
        
        image_results_data = data.get('image_results', [])
        print(f"DEBUG - Processing {len(image_results_data)} image results from JSON...")
        print(f"DEBUG - Counting raw statuses in JSON data...")
        
        for i, result_data in enumerate(image_results_data):
            status = result_data.get('processing_status', 'UNKNOWN')
            raw_statuses[status] = raw_statuses.get(status, 0) + 1
            
            # Collect paths of failed/completed images for exclusion
            image_path = result_data.get('image_path', '')
            if status == 'failed':
                failed_paths_from_json.add(image_path)
            elif status == 'completed':
                completed_paths_from_json.add(image_path)
            # Also check for error messages as additional failed indicator
            elif result_data.get('error_message', '').strip():
                failed_paths_from_json.add(image_path)
            
            # Show a few examples for debugging
            if i < 5 or status in ['failed', 'completed']:
                if i < 10:  # Limit output
                    print(f"DEBUG - Raw result {i}: {os.path.basename(image_path)} -> status: {status}")
        
        print(f"DEBUG - Raw JSON status distribution: {raw_statuses}")
        print(f"DEBUG - Failed paths from JSON: {len(failed_paths_from_json)}")
        print(f"DEBUG - Completed paths from JSON: {len(completed_paths_from_json)}")
        
        # Store exclusion lists for later use
        self._failed_paths_from_json = failed_paths_from_json
        self._completed_paths_from_json = completed_paths_from_json
        
        # DEDUPLICATION: Handle duplicate entries for the same image
        # Status priority: completed > failed > processing > pending
        # Keep the entry with the highest priority status
        
        status_priority = {
            'completed': 4,
            'failed': 3,
            'processing': 2,
            'pending': 1
        }
        
        # Deduplicate by image_path, keeping the highest priority status
        dedup_dict = {}
        duplicate_count = 0
        
        for i, result_data in enumerate(image_results_data):
            image_path = result_data.get('image_path', '')
            if not image_path:
                continue
                
            current_status = result_data.get('processing_status', 'pending')
            current_priority = status_priority.get(current_status, 0)
            
            # Check if this path already exists
            if image_path in dedup_dict:
                duplicate_count += 1
                existing_status = dedup_dict[image_path].get('processing_status', 'pending')
                existing_priority = status_priority.get(existing_status, 0)
                
                # Only replace if current has higher priority
                if current_priority > existing_priority:
                    print(f"DEDUP - Replacing {os.path.basename(image_path)}: {existing_status} -> {current_status}")
                    dedup_dict[image_path] = result_data
                else:
                    print(f"DEDUP - Keeping {os.path.basename(image_path)}: {existing_status} (ignoring {current_status})")
            else:
                dedup_dict[image_path] = result_data
        
        print(f"DEBUG - Found {duplicate_count} duplicate entries")
        print(f"DEBUG - Deduplicated to {len(dedup_dict)} unique images")
        
        # Convert deduplicated results back to dataclass objects
        image_results = []
        skipped_count = 0
        status_conversion_issues = 0
        status_distribution_after_dedup = {}
        
        for i, (image_path, result_data) in enumerate(dedup_dict.items()):
            try:
                original_status = result_data.get('processing_status', 'pending')
                
                # Track status distribution
                status_distribution_after_dedup[original_status] = status_distribution_after_dedup.get(original_status, 0) + 1
                
                result = ImageProcessingResult(**result_data)
                
                # Verify status was preserved correctly
                if result.processing_status != original_status:
                    print(f"WARNING - Status changed during loading: {original_status} -> {result.processing_status}")
                    status_conversion_issues += 1
                
                image_results.append(result)
                
                # Show first few for debugging
                if i < 5:
                    print(f"DEBUG - Image {i}: {result.image_name} - Status: {result.processing_status} - Error: '{result.error_message}'")
                    
            except Exception as e:
                print(f"ERROR - Failed to load image result for {os.path.basename(image_path)}: {e}")
                print(f"DEBUG - Problem entry status was: {original_status}")
                import traceback
                traceback.print_exc()
                skipped_count += 1
        
        print(f"DEBUG - Successfully loaded {len(image_results)} image results, skipped {skipped_count}")
        print(f"DEBUG - Status conversion issues: {status_conversion_issues}")
        print(f"DEBUG - Status distribution after dedup and loading:")
        for status, count in sorted(status_distribution_after_dedup.items()):
            print(f"DEBUG -   {status}: {count}")
        
        # STEP 6: Auto-detect and report interrupted images on resume
        interrupted_images = [r for r in image_results if r.processing_status == "processing"]
        if interrupted_images:
            print(f"\nDetected {len(interrupted_images)} interrupted images from previous run:")
            for img in interrupted_images[:5]:  # Show first 5
                print(f"   - {img.image_name}")
            if len(interrupted_images) > 5:
                print(f"   ... and {len(interrupted_images) - 5} more")
            print("   These will be reset to pending and reprocessed.\n")
        
        data['image_results'] = image_results
        return BatchState(**data)
    
    def _save_batch_state(self):
        """Save current batch state to file."""
        # Convert to dict for JSON serialization
        state_dict = asdict(self.batch_state)
        # Write using utf-8 to preserve unicode in any stored fields
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, default=str)
    
    def _get_config_snapshot(self) -> Dict:
        """Get current configuration snapshot."""
        from config import config, paths
        
        return {
            'MIN_ADIPOCYTE_AREA_MICRONS': config.MIN_ADIPOCYTE_AREA_MICRONS,
            'MAX_ADIPOCYTE_AREA_MICRONS': config.MAX_ADIPOCYTE_AREA_MICRONS,
            'GRID_CELL_SIZE_MICRONS': config.GRID_CELL_SIZE_MICRONS,
            'IOU_THRESHOLD': config.IOU_THRESHOLD,
            'MERGE_IOU_THRESHOLD': config.MERGE_IOU_THRESHOLD,
            'CONFIDENCE_THRESHOLD': config.CONFIDENCE_THRESHOLD,
            'SCALING_FACTOR': config.SCALING_FACTOR,
            'DEFAULT_MPP': config.DEFAULT_MPP,
            'ENABLE_TUMOR_SEGMENTATION': config.ENABLE_TUMOR_SEGMENTATION,
            'ENABLE_TISSUE_GUIDANCE': config.ENABLE_TISSUE_GUIDANCE,
            'DEBUG_MODE': config.DEBUG_MODE,
            'DEBUG_SAVE_UNPROCESSED_WINDOWS': getattr(config, 'DEBUG_SAVE_UNPROCESSED_WINDOWS', False),
            'ADIPOCYTE_MODEL_DIR': paths.ADIPOCYTE_MODEL_DIR,
            'ADIPOCYTE_MODEL_CHECKPOINT': paths.ADIPOCYTE_MODEL_CHECKPOINT,
            'TUMOR_MODEL_DIR': paths.TUMOR_MODEL_DIR,
            'TUMOR_MODEL_CHECKPOINT': paths.TUMOR_MODEL_CHECKPOINT,
            'TISSUE_MODEL_DIR': paths.TISSUE_MODEL_DIR,
            'TISSUE_MODEL_CHECKPOINT': paths.TISSUE_MODEL_CHECKPOINT,
        }
    
    def _save_config_snapshot(self):
        """Save configuration snapshot to file."""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.batch_state.config_snapshot, f, indent=2, default=str)
    
    def _initialize_summary_csv(self):
        """Initialize the summary CSV file with headers."""
        if not os.path.exists(self.summary_csv):
            with open(self.summary_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "Image Name", "Status", "Total Adipocytes", 
                    "Median Size (\u00B5m\u00B2)", "Average Size (\u00B5m\u00B2)", 
                    "Tumor Count", "Processing Time (s)", 
                    "Start Time", "End Time", "Error Message"
                ])
    
    def get_pending_images(self, retry_failed: bool = None) -> List[ImageProcessingResult]:
        """Get list of images that haven't been processed yet.
        
        Args:
            retry_failed: If True, include failed images. If None, uses config.RETRY_FAILED_IMAGES
            
        Returns:
            List of ImageProcessingResult objects that should be processed
        """
        
        # Use config setting if retry_failed not explicitly provided
        if retry_failed is None:
            retry_failed = config.RETRY_FAILED_IMAGES
        
        # STEP 1: Verify no duplicates in loaded results
        image_path_counts = {}
        for result in self.batch_state.image_results:
            image_path_counts[result.image_path] = image_path_counts.get(result.image_path, 0) + 1
        
        duplicates = {path: count for path, count in image_path_counts.items() if count > 1}
        if duplicates:
            print(f"WARNING - Found {len(duplicates)} duplicate image paths after loading!")
            for path, count in list(duplicates.items())[:3]:
                print(f"   - {os.path.basename(path)}: {count} occurrences")
        
        # STEP 2: Collect status statistics for debugging
        all_statuses = [result.processing_status for result in self.batch_state.image_results]
        status_counts = {}
        for status in all_statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"DEBUG - Status counts: {status_counts}")
        print(f"DEBUG - Total image_results loaded: {len(self.batch_state.image_results)}")
        print(f"DEBUG - retry_failed parameter: {retry_failed}")
        print(f"DEBUG - RETRY_FAILED_IMAGES config: {config.RETRY_FAILED_IMAGES}")
        
        # STEP 3: Build exclusion sets from cached JSON data
        # Use cached exclusion lists from JSON loading if available
        failed_paths_from_json = getattr(self, '_failed_paths_from_json', set())
        completed_paths_from_json = getattr(self, '_completed_paths_from_json', set())
        
        print(f"DEBUG - Failed paths from JSON cache: {len(failed_paths_from_json)}")
        print(f"DEBUG - Completed paths from JSON cache: {len(completed_paths_from_json)}")
        
        # STEP 4: Build comprehensive exclusion sets
        # Create a set of known failed image paths for safety
        failed_image_paths = set(failed_paths_from_json)  # Start with JSON cache
        completed_image_paths = set(completed_paths_from_json)  # Start with JSON cache
        
        # Collect failed and completed images from loaded results
        for result in self.batch_state.image_results:
            if result.processing_status == "failed":
                failed_image_paths.add(result.image_path)
            elif result.processing_status == "completed":
                completed_image_paths.add(result.image_path)
            # Also check for non-empty error messages as additional safety
            elif result.error_message and result.error_message.strip():
                failed_image_paths.add(result.image_path)
                if result.image_path not in failed_paths_from_json:
                    print(f"DEBUG - Found image with error message: {result.image_name} - {result.error_message}")
        
        print(f"DEBUG - Total failed image paths identified: {len(failed_image_paths)}")
        print(f"DEBUG - Total completed image paths identified: {len(completed_image_paths)}")
        
        # STEP 5: Filter results based on retry_failed setting
        if retry_failed:
            # Include pending, processing (interrupted), and failed images when retrying is enabled
            pending_results = [result for result in self.batch_state.image_results 
                              if result.processing_status in ["pending", "processing", "failed"]]
            print(f"Retry mode: Including {len(pending_results)} images (pending + processing + failed)")
        else:
            # Only include pending and processing (interrupted) images, skip failed and completed ones
            # Use multiple criteria to identify what to skip
            pending_results = []
            excluded_count = 0
            excluded_failed = 0
            excluded_completed = 0
            interrupted_count = 0
            
            for result in self.batch_state.image_results:
                should_exclude = False
                exclude_reason = ""
                
                # Skip if explicitly failed
                if result.processing_status == "failed":
                    should_exclude = True
                    exclude_reason = "STATUS_FAILED"
                    excluded_failed += 1
                # Skip if explicitly completed
                elif result.processing_status == "completed":
                    should_exclude = True
                    exclude_reason = "STATUS_COMPLETED"
                    excluded_completed += 1
                # Skip if in our failed paths set
                elif result.image_path in failed_image_paths:
                    should_exclude = True
                    exclude_reason = "IN_FAILED_PATHS"
                    excluded_failed += 1
                # Skip if in our completed paths set  
                elif result.image_path in completed_image_paths:
                    should_exclude = True
                    exclude_reason = "IN_COMPLETED_PATHS"
                    excluded_completed += 1
                
                if should_exclude:
                    excluded_count += 1
                    if excluded_count <= 5:  # Show first few exclusions
                        print(f"DEBUG - Excluding: {result.image_name} ({exclude_reason})")
                else:
                    # Include both pending and processing (interrupted) images
                    if result.processing_status in ["pending", "processing"]:
                        if result.processing_status == "processing":
                            interrupted_count += 1
                            print(f"Found interrupted image: {result.image_name}")
                        pending_results.append(result)
                    else:
                        print(f"WARNING - Unexpected status for non-excluded image: {result.image_name} ({result.processing_status})")
            
            print(f"Safe mode: Including {len(pending_results)} images to process")
            print(f"   - Pending: {len(pending_results) - interrupted_count}")
            print(f"   - Interrupted (processing): {interrupted_count}")
            print(f"Safe mode: Excluded {excluded_count} total images ({excluded_failed} failed + {excluded_completed} completed)")
        
        # STEP 6: Reset interrupted images to pending status
        # This ensures they start fresh and don't have stale start timestamps
        reset_count = 0
        for result in pending_results:
            if result.processing_status == "processing":
                print(f"Resetting interrupted image to pending: {result.image_name}")
                result.processing_status = "pending"
                result.timestamp_start = ""  # Clear stale start time
                result.timestamp_end = ""
                result.error_message = ""  # Clear any error from interruption
                reset_count += 1
        
        if reset_count > 0:
            print(f"? Reset {reset_count} interrupted images to pending status")
            self._save_batch_state()  # Save the reset statuses
        
        # STEP 7: Final verification - ensure no duplicates in output
        final_paths = [r.image_path for r in pending_results]
        if len(final_paths) != len(set(final_paths)):
            print(f"WARNING - Duplicate images in pending results!")
            # Remove duplicates, keeping first occurrence
            seen = set()
            deduplicated_results = []
            for result in pending_results:
                if result.image_path not in seen:
                    seen.add(result.image_path)
                    deduplicated_results.append(result)
            print(f"Deduplicating: {len(pending_results)} -> {len(deduplicated_results)} images")
            pending_results = deduplicated_results
        
        return pending_results
    
    def mark_image_started(self, image_path: str):
        """Mark an image as started processing."""
        for result in self.batch_state.image_results:
            if result.image_path == image_path:
                result.processing_status = "processing"
                result.timestamp_start = datetime.now().isoformat()
                break
        
        self._save_batch_state()
        self._log_progress(f"Started processing: {Path(image_path).name}")
    
    def mark_image_completed(self, image_path: str, processing_result: Dict):
        """Mark an image as completed and update statistics."""
        for result in self.batch_state.image_results:
            if result.image_path == image_path:
                # Update result with processing data
                result.processing_status = "completed"
                result.timestamp_end = datetime.now().isoformat()
                result.total_adipocytes = processing_result.get('total_adipocytes', 0)
                result.num_tumors = processing_result.get('num_tumors', 0)
                result.total_time = processing_result.get('total_time', 0.0)
                result.error_message = (
                    processing_result.get('warning_message', '')
                    or processing_result.get('error_message', '')
                    or ''
                )
                
                # Get adipocyte size statistics
                result.median_size, result.average_size = self._get_adipocyte_size_stats(
                    result.output_dir, result.image_name
                )
                
                # Update batch totals
                self.batch_state.processed_images += 1
                self.batch_state.successful_images += 1
                self.batch_state.total_adipocytes += result.total_adipocytes
                self.batch_state.total_tumors += result.num_tumors
                self.batch_state.total_processing_time += result.total_time
                break
        
        # Save state and update summary
        self._save_batch_state()
        self._update_summary_csv()
        self._log_progress(f"Completed: {Path(image_path).name} - {processing_result.get('total_adipocytes', 0)} adipocytes")
        self._print_progress_update()
    
    def mark_image_failed(self, image_path: str, error_message: str):
        """Mark an image as failed."""
        for result in self.batch_state.image_results:
            if result.image_path == image_path:
                result.processing_status = "failed"
                result.timestamp_end = datetime.now().isoformat()
                result.error_message = error_message
                
                # Update batch totals
                self.batch_state.processed_images += 1
                self.batch_state.failed_images += 1
                break
        
        # Save state and update summary
        self._save_batch_state()
        self._update_summary_csv()
        self._log_progress(f"Failed: {Path(image_path).name} - {error_message}")
    
    def _get_adipocyte_size_stats(self, output_dir: str, image_name: str) -> Tuple[float, float]:
        """Get median and average adipocyte size from CSV files."""
        try:
            # Try both possible naming conventions
            csv_paths = [
                os.path.join(output_dir, f"adipocyte_information_{image_name}.csv"),
                os.path.join(output_dir, f"{image_name}_adipocyte_information.csv"),
                os.path.join(output_dir, "adipocyte_information.csv")
            ]
            
            for csv_path in csv_paths:
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if 'Area_Microns_Squared' in df.columns and len(df) > 0:
                        median_size = df['Area_Microns_Squared'].median()
                        average_size = df['Area_Microns_Squared'].mean()
                        return float(median_size), float(average_size)
                    break
        except Exception as e:
            logging.warning(f"Could not extract size stats for {image_name}: {e}")
        return 0.0, 0.0
    
    def update_size_statistics(self):
        """Update adipocyte size statistics for all completed images."""
        updated = False
        for result in self.batch_state.image_results:
            if result.processing_status == "completed" and (result.median_size == 0.0 or result.average_size == 0.0):
                median_size, average_size = self._get_adipocyte_size_stats(result.output_dir, result.image_name)
                if median_size > 0.0 or average_size > 0.0:
                    result.median_size = median_size
                    result.average_size = average_size
                    updated = True
        
        if updated:
            self._save_batch_state()
            self._update_summary_csv()
            print(f"Updated adipocyte size statistics for batch {self.batch_id}")
        
        return updated
    
    def _update_summary_csv(self):
        """Update the summary CSV with current results."""
        with open(self.summary_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow([
                "Image Name", "Status", "Total Adipocytes", 
                "Median Size (\u00B5m\u00B2)", "Average Size (\u00B5m\u00B2)", 
                "Tumor Count", "Processing Time (s)", 
                "Start Time", "End Time", "Error Message"
            ])
            
            # Write data for all images
            for result in self.batch_state.image_results:
                writer.writerow([
                    result.image_name,
                    result.processing_status,
                    result.total_adipocytes,
                    f"{result.median_size:.2f}" if result.median_size > 0 else "",
                    f"{result.average_size:.2f}" if result.average_size > 0 else "",
                    result.num_tumors,
                    f"{result.total_time:.2f}" if result.total_time > 0 else "",
                    result.timestamp_start,
                    result.timestamp_end,
                    result.error_message
                ])
    
    def _log_progress(self, message: str):
        """Log progress message to progress log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.progress_log, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def _print_progress_update(self):
        """Print current progress to console."""
        completed = self.batch_state.successful_images
        failed = self.batch_state.failed_images
        total = self.batch_state.total_images
        
        print(f"Progress: {completed + failed}/{total} processed "
              f"({completed} successful, {failed} failed) | "
              f"Total adipocytes: {self.batch_state.total_adipocytes}")
    
    def create_final_summary(self):
        """Create final comprehensive batch summary."""
        end_time = datetime.now()
        start_time = datetime.fromisoformat(self.batch_state.start_time)
        total_elapsed = (end_time - start_time).total_seconds()
        
        # Create detailed statistics file
        stats_file = os.path.join(self.summary_dir, f"{self.batch_id}_final_statistics.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("BATCH PROCESSING FINAL SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Batch ID: {self.batch_id}\n")
            f.write(f"Start Time: {self.batch_state.start_time}\n")
            f.write(f"End Time: {end_time.isoformat()}\n")
            f.write(f"Total Elapsed Time: {total_elapsed:.1f} seconds\n\n")
            
            f.write("PROCESSING STATISTICS:\n")
            f.write(f"Total Images: {self.batch_state.total_images}\n")
            f.write(f"Successfully Processed: {self.batch_state.successful_images}\n")
            f.write(f"Failed: {self.batch_state.failed_images}\n")
            f.write(f"Success Rate: {(self.batch_state.successful_images/self.batch_state.total_images)*100:.1f}%\n\n")
            
            f.write("DETECTION STATISTICS:\n")
            f.write(f"Total Adipocytes Detected: {self.batch_state.total_adipocytes}\n")
            f.write(f"Total Tumor Regions: {self.batch_state.total_tumors}\n")
            if self.batch_state.successful_images > 0:
                f.write(f"Average Adipocytes per Image: {self.batch_state.total_adipocytes/self.batch_state.successful_images:.1f}\n")
                f.write(f"Average Processing Time per Image: {self.batch_state.total_processing_time/self.batch_state.successful_images:.1f} seconds\n")
        
        print(f"\nFinal batch summary created:")
        print(f"   . State file: {self.state_file}")
        print(f"   . Summary CSV: {self.summary_csv}")
        print(f"   . Configuration: {self.config_file}")
        print(f"   . Final statistics: {stats_file}")
        print(f"   . Progress log: {self.progress_log}")
        
        return {
            'state_file': self.state_file,
            'summary_csv': self.summary_csv,
            'config_file': self.config_file,
            'stats_file': stats_file,
            'progress_log': self.progress_log
        }
    
    def get_resume_info(self) -> str:
        """Get resume command for this batch."""
        return f"To resume this batch, use: --resume_batch {self.state_file}"


def find_resumable_batches(base_dir: str) -> List[Dict]:
    """Find all resumable batch state files in a directory."""
    resumable_batches = []
    
    if not os.path.exists(base_dir):
        return resumable_batches
    
    # Look for batch summary directories
    for item in os.listdir(base_dir):
        summary_path = os.path.join(base_dir, item)
        if os.path.isdir(summary_path) and item == "batch_summary":
            # Look for state files
            for file in os.listdir(summary_path):
                if file.endswith("_state.json"):
                    state_file = os.path.join(summary_path, file)
                    try:
                        with open(state_file, 'r') as f:
                            state_data = json.load(f)
                        
                        # Calculate progress
                        total_images = state_data.get('total_images', 0)
                        processed_images = state_data.get('processed_images', 0)
                        successful_images = state_data.get('successful_images', 0)
                        
                        resumable_batches.append({
                            'state_file': state_file,
                            'batch_id': state_data.get('batch_id', 'unknown'),
                            'start_time': state_data.get('start_time', 'unknown'),
                            'total_images': total_images,
                            'processed_images': processed_images,
                            'successful_images': successful_images,
                            'remaining_images': total_images - processed_images,
                            'base_output_dir': state_data.get('base_output_dir', 'unknown')
                        })
                    except Exception as e:
                        logging.warning(f"Could not read state file {state_file}: {e}")
    
    return resumable_batches
