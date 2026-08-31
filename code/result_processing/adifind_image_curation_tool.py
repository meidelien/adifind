#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdiFind Image Curation Tool
============================

Tkinter GUI for reviewing and curating AdiFind batch results.
Opens annotated TIFF images in IrfanView for visual inspection,
allowing approve/reject decisions with session resume support.

Usage:
    python -m result_processing.adifind_image_curation_tool
    # or via CLI entry point:
    adifind-curate
"""

import os
import logging
import subprocess
import json
from datetime import datetime
import re

import pandas as pd
import tkinter as tk
from tkinter import messagebox, filedialog

logger = logging.getLogger(__name__)


class ImageCurationTool:
    # Column name mappings to handle different CSV formats
    COLUMN_MAPPINGS = {
        'image_name': ['Image Name', 'image_name', 'ImageName'],
        'adipocyte_count': ['Total Adipocytes', 'Adipocyte Count', 'total_adipocytes'],
        'avg_size': ['Average Adipocyte Size (microns squared)', 'Average Size (\u00B5m\u00B2)', 'Average Size', 'average_size'],
        'median_size': ['Median Adipocyte Size (microns squared)', 'Median Size (\u00B5m\u00B2)', 'Median Size', 'median_size']
    }

    # Minimum adipocyte count filter threshold
    MIN_ADIPOCYTE_COUNT = 100

    # Pattern for AdiFind result folder names: adifind_results_{name}_{YYYYMMDD}_{HHMMSS}
    _RESULT_FOLDER_RE = re.compile(
        r'^adifind_results_(.+)_(\d{8})_(\d{6})$'
    )

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AdiFind Curation Tool")
        self.root.geometry("600x250")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize variables
        self.csv_path = None
        self.output_csv_path = None
        self.output_folder = None
        self.base_folder = None
        self.df = None
        self.current_index = 0
        self.approved_images = []
        self.rejected_images = {}  # Dict to store rejected images with reasons
        self.irfanview_path = None  # Will be set in setup
        self.progress_file = None  # Path to progress file
        self.session_data = {}  # Session data for resuming
        self.resume_start_index = 0  # Track where we resumed from (0 for new sessions)

        # Store detected column names for this CSV
        self.detected_columns = {}

        # Create initial UI for setup
        self.setup_ui()

    def get_column_value(self, row, column_type):
        """Get value from row using flexible column name matching"""
        # First check if we've already detected the column for this type
        if column_type in self.detected_columns:
            col_name = self.detected_columns[column_type]
            if col_name in row.index:
                return row[col_name]

        # If not detected yet, search for it
        for col_name in self.COLUMN_MAPPINGS.get(column_type, []):
            if col_name in row.index:
                # Cache the detected column name
                self.detected_columns[column_type] = col_name
                return row[col_name]

        return None

    def detect_csv_columns(self):
        """Detect and validate CSV columns, providing user feedback"""
        if self.df is None:
            return False

        # Detect image name column
        image_col = None
        for col_name in self.COLUMN_MAPPINGS['image_name']:
            if col_name in self.df.columns:
                image_col = col_name
                self.detected_columns['image_name'] = col_name
                break

        if not image_col:
            messagebox.showerror("Error",
                f"CSV must have one of these columns for image names:\n" +
                "\n".join(self.COLUMN_MAPPINGS['image_name']))
            return False

        # Detect other columns (optional, but provide info)
        detected_info = [f"\u2705 Image Name: '{image_col}'"]

        for col_type in ['adipocyte_count', 'avg_size', 'median_size']:
            for col_name in self.COLUMN_MAPPINGS[col_type]:
                if col_name in self.df.columns:
                    self.detected_columns[col_type] = col_name
                    detected_info.append(f"\u2705 {col_type.replace('_', ' ').title()}: '{col_name}'")
                    break
            else:
                detected_info.append(f"\u274c {col_type.replace('_', ' ').title()}: Not found")

        # Log what was detected
        info_msg = "CSV Column Detection:\n" + "\n".join(detected_info)
        logger.info(info_msg)

        return True

    def setup_ui(self):
        """Initial UI to set up paths"""
        for widget in self.root.winfo_children():
            widget.destroy()

        # Instructions
        tk.Label(self.root, text="AdiFind Curation Tool", font=("Arial", 16)).pack(pady=10)

        # Buttons for selecting files/folders
        tk.Button(self.root, text="Select CSV File", command=self.select_csv).pack(pady=5)
        tk.Button(self.root, text="Select Base Folder", command=self.select_base_folder).pack(pady=5)
        tk.Button(self.root, text="Select IrfanView Path", command=self.select_irfanview).pack(pady=5)
        tk.Button(self.root, text="Set Output Folder (Optional)", command=self.select_output_folder).pack(pady=5)

        # Session management buttons
        session_frame = tk.Frame(self.root)
        session_frame.pack(pady=5)
        tk.Button(session_frame, text="Resume Previous Session", command=self.resume_session).pack(side=tk.LEFT, padx=5)
        tk.Button(session_frame, text="Start New Curation", command=self.start_curation).pack(side=tk.LEFT, padx=5)

    def select_csv(self):
        """Select the input CSV file"""
        self.csv_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv")]
        )
        if self.csv_path:
            self.output_csv_path = os.path.join(
                os.path.dirname(self.csv_path),
                "curated_" + os.path.basename(self.csv_path)
            )
            # If output folder was already selected, use that instead
            if hasattr(self, 'output_folder') and self.output_folder:
                self.output_csv_path = os.path.join(
                    self.output_folder,
                    "curated_" + os.path.basename(self.csv_path)
                )
            messagebox.showinfo("Selected", f"CSV: {self.csv_path}")

    def select_base_folder(self):
        """Select the base folder containing image folders"""
        self.base_folder = filedialog.askdirectory(title="Select Base Folder")
        if self.base_folder:
            messagebox.showinfo("Selected", f"Base folder: {self.base_folder}")

    def select_irfanview(self):
        """Select the IrfanView executable"""
        self.irfanview_path = filedialog.askopenfilename(
            title="Select IrfanView Executable",
            filetypes=[("Executable", "*.exe")]
        )
        if self.irfanview_path:
            messagebox.showinfo("Selected", f"IrfanView: {self.irfanview_path}")

    def select_output_folder(self):
        """Select the output folder for the curated CSV"""
        output_folder = filedialog.askdirectory(title="Select Output Folder")
        if output_folder and self.csv_path:
            # Create the output path using the selected folder and original CSV filename
            original_filename = os.path.basename(self.csv_path)
            self.output_csv_path = os.path.join(output_folder, "curated_" + original_filename)
            messagebox.showinfo("Selected", f"Output folder: {output_folder}")
        elif output_folder and not self.csv_path:
            # Store the folder for later when CSV is selected
            self.output_folder = output_folder
            messagebox.showinfo("Selected", f"Output folder: {output_folder}\nCSV filename will be set when you select the input CSV.")

    def save_progress(self):
        """Save current progress to a file"""
        if not self.progress_file:
            return

        # Store the list of image names from the filtered dataframe to ensure consistency
        image_names_list = []
        if self.df is not None:
            image_col = self.detected_columns.get('image_name', 'Image Name')
            if image_col in self.df.columns:
                image_names_list = self.df[image_col].tolist()

        progress_data = {
            'csv_path': self.csv_path,
            'output_csv_path': self.output_csv_path,
            'base_folder': self.base_folder,
            'irfanview_path': self.irfanview_path,
            'current_index': self.current_index,
            'approved_images': self.approved_images,
            'rejected_images': self.rejected_images,
            'total_images': len(self.df) if self.df is not None else 0,
            'timestamp': datetime.now().isoformat(),
            'detected_columns': self.detected_columns,
            'filtered_image_names': image_names_list,  # Store the filtered image list
            'adipocyte_filter_applied': self.detected_columns.get('adipocyte_count') is not None,
            'min_adipocyte_count': self.MIN_ADIPOCYTE_COUNT  # Store the threshold used
        }

        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            logger.info(f"Progress saved to: {self.progress_file}")
        except Exception as e:
            logger.error(f"Error saving progress: {e}")

    def load_progress(self, progress_file):
        """Load progress from a file"""
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
            return None

    def resume_session(self):
        """Resume a previous curation session"""
        progress_file = filedialog.askopenfilename(
            title="Select Progress File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not progress_file:
            return

        progress_data = self.load_progress(progress_file)
        if not progress_data:
            messagebox.showerror("Error", "Failed to load progress file")
            return

        # Restore session data
        self.csv_path = progress_data.get('csv_path')
        self.output_csv_path = progress_data.get('output_csv_path')
        self.base_folder = progress_data.get('base_folder')
        self.irfanview_path = progress_data.get('irfanview_path')
        self.current_index = progress_data.get('current_index', 0)
        self.approved_images = progress_data.get('approved_images', [])
        self.rejected_images = progress_data.get('rejected_images', {})
        self.progress_file = progress_file

        # Restore detected columns from progress file
        saved_detected_columns = progress_data.get('detected_columns', {})
        filtered_image_names = progress_data.get('filtered_image_names', [])
        adipocyte_filter_applied = progress_data.get('adipocyte_filter_applied', False)
        saved_min_adipocyte_count = progress_data.get('min_adipocyte_count', self.MIN_ADIPOCYTE_COUNT)

        # IMPORTANT: Always use the current MIN_ADIPOCYTE_COUNT for filtering
        # The saved count is just for informational/warning purposes
        filter_threshold = self.MIN_ADIPOCYTE_COUNT

        # Validate that files still exist
        if not all([self.csv_path, self.base_folder, self.irfanview_path]):
            messagebox.showerror("Error", "Some required paths are missing from the session")
            return

        if not os.path.exists(self.csv_path):
            messagebox.showerror("Error", f"CSV file not found: {self.csv_path}")
            return

        if not os.path.exists(self.base_folder):
            messagebox.showerror("Error", f"Base folder not found: {self.base_folder}")
            return

        if not os.path.exists(self.irfanview_path):
            messagebox.showerror("Error", f"IrfanView not found: {self.irfanview_path}")
            return

        # Load the CSV
        try:
            self.df = pd.read_csv(self.csv_path)
            original_count = len(self.df)
            logger.info(f"CSV loaded: {original_count} total rows")

            # Detect and validate columns
            if not self.detect_csv_columns():
                return

            # CRITICAL: ALWAYS apply adipocyte filtering when resuming
            adipocyte_col = None
            for col_name in self.COLUMN_MAPPINGS['adipocyte_count']:
                if col_name in self.df.columns:
                    adipocyte_col = col_name
                    logger.info(f"Found adipocyte column: '{col_name}'")
                    break

            if adipocyte_col:
                # Apply filter
                logger.info(f"Applying filter: {adipocyte_col} >= {filter_threshold}")
                self.df[adipocyte_col] = pd.to_numeric(self.df[adipocyte_col], errors='coerce').fillna(0)
                self.df = self.df[self.df[adipocyte_col] >= filter_threshold].reset_index(drop=True)
                filtered_count = len(self.df)
                logger.info(f"After filtering: {filtered_count} rows (removed {original_count - filtered_count})")

                # Warn user if threshold changed
                if saved_min_adipocyte_count != filter_threshold:
                    threshold_msg = (
                        f"Filter Threshold Updated!\n\n"
                        f"Session was saved with: \u2265{saved_min_adipocyte_count} adipocytes\n"
                        f"Current threshold: \u2265{filter_threshold} adipocytes\n\n"
                        f"Applied current filter:\n"
                        f"\u2022 Total images in CSV: {original_count}\n"
                        f"\u2022 After filtering: {filtered_count}\n"
                        f"\u2022 Removed: {original_count - filtered_count}\n\n"
                        f"\u26a0 Your progress indices may not align perfectly.\n"
                        f"Recommendation: Start a new curation session.\n\n"
                        f"Continue anyway?"
                    )
                    if not messagebox.askyesno("Threshold Changed", threshold_msg):
                        self.reset_session()
                        return

                # Check expected total from progress file
                expected_total = progress_data.get('total_images', 0)
                if expected_total > 0 and filtered_count != expected_total:
                    mismatch_msg = (
                        f"Data Mismatch Warning!\n\n"
                        f"Progress file expected: {expected_total} images\n"
                        f"After filtering got: {filtered_count} images\n"
                        f"Difference: {abs(filtered_count - expected_total)}\n\n"
                        f"Possible causes:\n"
                        f"\u2022 CSV was modified\n"
                        f"\u2022 Filter threshold changed\n\n"
                        f"\u26a0 This may cause issues with image indices.\n"
                        f"Start a new session instead?"
                    )
                    if messagebox.askyesno("Data Mismatch", mismatch_msg):
                        self.reset_session()
                        return
            else:
                logger.warning("No adipocyte column found - no filtering applied")
                messagebox.showwarning(
                    "No Filter",
                    f"Could not find adipocyte count column.\n"
                    f"Proceeding with all {original_count} images from CSV."
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
            return

        # Show resume information
        total_images = len(self.df)
        completed = self.current_index
        approved = len(self.approved_images)
        rejected = len(self.rejected_images)
        remaining = total_images - completed

        resume_msg = (
            f"Session loaded successfully!\n\n"
            f"Total filtered images: {total_images}\n"
            f"Progress: {completed}/{total_images} images reviewed\n"
            f"Approved: {approved} images\n"
            f"Rejected: {rejected} images\n"
            f"Remaining: {remaining} images\n"
            f"Starting from image {completed + 1}\n\n"
            f"Continue curation?"
        )

        if messagebox.askyesno("Resume Session", resume_msg):
            # Store the resume starting point so we can navigate back into previous decisions
            self.resume_start_index = self.current_index
            # Switch to curation UI and continue
            self.create_curation_ui()
            self.process_current_image()
        else:
            # Reset to allow new session
            self.reset_session()

    def reset_session(self):
        """Reset session data"""
        self.csv_path = None
        self.output_csv_path = None
        self.base_folder = None
        self.df = None
        self.current_index = 0
        self.approved_images = []
        self.rejected_images = {}
        self.irfanview_path = None
        self.progress_file = None
        self.session_data = {}
        self.detected_columns = {}
        self.resume_start_index = 0

    def select_output_csv(self):
        """Select the output CSV file path"""
        self.output_csv_path = filedialog.asksaveasfilename(
            title="Save Curated CSV As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if self.output_csv_path:
            messagebox.showinfo("Selected", f"Output CSV: {self.output_csv_path}")

    def start_curation(self):
        """Start the curation process"""
        if not all([self.csv_path, self.base_folder, self.irfanview_path]):
            messagebox.showerror("Error", "Please select all required paths first")
            return

        # Load the CSV
        try:
            self.df = pd.read_csv(self.csv_path)

            # Detect and validate columns
            if not self.detect_csv_columns():
                return

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
            return

        # Filter images with minimum adipocyte count
        original_count = len(self.df)

        # Check if adipocyte count column exists
        adipocyte_col = self.detected_columns.get('adipocyte_count')
        if adipocyte_col:
            # Convert to numeric, replacing any non-numeric values with 0
            self.df[adipocyte_col] = pd.to_numeric(self.df[adipocyte_col], errors='coerce').fillna(0)

            # Filter for images with at least MIN_ADIPOCYTE_COUNT adipocytes
            self.df = self.df[self.df[adipocyte_col] >= self.MIN_ADIPOCYTE_COUNT].reset_index(drop=True)

            filtered_count = len(self.df)
            excluded_count = original_count - filtered_count

            # Show filtering results
            filter_msg = (
                f"Adipocyte Filter Applied:\n\n"
                f"Original images: {original_count}\n"
                f"Images with \u2265{self.MIN_ADIPOCYTE_COUNT} adipocytes: {filtered_count}\n"
                f"Images excluded by filter: {excluded_count}\n\n"
                f"Continue with curation of {filtered_count} images?"
            )

            if not messagebox.askyesno("Filter Applied", filter_msg):
                return

            if filtered_count == 0:
                messagebox.showinfo("No Images", f"No images meet the minimum adipocyte criteria (\u2265{self.MIN_ADIPOCYTE_COUNT}).")
                return
        else:
            # If no adipocyte count column, warn user but continue
            messagebox.showwarning(
                "Warning",
                f"No adipocyte count column found in CSV. "
                f"Proceeding without adipocyte filtering."
            )

        # Create progress file path - store in output folder
        if self.output_csv_path:
            output_dir = os.path.dirname(self.output_csv_path)
        else:
            output_dir = os.path.dirname(self.csv_path)

        csv_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        self.progress_file = os.path.join(output_dir, f"{csv_name}_curation_progress.json")

        # Reset session for new curation
        self.current_index = 0
        self.approved_images = []
        self.rejected_images = {}
        self.resume_start_index = 0

        # Switch to curation UI
        self.create_curation_ui()
        # Process the first image
        self.process_current_image()

    def create_curation_ui(self):
        """Create the UI for image curation"""
        for widget in self.root.winfo_children():
            widget.destroy()

        # Keep the window always on top
        self.root.attributes('-topmost', True)

        # Resize window to accommodate metrics
        self.root.geometry("650x350")

        # Status information
        self.status_label = tk.Label(self.root, text="", font=("Arial", 12), wraplength=600)
        self.status_label.pack(pady=10)

        # Progress information
        self.progress_label = tk.Label(self.root, text="")
        self.progress_label.pack(pady=5)

        # Metrics frame
        metrics_frame = tk.LabelFrame(self.root, text="Image Metrics", font=("Arial", 10, "bold"))
        metrics_frame.pack(pady=10, padx=20, fill="x")

        self.adipocytes_label = tk.Label(metrics_frame, text="Total Adipocytes: --")
        self.adipocytes_label.pack(pady=2)

        self.avg_size_label = tk.Label(metrics_frame, text="Average Size: --")
        self.avg_size_label.pack(pady=2)

        self.median_size_label = tk.Label(metrics_frame, text="Median Size: --")
        self.median_size_label.pack(pady=2)

        # Navigation and decision buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        # Previous button
        self.prev_button = tk.Button(button_frame, text="\u25c0 Previous (P)", width=15,
                                     command=self.previous_image, bg="lightblue")
        self.prev_button.pack(side=tk.LEFT, padx=5)

        # Include/Exclude buttons
        tk.Button(button_frame, text="Include (Y)", width=15, command=self.approve_image,
                 bg="lightgreen").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Exclude (N)", width=15, command=self.show_exclusion_dialog,
                 bg="lightcoral").pack(side=tk.LEFT, padx=5)

        # Keyboard bindings
        self.root.bind('y', lambda e: self.approve_image())
        self.root.bind('Y', lambda e: self.approve_image())
        self.root.bind('n', lambda e: self.show_exclusion_dialog())
        self.root.bind('N', lambda e: self.show_exclusion_dialog())
        self.root.bind('p', lambda e: self.previous_image())
        self.root.bind('P', lambda e: self.previous_image())

    def process_current_image(self):
        """Process the current image"""
        if self.current_index >= len(self.df):
            self.finish_curation()
            return

        # Update Previous button state
        if hasattr(self, 'prev_button'):
            if self.current_index > 0:
                self.prev_button.config(state=tk.NORMAL)
            else:
                self.prev_button.config(state=tk.DISABLED)

        # Get current image data
        current_row = self.df.iloc[self.current_index]

        # Get image name using flexible column mapping
        image_name = self.get_column_value(current_row, 'image_name')
        if not image_name:
            # Fallback to first column if no match
            image_name = current_row.iloc[0]

        # Check if this is a previously-decided image from an earlier session
        is_reevaluating = (hasattr(self, 'resume_start_index') and
                          self.current_index < self.resume_start_index)

        # Update status with indicator if re-evaluating
        if is_reevaluating:
            self.status_label.config(text=f"\U0001f504 Re-evaluating: {image_name}")
        else:
            self.status_label.config(text=f"Reviewing: {image_name}")

        self.progress_label.config(text=f"Image {self.current_index + 1} of {len(self.df)}")

        # Update metrics using flexible column mapping
        try:
            # Get adipocyte count
            total_adipocytes = self.get_column_value(current_row, 'adipocyte_count')
            if total_adipocytes is None:
                total_adipocytes = "N/A"

            # Get average size
            avg_size = self.get_column_value(current_row, 'avg_size')
            if avg_size is not None and avg_size != "N/A":
                try:
                    avg_size = f"{float(avg_size):.2f}"
                except (ValueError, TypeError):
                    avg_size = str(avg_size)
            else:
                avg_size = "N/A"

            # Get median size
            median_size = self.get_column_value(current_row, 'median_size')
            if median_size is not None and median_size != "N/A":
                try:
                    median_size = f"{float(median_size):.2f}"
                except (ValueError, TypeError):
                    median_size = str(median_size)
            else:
                median_size = "N/A"

            self.adipocytes_label.config(text=f"Total Adipocytes: {total_adipocytes}")
            self.avg_size_label.config(text=f"Average Size: {avg_size} \u00B5m\u00B2")
            self.median_size_label.config(text=f"Median Size: {median_size} \u00B5m\u00B2")
        except Exception as e:
            logger.warning(f"Error updating metrics: {e}")
            self.adipocytes_label.config(text="Total Adipocytes: N/A")
            self.avg_size_label.config(text="Average Size: N/A")
            self.median_size_label.config(text="Median Size: N/A")

        # Find and open the image
        self.open_image(image_name)

    def _extract_image_name_from_folder(self, folder_name):
        """
        Extract the image name from an AdiFind result folder name.

        AdiFind result folders follow the pattern:
            adifind_results_{image_name}_{YYYYMMDD}_{HHMMSS}

        Returns the image name portion, or None if the folder doesn't match.
        """
        m = self._RESULT_FOLDER_RE.match(folder_name)
        if m:
            return m.group(1)
        return None

    def open_image(self, image_name):
        """Find and open the TIFF file for the given image name"""
        logger.debug(f"Processing image: {image_name}")

        # Look for a result folder that contains this image name
        matching_folder = None
        try:
            all_entries = os.listdir(self.base_folder)

            for entry in all_entries:
                folder_path = os.path.join(self.base_folder, entry)
                if not os.path.isdir(folder_path):
                    continue

                # Try the AdiFind result folder pattern first
                extracted_name = self._extract_image_name_from_folder(entry)
                if extracted_name and extracted_name == image_name:
                    matching_folder = folder_path
                    logger.debug(f"Matched result folder: {entry}")
                    break

                # Fallback: folder starts with the image name
                if entry.startswith(image_name):
                    matching_folder = folder_path
                    logger.debug(f"Matched folder by prefix: {entry}")
                    break

        except Exception as e:
            logger.error(f"Error accessing base folder: {e}")
            messagebox.showerror("Error", f"Error accessing base folder: {str(e)}")
            self.move_to_next()
            return

        if not matching_folder:
            logger.warning(f"No matching folder found for: {image_name}")
            messagebox.showwarning("Warning", f"No folder found for {image_name}")
            self.move_to_next()
            return

        # Look for the annotated TIFF file (current pipeline naming convention)
        expected_tiff = os.path.join(matching_folder, f"{image_name}_adifind_annotated.tiff")

        if os.path.exists(expected_tiff):
            self._open_in_irfanview(expected_tiff)
        else:
            logger.debug(f"Expected TIFF not found: {expected_tiff}, trying fallback...")
            # Fallback: find any TIFF files in the matching folder
            tiff_files = []
            try:
                for file in os.listdir(matching_folder):
                    if file.lower().endswith(('.tif', '.tiff')):
                        tiff_files.append(os.path.join(matching_folder, file))
            except Exception as e:
                logger.error(f"Error accessing folder contents: {e}")
                messagebox.showerror("Error", f"Error accessing folder {matching_folder}: {str(e)}")
                self.move_to_next()
                return

            if not tiff_files:
                messagebox.showwarning("Warning", f"No TIFF file found in {matching_folder}")
                self.move_to_next()
                return

            # Open the first TIFF file found
            self._open_in_irfanview(tiff_files[0])

    def _open_in_irfanview(self, tiff_path):
        """Open a TIFF file in IrfanView."""
        try:
            normalized_path = os.path.normpath(tiff_path)
            logger.debug(f"Opening in IrfanView: {normalized_path}")
            subprocess.Popen([self.irfanview_path, normalized_path])

            # Bring the main window back to front after IrfanView opens
            self.root.after(1000, self._bring_window_to_front)

        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            messagebox.showerror("Error", f"Failed to open image: {str(e)}")
            self.move_to_next()

    def _bring_window_to_front(self):
        """Helper method to bring the main window to front and give it focus"""
        try:
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.attributes('-topmost', False)
            self.root.attributes('-topmost', True)
            self.root.focus_force()
        except Exception as e:
            logger.debug(f"Error bringing window to front: {e}")

    def show_exclusion_dialog(self):
        """Show dialog to select exclusion reason"""
        # Create a new dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Exclusion Reason")
        dialog.geometry("500x450")
        dialog.attributes('-topmost', True)
        dialog.grab_set()  # Make dialog modal
        dialog.focus_set()  # Give focus to dialog for keyboard shortcuts

        # Position dialog beneath the main window
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        main_height = self.root.winfo_height()
        dialog.geometry(f"500x450+{main_x}+{main_y + main_height + 10}")

        dialog.transient(self.root)

        tk.Label(dialog, text="Why are you excluding this image? (Use 1-8 keys)",
                font=("Arial", 12, "bold")).pack(pady=10)

        # Exclusion reasons with numbers
        exclusion_var = tk.StringVar()
        reasons = [
            "1. Poor tissue quality",
            "2. Too few total adipocytes",
            "3. Tissue processing artifacts",
            "4. Staining issues",
            "5. Image quality problems",
            "6. Excessive background",
            "7. Tissue damage",
            "8. Other technical issues"
        ]

        # Store the actual reason text without numbers for processing
        reason_values = [
            "Poor tissue quality",
            "Too few total adipocytes",
            "Tissue processing artifacts",
            "Staining issues",
            "Image quality problems",
            "Excessive background",
            "Tissue damage",
            "Other technical issues"
        ]

        for i, reason in enumerate(reasons):
            tk.Radiobutton(dialog, text=reason, variable=exclusion_var,
                          value=reason_values[i], font=("Arial", 10)).pack(anchor='w', padx=20, pady=2)

        # Custom reason
        tk.Label(dialog, text="Custom reason (or press 9 to focus here):", font=("Arial", 10)).pack(anchor='w', padx=20, pady=(10,0))
        custom_entry = tk.Entry(dialog, width=50)
        custom_entry.pack(padx=20, pady=5)

        # Button frame
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=20)

        def confirm_exclusion():
            reason = exclusion_var.get()
            if not reason and custom_entry.get().strip():
                reason = custom_entry.get().strip()
            if not reason:
                reason = "No reason specified"

            self.reject_image(reason)
            dialog.destroy()

        def cancel_exclusion():
            dialog.destroy()

        # Keyboard event handlers
        def on_key_press(event):
            key = event.char
            if key in '12345678':
                # Select the corresponding reason
                idx = int(key) - 1
                if idx < len(reason_values):
                    exclusion_var.set(reason_values[idx])
            elif key == '9':
                # Focus on custom entry
                custom_entry.focus_set()
            elif event.keysym == 'Return':
                confirm_exclusion()
            elif event.keysym == 'Escape':
                cancel_exclusion()

        # Bind keyboard events
        dialog.bind('<KeyPress>', on_key_press)
        dialog.bind('<Return>', lambda e: confirm_exclusion())
        dialog.bind('<Escape>', lambda e: cancel_exclusion())

        tk.Button(button_frame, text="Confirm Exclusion (Enter)", command=confirm_exclusion,
                 bg="lightcoral").pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Cancel (Esc)", command=cancel_exclusion).pack(side=tk.LEFT, padx=10)

        # Set default selection
        exclusion_var.set(reason_values[0])

    def approve_image(self):
        """Mark current image as approved"""
        if self.current_index < len(self.df):
            self.approved_images.append(self.current_index)
            # Save progress after each decision
            self.save_progress()
            # Kill any running IrfanView instance
            self.close_irfanview()
            self.move_to_next()

    def reject_image(self, reason="Not specified"):
        """Mark current image as rejected with reason"""
        if self.current_index < len(self.df):
            current_row = self.df.iloc[self.current_index]
            image_name = self.get_column_value(current_row, 'image_name')
            if not image_name:
                # Fallback to first column if no match
                image_name = current_row.iloc[0]

            self.rejected_images[str(self.current_index)] = {
                'image_name': image_name,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }

        # Save progress after each decision
        self.save_progress()
        # Kill any running IrfanView instance
        self.close_irfanview()
        self.move_to_next()

    def move_to_next(self):
        """Move to the next image"""
        self.current_index += 1
        self.process_current_image()

    def previous_image(self):
        """Go back to the previous image and allow re-evaluation"""
        if self.current_index > 0:
            # Close current IrfanView instance
            self.close_irfanview()

            # Move back one image
            self.current_index -= 1

            # Check if we're going back into a previously-decided image
            was_previously_decided = (self.current_index in self.approved_images or
                                     str(self.current_index) in self.rejected_images)

            # Remove this image from approved list if it was approved
            if self.current_index in self.approved_images:
                self.approved_images.remove(self.current_index)
                logger.info(f"Undid approval for image at index {self.current_index}")

            # Remove this image from rejected dict if it was rejected
            if str(self.current_index) in self.rejected_images:
                rejection_info = self.rejected_images[str(self.current_index)]
                del self.rejected_images[str(self.current_index)]
                logger.info(f"Undid rejection for image at index {self.current_index} (was: {rejection_info['reason']})")

            # Inform user if they're reviewing a previously-decided image from an earlier session
            if was_previously_decided and hasattr(self, 'resume_start_index') and self.current_index < self.resume_start_index:
                current_row = self.df.iloc[self.current_index]
                image_name = self.get_column_value(current_row, 'image_name')
                if not image_name:
                    image_name = current_row.iloc[0]
                logger.info(f"Reviewing previously-decided image from earlier session: {image_name} (index {self.current_index})")

            # Save progress after undoing
            self.save_progress()

            # Process the previous image
            self.process_current_image()

    def close_irfanview(self):
        """Close any open IrfanView instances"""
        try:
            subprocess.call(["taskkill", "/f", "/im", "i_view32.exe"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.call(["taskkill", "/f", "/im", "i_view64.exe"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass  # Ignore errors if IrfanView wasn't open

    def finish_curation(self):
        """Complete the curation process and save results"""
        # Create a new dataframe with only approved images
        approved_df = self.df.iloc[self.approved_images].copy()

        # Save the approved dataframe to CSV
        try:
            approved_df.to_csv(self.output_csv_path, index=False)

            # Save rejected images summary
            if self.rejected_images:
                rejected_summary_path = self.output_csv_path.replace('.csv', '_rejected_summary.csv')
                rejected_data = []
                for idx, info in self.rejected_images.items():
                    row_data = self.df.iloc[int(idx)].to_dict()
                    row_data['Rejection_Reason'] = info['reason']
                    row_data['Rejection_Timestamp'] = info['timestamp']
                    rejected_data.append(row_data)

                rejected_df = pd.DataFrame(rejected_data)
                rejected_df.to_csv(rejected_summary_path, index=False)
                rejected_msg = f"\nRejected images summary: {rejected_summary_path}"
            else:
                rejected_msg = ""

            # Clean up progress file
            if self.progress_file and os.path.exists(self.progress_file):
                os.remove(self.progress_file)
                logger.info(f"Progress file deleted: {self.progress_file}")

            messagebox.showinfo(
                "Curation Complete",
                f"Selected {len(self.approved_images)} of {len(self.df)} images.\n"
                f"Rejected {len(self.rejected_images)} images.\n"
                f"Approved images saved to: {self.output_csv_path}{rejected_msg}\n\n"
                f"Progress file has been cleaned up."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

        self.root.quit()

    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?\n\nYour progress has been saved and you can resume later."):
            self.close_irfanview()
            # Save final progress before closing
            self.save_progress()
            self.root.destroy()

    def run(self):
        """Start the application"""
        self.root.mainloop()


def main():
    """CLI entry point for the AdiFind curation tool."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = ImageCurationTool()
    app.run()


if __name__ == "__main__":
    main()
