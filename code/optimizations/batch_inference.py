#!/usr/bin/env python3
"""
Batch Inference Optimization
============================

Implements batched window processing for GPU acceleration.
"""

import torch
import numpy as np
from collections import defaultdict
import logging

class BatchedInferenceManager:
    """Manages batched window processing for improved GPU utilization."""
    
    def __init__(self, predictor, batch_size=4):
        self.predictor = predictor
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def process_window_batch(self, window_batch, window_coords_batch):
        """
        Process a batch of windows simultaneously for improved GPU utilization.
        
        Args:
            window_batch: List of numpy arrays (windows)
            window_coords_batch: List of (x, y) coordinates
            
        Returns:
            List of detection results
        """
        try:
            # Convert to tensors and move to GPU
            batch_tensors = []
            for window in window_batch:
                if len(window.shape) == 3:
                    batch_tensors.append(torch.from_numpy(window).permute(2, 0, 1))
            
            if not batch_tensors:
                return []
            
            # Batch inference
            with torch.no_grad():
                batch_outputs = self.predictor(batch_tensors)
            
            # Process results
            results = []
            for i, (output, (x, y)) in enumerate(zip(batch_outputs, window_coords_batch)):
                result = {
                    'instances': output['instances'],
                    'coords': (x, y),
                    'window_idx': i
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"Batch inference error: {e}")
            return []
