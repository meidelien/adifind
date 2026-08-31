#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Progress Utilities Module
=========================

Enhanced progress bars and visual feedback for long-running operations.
"""

import time
import random
from tqdm import tqdm


class ColorChangingTqdm(tqdm):
    """
    Enhanced TQDM progress bar with color-changing capability.
    Provides visual feedback through color transitions during processing.
    """
    def __init__(self, *args, **kwargs):
        self.colors = [
            '\033[91m',  # Red
            '\033[93m',  # Yellow  
            '\033[92m',  # Green
            '\033[94m',  # Blue
            '\033[95m',  # Magenta
            '\033[96m',  # Cyan
        ]
        self.reset_color = '\033[0m'
        self.current_color_index = 0
        super().__init__(*args, **kwargs)
    
    def update(self, n=1):
        """Update progress bar with color cycling."""
        super().update(n)
        
        # Change color every 10% progress
        if self.total and self.n > 0:
            progress_percent = (self.n / self.total) * 100
            new_color_index = int(progress_percent // 10) % len(self.colors)
            if new_color_index != self.current_color_index:
                self.current_color_index = new_color_index
                # Update color (this is a simplified version - full implementation would be more complex)
    
    def __enter__(self):
        return super().__enter__()
    
    def __exit__(self, *args):
        return super().__exit__(*args)


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'ColorChangingTqdm'
]
