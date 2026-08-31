#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Statistics Updater
=========================

Updates existing batch summaries with corrected adipocyte size statistics.
"""

import os
import sys
import json
from pathlib import Path
from batch_processing import BatchProcessor, find_resumable_batches


def update_batch_statistics(state_file_path: str):
    """Update statistics for a specific batch."""
    try:
        print(f"?? Updating statistics for: {state_file_path}")
        
        # Load the batch state (use utf-8 and replace invalid bytes)
        with open(state_file_path, 'r', encoding='utf-8', errors='replace') as f:
            state_data = json.load(f)
        
        base_output_dir = state_data['base_output_dir']
        batch_id = state_data['batch_id']
        
        # Create a temporary args object (not used for resume operations)
        class MockArgs:
            def __init__(self, state_data):
                args_data = state_data.get('args_snapshot', {})
                for key, value in args_data.items():
                    setattr(self, key, value)
        
        mock_args = MockArgs(state_data)
        
        # Initialize batch processor with the existing state
        batch_processor = BatchProcessor(base_output_dir, [], mock_args, state_file_path)
        
        # Update size statistics
        updated = batch_processor.update_size_statistics()
        
        if updated:
            print(f"? Successfully updated statistics for batch {batch_id}")
        else:
            print(f"??  No updates needed for batch {batch_id}")
            
        return True
        
    except Exception as e:
        print(f"? Error updating batch statistics: {e}")
        return False


def main():
    """Main function to update batch statistics."""
    if len(sys.argv) > 1:
        # Update specific state file
        state_file = sys.argv[1]
        if not os.path.exists(state_file):
            print(f"? State file not found: {state_file}")
            return
        
        update_batch_statistics(state_file)
    else:
        # Find and update all resumable batches in current directory
        print("?? Searching for batch jobs to update...")
        current_dir = os.getcwd()
        resumable_batches = find_resumable_batches(current_dir)
        
        if not resumable_batches:
            print("? No batch jobs found in current directory")
            return
        
        print(f"?? Found {len(resumable_batches)} batch job(s) to update")
        print("-" * 80)
        
        for i, batch in enumerate(resumable_batches, 1):
            print(f"\n{i}. Updating batch: {batch['batch_id']}")
            success = update_batch_statistics(batch['state_file'])
            if not success:
                print(f"??  Failed to update batch {batch['batch_id']}")
        
        print("\n? Batch statistics update completed!")


if __name__ == "__main__":
    main()
