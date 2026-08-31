#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Status Checker
====================

Quick utility to check the status of batch processing jobs.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from batch_processing import find_resumable_batches


def main():
    """Main function to check batch status."""
    if len(sys.argv) > 1:
        check_dir = sys.argv[1]
    else:
        check_dir = os.getcwd()
    
    print(f"?? Checking for batch processing status in: {check_dir}")
    print("=" * 80)
    
    # Find all resumable batches
    resumable_batches = find_resumable_batches(check_dir)
    
    if not resumable_batches:
        print("? No batch processing jobs found")
        return
    
    print(f"?? Found {len(resumable_batches)} batch job(s):")
    print("-" * 80)
    
    for i, batch in enumerate(resumable_batches, 1):
        print(f"{i}. Batch: {batch['batch_id']}")
        print(f"   ?? Started: {batch['start_time']}")
        print(f"   ?? Progress: {batch['successful_images']}/{batch['total_images']} completed")
        
        if batch['remaining_images'] > 0:
            print(f"   ? Remaining: {batch['remaining_images']} images")
            print(f"   ?? Resume: python main.py --resume_batch {batch['state_file']}")
        else:
            print(f"   ? Status: COMPLETED")
        
        print(f"   ?? Output: {batch['base_output_dir']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
