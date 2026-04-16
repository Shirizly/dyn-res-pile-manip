#!/usr/bin/env python
"""Debug script to check output data format"""
import os
import numpy as np
from glob import glob

# Find the most recent output directory
output_dirs = sorted(glob('outputs/mpc_run_*'))
if not output_dirs:
    print("No output directories found")
    exit(1)

latest_dir = output_dirs[-1]
print(f"Checking directory: {latest_dir}\n")

# Check if there's raw observation data
npy_files = glob(os.path.join(latest_dir, '*.npy'))
image_files = glob(os.path.join(latest_dir, 'images', '*.png'))

print(f"Found {len(npy_files)} .npy files")
print(f"Found {len(image_files)} image files\n")

# Try to load and check the raw_obs if saved
# For now, let's check one of the saved images
if image_files:
    import cv2
    img = cv2.imread(image_files[0])
    print(f"First image: {image_files[0]}")
    print(f"  Shape: {img.shape if img is not None else 'Failed to load'}")
    print(f"  Dtype: {img.dtype if img is not None else 'N/A'}")
    print(f"  Min/Max: {img.min() if img is not None else 'N/A'} / {img.max() if img is not None else 'N/A'}")
