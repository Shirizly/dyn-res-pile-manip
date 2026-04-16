#!/usr/bin/env python
"""Test to understand the actual render output format"""
import numpy as np
import cv2
import os
from glob import glob

# Find the most recent output
output_dirs = sorted(glob('outputs/mpc_run_*'))
if output_dirs:
    latest_dir = output_dirs[-1]
    
    # Try to load a saved "broken" image to analyze
    images = sorted(glob(os.path.join(latest_dir, 'images', '*.png')))
    if images:
        img = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)
        print(f"Saved image: {images[0]}")
        print(f"  Shape: {img.shape}")
        print(f"  Dtype: {img.dtype}")
        print(f"  Min/Max per channel:")
        if len(img.shape) == 3:
            for i in range(img.shape[2]):
                print(f"    Channel {i}: [{img[:,:,i].min()}, {img[:,:,i].max()}]")
        
        # Check if there's pattern in the data
        print(f"\n  Sample pixel values (first 5 pixels):")
        print(img[360, 360:365])
        
    # Load goal image to compare
    goal_img = cv2.imread(os.path.join(latest_dir, 'goal.png'))
    if goal_img is not None:
        print(f"\nGoal image:")
        print(f"  Shape: {goal_img.shape}")
        print(f"  Min/Max per channel:")
        for i in range(goal_img.shape[2]):
            print(f"    Channel {i}: [{goal_img[:,:,i].min()}, {goal_img[:,:,i].max()}]")

# Now let's see what the raw_obs format would look like
print("\n" + "="*50)
print("Testing conversion logic:")
print("="*50)

# Simulate what render returns (RGBA + depth)
test_render = np.zeros((720, 720, 5), dtype=np.uint8)
test_render[..., :3] = 128  # Gray RGB
test_render[..., 3] = 255   # Alpha fully opaque
test_render[..., 4] = 100   # Some depth value

# This is what gets stored in raw_obs (float64 array)
raw_obs_like = np.zeros((1, 720, 720, 5), dtype=np.float64)
raw_obs_like[0] = test_render

print(f"\nSimulated raw_obs:")
print(f"  Dtype: {raw_obs_like.dtype}")
print(f"  RGB values: [{raw_obs_like[0, :, :, :3].min()}, {raw_obs_like[0, :, :, :3].max()}]")

# Test extraction
rgb = raw_obs_like[0, ..., :3].copy()
print(f"\nExtracted RGB:")
print(f"  Dtype: {rgb.dtype}")
print(f"  Values: [{rgb.min()}, {rgb.max()}]")

# Test our conversion
if rgb.dtype in [np.float32, np.float64]:
    if rgb.max() <= 1.0:
        rgb_converted = (rgb * 255).astype(np.uint8)
    else:
        rgb_converted = np.clip(rgb, 0, 255).astype(np.uint8)
    
print(f"\nConverted RGB:")
print(f"  Dtype: {rgb_converted.dtype}")
print(f"  Values: [{rgb_converted.min()}, {rgb_converted.max()}]")

# Convert to BGR
bgr = rgb_converted[..., ::-1]
print(f"\nBGR for cv2:")
print(f"  Shape: {bgr.shape}")
print(f"  Values: [{bgr.min()}, {bgr.max()}]")
