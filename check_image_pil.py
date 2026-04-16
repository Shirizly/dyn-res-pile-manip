#!/usr/bin/env python
"""Check render output without cv2"""
import numpy as np
from PIL import Image
import os
from glob import glob

# Find the most recent output
output_dirs = sorted(glob('outputs/mpc_run_*'))
if output_dirs:
    latest_dir = output_dirs[-1]
    print(f"Checking directory: {latest_dir}\n")
    
    # Try to load a saved "broken" image to analyze
    images = sorted(glob(os.path.join(latest_dir, 'images', '*.png')))
    if images:
        img = Image.open(images[0])
        img_array = np.array(img)
        print(f"Saved image: {os.path.basename(images[0])}")
        print(f"  Shape: {img_array.shape}")
        print(f"  Dtype: {img_array.dtype}")
        print(f"  Mode: {img.mode}")
        print(f"  Min/Max per channel:")
        if len(img_array.shape) == 3:
            for i in range(img_array.shape[2]):
                print(f"    Channel {i}: [{img_array[:,:,i].min()}, {img_array[:,:,i].max()}]")
        
        # Check a few different regions
        print(f"\n  Sample pixels from center (360, 360-364):")
        print(img_array[360, 360:365])
        
        print(f"\n  Sample pixels from top-left (10, 10-14):")
        print(img_array[10, 10:15])
        
        # Save a small crop for manual inspection
        crop = img_array[300:420, 300:420]
        Image.fromarray(crop).save('debug_crop.png')
        print(f"\n  Saved debug_crop.png for inspection")
