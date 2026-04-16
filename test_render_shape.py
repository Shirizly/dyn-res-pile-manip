#!/usr/bin/env python3
"""Quick test to determine actual pyflex render output format"""
import sys
import os
import numpy as np
sys.path.append(os.environ.get('PYFLEXROOT', ''))
import pyflex

# Initialize with minimal config
pyflex.set_screenWidth(720)
pyflex.set_screenHeight(720)
pyflex.init(False)  # False = not headless

# Test render (no scene needed, just check output format)
print("Testing pyflex.render()...")
pyflex.step()  # Need at least one step before rendering
render_output = pyflex.render(render_depth=True)
print(f"Render output size: {render_output.size}")
print(f"Render output shape before reshape: {render_output.shape}")
print(f"Render output dtype: {render_output.dtype}")
print(f"Expected size for 720x720x5: {720 * 720 * 5}")
print(f"Expected size for 720x720x4: {720 * 720 * 4}")

# Calculate what dimensions this could be
total_size = render_output.size
print(f"\nTrying to factor {total_size}:")

# Try different channel counts
for channels in [4, 5, 6]:
    pixels = total_size / channels
    if pixels == int(pixels):
        pixels = int(pixels)
        sqrt_pixels = np.sqrt(pixels)
        print(f"  {channels} channels: {pixels} pixels, sqrt={sqrt_pixels:.2f}")
        
        # Try to find integer factors near sqrt
        for h in range(int(sqrt_pixels) - 5, int(sqrt_pixels) + 6):
            if pixels % h == 0:
                w = pixels // h
                print(f"    -> Could be {h}x{w}x{channels} = {h * w * channels}")
