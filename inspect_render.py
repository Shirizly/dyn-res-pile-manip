#!/usr/bin/env python3
"""Inspect what pyflex.render actually returns"""
import numpy as np
import sys
import os

# Setup paths
sys.path.append(os.getcwd())
from utils import load_yaml

# Load config
config = load_yaml('config/mpc/config.yaml')

# Import after config
sys.path.append(os.environ.get('PYFLEXROOT', ''))
import pyflex

# Initialize PyFleX
pyflex.set_screenWidth(720)
pyflex.set_screenHeight(720)
pyflex.init(config['dataset']['headless'])

# Get render output and inspect
print("Calling pyflex.render(render_depth=True)...")
result = pyflex.render(render_depth=True)

print(f"\nResult type: {type(result)}")

if isinstance(result, tuple):
    print(f"Result is a TUPLE with {len(result)} elements:")
    for i, item in enumerate(result):
        print(f"  [{i}] type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'N/A'}, size={item.size if hasattr(item, 'size') else 'N/A'}")
else:
    print(f"Result is NOT a tuple")
    print(f"  shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
    print(f"  size: {result.size if hasattr(result, 'size') else 'N/A'}")
    print(f"  dtype: {result.dtype if hasattr(result, 'dtype') else 'N/A'}")
    
    # Try to understand the structure
    total_size = result.size
    print(f"\nAnalyzing structure (total size: {total_size}):")
    print(f"  Expected 720x720x5: {720 * 720 * 5}")
    print(f"  Expected 720x720x4: {720 * 720 * 4}")
    print(f"  Ratio: {total_size / (720 * 720 * 5):.2f}x")
    
    # Check if it could be interpreted as different channel layouts
    for channels in [1, 3, 4, 5, 6]:
        if total_size % channels == 0:
            pixels = total_size // channels
            sqrt_pixels = np.sqrt(pixels)
            print(f"  If {channels} channels: {pixels} pixels, sqrt={sqrt_pixels:.1f}")
