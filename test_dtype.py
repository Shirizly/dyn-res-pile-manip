#!/usr/bin/env python
"""Quick test to check render output format"""
import numpy as np

# Check what format np.zeros creates
test_array = np.zeros((2, 720, 720, 5))
print(f"Default np.zeros dtype: {test_array.dtype}")

# Simulate what happens when we assign different types to it
test_uint8 = np.ones((720, 720, 5), dtype=np.uint8) * 100
test_array[0] = test_uint8
print(f"After assigning uint8 data: {test_array.dtype}")
print(f"Values range: [{test_array[0, :, :, :3].min()}, {test_array[0, :, :, :3].max()}]")

# What if we try to convert this back?
rgb = test_array[0, :, :, :3]
print(f"\nExtracted RGB dtype: {rgb.dtype}")
rgb_uint8 = rgb.astype(np.uint8)
print(f"After .astype(uint8): min={rgb_uint8.min()}, max={rgb_uint8.max()}")

# This is the issue! numpy zeros creates float64 by default
# When we assign uint8 data to it, it converts to float64
# So we need to handle this conversion properly
