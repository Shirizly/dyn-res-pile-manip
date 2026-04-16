#!/usr/bin/env python
import sys
import pyflex
import numpy as np

print("Testing PyFleX render dimensions...")

# Initialize with 720x720
pyflex.set_screenWidth(720)
pyflex.set_screenHeight(720)
pyflex.set_light_dir(np.array([0.1, 2.0, 0.1]))
pyflex.set_light_fov(70.)

print(f"Set screen dimensions to: 720x720")

# Try headless first
print("\nInitializing PyFleX in headless mode...")
pyflex.init(True)  # Positional argument, True = headless

# Try rendering
print("Calling pyflex.render(render_depth=True)...")
render_output = pyflex.render(render_depth=True)

print(f"\nRender output:")
print(f"  Type: {type(render_output)}")
print(f"  Shape: {render_output.shape if hasattr(render_output, 'shape') else 'N/A'}")
print(f"  Size: {render_output.size if hasattr(render_output, 'size') else len(render_output)}")
print(f"  Dtype: {render_output.dtype if hasattr(render_output, 'dtype') else type(render_output[0])}")

# Calculate expected vs actual
expected = 720 * 720 * 5
actual = render_output.size if hasattr(render_output, 'size') else len(render_output)
print(f"\nExpected size: {expected}")
print(f"Actual size: {actual}")
print(f"Ratio: {actual / expected:.2f}x")

# Try to infer dimensions
if actual % 5 == 0:
    pixels = actual // 5
    print(f"\nIf 5 channels, total pixels: {pixels}")
    
    # Try to find dimensions
    for w in [720, 1280, 1920, 2560, 3840]:
        if pixels % w == 0:
            h = pixels // w
            print(f"  Possible: {w}x{h}")

pyflex.clean()
print("\nPyFleX cleaned up.")
