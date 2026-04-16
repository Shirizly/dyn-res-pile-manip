# Code Fixes Required for dyn-res-pile-manip

## 1. Render Buffer Size Mismatch Fix

**File:** `env/flex_env.py`  
**Method:** `render()` (lines ~891-920)  
**Issue:** PyFleX creates render window at actual display resolution (accounting for HiDPI/display scaling), not requested size. For example, requesting 720×720 may result in 2408×2868 buffer.

**Fix:** Auto-detect actual render buffer dimensions and resize to expected size:
- Calculate actual dimensions by factoring `render_output.size / 5`
- Find factor pair closest to square aspect ratio
- Reshape with actual dimensions
- Resize to expected `screenWidth × screenHeight` using `cv2.resize()`
- Handle depth channel dimension properly (add axis if needed)

**Code Changes:**
```python
# Instead of directly reshaping:
# render_output.reshape(self.screenHeight, self.screenWidth, 5)

# Now checks size first and auto-adapts:
if total_size != expected_size:
    # Find actual dimensions
    actual_pixels = total_size // 5
    # Find best factor pair (closest to square)
    # ... factorization logic ...
    # Reshape and resize
    img = render_output.reshape(actual_height, actual_width, 5)
    img_resized = cv2.resize(img[..., :4], (self.screenWidth, self.screenHeight))
    depth_resized = cv2.resize(img[..., 4], (self.screenWidth, self.screenHeight))
    if depth_resized.ndim == 2:
        depth_resized = depth_resized[..., np.newaxis]
    return np.concatenate([img_resized, depth_resized], axis=2)
```

## 2. Custom Robot Models

**Files:** `scripts/custom_robot_model.sh`  
**Issue:** Default PyBullet robot models lack proper collision meshes, causing segmentation faults.

**Fix:** Download custom Franka and Kinova models via provided script:
```bash
bash scripts/custom_robot_model.sh
```

## Summary

The main issue was PyFleX rendering at native display resolution instead of requested size. The fix dynamically detects and adapts to actual render buffer dimensions, making the code robust across different display configurations.
