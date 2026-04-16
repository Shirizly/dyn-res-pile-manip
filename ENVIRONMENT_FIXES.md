# Environment Setup Fixes

## Quick Start (Automated Installation)

**For a one-command installation, use the automated script:**

```bash
chmod +x install_all.sh
./install_all.sh
```

This script will:
1. Check and install Docker if needed
2. Install NVIDIA Container Toolkit
3. Create conda environment with Python 3.9
4. Install all Python packages with correct versions
5. Compile PyFleX inside Docker
6. Download data and models
7. Configure environment variables

**Note:** If the script asks you to reboot (for Docker group membership or NVIDIA driver issues), please do so and run the script again.

After installation, use the project with:
```bash
conda activate dyn-res-pile-manip
source setup_env.sh
python visualize_mpc.py
```

---

## Manual Installation & Troubleshooting

The following sections document the detailed changes and manual installation steps for troubleshooting purposes.

## Summary of Changes Made to Get visualize_mpc.py Running

### Package Version Issues Fixed

1. **PyTorch Downgrade**: 
   - Changed from PyTorch 2.8.0 → 2.0.1 with CUDA 11.8 support
   - Command: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`

2. **DGL with CUDA Support**:
   - Installed DGL 2.1.0 with CUDA 11.8 bindings (instead of CPU-only version)
   - Command: `pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu118/repo.html`

3. **torchdata Downgrade**:
   - Changed from torchdata 0.11.0 → 0.7.1
   - Reason: DGL 2.1.0 requires torchdata.datapipes module which was removed in 0.11.0
   - Command: `pip install torchdata==0.7.1`

4. **pydantic Installation**:
   - Added pydantic (required by DGL graphbolt module)
   - Command: `pip install pydantic`

5. **CUDA Libraries**:
   - Installed nvidia-cusparse-cu11 (required by DGL CUDA operations)
   - Command: `pip install nvidia-cusparse-cu11`

6. **NumPy Downgrade**:
   - Changed from numpy 2.0.2 → 1.26.4
   - Reason: DGL and other compiled modules were built against NumPy 1.x API
   - Command: `pip install 'numpy<2'`

### Environment Variables Required

```bash
# Force NVIDIA GPU rendering (for hybrid graphics systems with Intel + NVIDIA)
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only

# CUDA libraries (for DGL)
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH

# PyFleX (after installation)
export PYFLEXROOT=/home/shirizly/Code/GranularBaselines/dyn-res-pile-manip/PyFleX
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
```

**Important:** The NVIDIA GPU rendering variables are critical for hybrid graphics systems (laptops with both Intel and NVIDIA GPUs). Without these, PyFleX will try to use Intel Mesa drivers which don't support the required OpenGL extensions, resulting in shader compilation errors and segmentation faults.

### Docker Setup (Required for PyFleX Installation)

PyFleX requires compilation inside a Docker container with specific build tools and dependencies.

#### 1. Install Docker

```bash
# Check if Docker is installed
docker --version

# Install Docker Engine if not present
sudo apt-get update
sudo apt-get install docker.io

# Add user to docker group (to avoid sudo)
sudo usermod -aG docker $USER

# Apply group membership (choose one):
newgrp docker  # Immediate, current shell only
# OR logout and login again
# OR reboot system
```

#### 2. Install NVIDIA Container Toolkit

Required for GPU support inside Docker containers:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 3. Verify Docker Setup

```bash
# Test Docker access (should work without sudo)
docker run hello-world

# Test NVIDIA Docker integration
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

#### 4. Common Docker Issues

**"Permission denied" error:**
- Make sure you added user to docker group and logged out/in
- Or run: `sudo usermod -aG docker $USER && newgrp docker`

**"nvidia-container-cli: driver/library version mismatch":**
- NVIDIA kernel driver and userspace libraries are out of sync
- **Solution:** Reboot your system (`sudo reboot`)
- This commonly happens after driver updates

**"Cannot connect to Docker daemon":**
- Start Docker service: `sudo systemctl start docker`
- Enable on boot: `sudo systemctl enable docker`

### PyFleX Installation

#### Prerequisites
- Docker installed and configured (see Docker Setup above)
- X11 display for graphics (usually already available on Linux desktop)

#### Installation Steps

```bash
# Navigate to project root
cd /home/shirizly/Code/GranularBaselines/dyn-res-pile-manip

# Ensure conda environment is active
conda activate dyn-res-pile-manip

# Make installation script executable
chmod +x scripts/install_pyflex.sh

# Run the installation (this pulls the Docker image and compiles PyFleX)
./scripts/install_pyflex.sh
```

#### What install_pyflex.sh Does

1. **Pulls Docker image** `xingyu/softgym` (~1.5GB download first time)
2. **Mounts directories** into container:
   - PyFleX source code
   - Your conda environment
   - X11 socket for graphics
3. **Compiles PyFleX** using CMake and Make
4. **Adds environment variables** to `~/.bashrc`:
   - `PYFLEXROOT` - Path to PyFleX installation
   - `PYTHONPATH` - So Python finds compiled bindings
   - `LD_LIBRARY_PATH` - For SDL2 library dependency

#### Post-Installation

After successful installation, reload your shell or run:

```bash
source ~/.bashrc
```

The PyFleX environment variables will be automatically loaded in new shell sessions.

### Custom Robot Models

The project requires custom robot models with specific mesh files that aren't included in the standard PyBullet package. These must be downloaded separately:

```bash
chmod +x scripts/custom_robot_model.sh
bash scripts/custom_robot_model.sh
```

**What this downloads:**
- Custom Franka Panda robot model with all visual and collision meshes
- Custom Kinova robot model

**Symptom if missing:** Segmentation fault during `env.reset()` when loading robot URDF, specifically in `FlexRobotHelper.loadURDF()` at the `pyflex.add_mesh()` call.

### Data Download

The project requires pre-trained models and possibly other data files.

```bash
# Navigate to project root
cd /home/shirizly/Code/GranularBaselines/dyn-res-pile-manip

# Make download script executable
chmod +x scripts/download_data.sh

# Run the download script
./scripts/download_data.sh
```

This script typically downloads:
- Pre-trained GNN dynamics model weights
- Initial action sequences
- Configuration files
- Test scenarios/datasets

## Recommended env.yaml Structure

```yaml
name: dyn-res-pile-manip
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.9
  - pip=22.2.2
  - pybind11
  - cmake  # Required for PyFleX compilation
  - pip:
    # PyTorch with CUDA 11.8
    - torch==2.0.1
    - torchvision==0.15.2
    
    # DGL with compatible versions
    - dgl==2.1.0  # Install with: pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu118/repo.html
    - torchdata==0.7.1
    - pydantic
    
    # CUDA libraries
    - nvidia-cusparse-cu11
    
    # Core dependencies
    - numpy<2  # CRITICAL: Must be <2 for compatibility
    - scipy
    - opencv-python
    - matplotlib
    - pybullet
    - gym  # Note: Shows deprecation warning, consider migrating to gymnasium
    - beautifulsoup4
    - open3d
    - PyYAML
    - lxml
    - gdown
    - scikit-optimize

environment_variables:
  - LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH}
```

## Complete Installation Order

### 1. System Prerequisites

```bash
# Install Docker
sudo apt-get update
sudo apt-get install docker.io
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify Docker
docker run hello-world
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 2. Create Conda Environment

```bash
cd /home/shirizly/Code/GranularBaselines/dyn-res-pile-manip
conda create -n dyn-res-pile-manip python=3.9
conda activate dyn-res-pile-manip
```

### 3. Install Python Packages

```bash
# Install PyTorch 2.0.1 with CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install DGL 2.1.0 with CUDA 11.8
pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu118/repo.html

# Install compatible torchdata
pip install torchdata==0.7.1

# Install other dependencies
pip install pydantic nvidia-cusparse-cu11
pip install 'numpy<2'

# Install remaining packages from requirements-fixed.txt
pip install -r requirements-fixed.txt
```

### 4. Install PyFleX

```bash
# Make sure you're in project root
cd /home/shirizly/Code/GranularBaselines/dyn-res-pile-manip

# Run PyFleX installation
chmod +x scripts/install_pyflex.sh
./scripts/install_pyflex.sh

# Reload shell to apply environment variables
source ~/.bashrc
```

### 5. Download Custom Robot Models

```bash
# Download Franka and Kinova robot models with custom meshes
chmod +x scripts/custom_robot_model.sh
bash scripts/custom_robot_model.sh
```

### 6. Download Data

```bash
# Download pre-trained models and data
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

### 6. Setup Environment Helper Script

The `setup_env.sh` script is already created to automatically set all required environment variables:

```bash
# Use this before running the code
source setup_env.sh
```

## Quick Test

To verify the environment is set up correctly:

```bash
# Activate environment
conda activate dyn-res-pile-manip

# Source environment variables
source setup_env.sh

# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import dgl; print(f'DGL: {dgl.__version__}')"
python -c "from dgl.geometry import farthest_point_sampler; print('DGL CUDA: OK')"
python -c "import pyflex; print('PyFleX: OK')"
```

## Usage

After completing all installation steps:

```bash
# Activate environment
conda activate dyn-res-pile-manip

# Set environment variables
source setup_env.sh

# Run the visualization script
python visualize_mpc.py
```

## Current Status Checklist

✅ PyTorch 2.0.1 with CUDA 11.8 installed  
✅ DGL 2.1.0 with CUDA bindings installed  
✅ torchdata 0.7.1 (compatible version) installed  
✅ pydantic installed  
✅ nvidia-cusparse-cu11 installed  
✅ NumPy downgraded to 1.26.4  
✅ NVIDIA GPU rendering environment variables configured  
✅ setup_env.sh helper script created  
✅ requirements-fixed.txt created  
✅ Docker and NVIDIA Container Toolkit setup documented  
✅ Custom robot models (Franka, Kinova) download script  
✅ Debug tools created (debug_visualize.py, test_pyflex_render.py)  
⚠️ PyFleX - needs compilation via scripts/install_pyflex.sh  
⚠️ Custom robot models - needs download via scripts/custom_robot_model.sh  
⚠️ Data download - needs to run scripts/download_data.sh  
⚠️ PyFleX render size mismatch - needs investigation/workaround  
⚠️ PyFleX headless mode - has OpenGL shader issues, use windowed mode  

## Known Issues

### 1. Segmentation Fault During env.reset()
**Cause:** Missing custom robot models  
**Solution:** Run `bash scripts/custom_robot_model.sh`  
**Fixed:** ✅

### 2. OpenGL Shader Errors (gl_PositionIn undeclared)
**Cause:** System using Intel integrated graphics instead of NVIDIA GPU  
**Solution:** Set NVIDIA rendering environment variables (included in setup_env.sh)  
**Fixed:** ✅

### 3. PyFleX Headless Mode Crashes
**Cause:** OpenGL shader issues in headless/EGL mode  
**Workaround:** Use windowed mode (`headless: False` in config) or `xvfb-run`  
**Status:** ⚠️ Known limitation

### 4. Render Buffer Size Mismatch
**Cause:** Display scaling/HiDPI causes PyFleX to return actual window manager buffer (e.g., 1440×1440 from 2x scaling) instead of requested 720×720  
**Solution:** Modified `env/flex_env.py` render() method to auto-detect actual dimensions and resize to expected size using cv2.resize  
**Fixed:** ✅

**Technical Details:**
```python
# Auto-detect actual render buffer size
total_size = render_output.size
expected_size = self.screenHeight * self.screenWidth * 5

if total_size != expected_size:
    # Calculate actual dimensions (buffer is width×height×5 channels)
    actual_pixels = total_size // 5
    actual_height = int(np.sqrt(actual_pixels))
    actual_width = actual_pixels // actual_height
    
    # Reshape with actual dimensions, then resize to expected
    img = render_output.reshape(actual_height, actual_width, 5)
    img_resized = cv2.resize(img[..., :4], (self.screenWidth, self.screenHeight), 
                             interpolation=cv2.INTER_AREA)
    depth_resized = cv2.resize(img[..., 4:5], (self.screenWidth, self.screenHeight), 
                               interpolation=cv2.INTER_NEAREST)
    return np.concatenate([img_resized, depth_resized], axis=2)
```  

## Troubleshooting

### Docker Issues

**Issue:** "permission denied while trying to connect to Docker daemon"  
**Solution:** 
```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Issue:** "nvidia-container-cli: driver/library version mismatch"  
**Solution:** Reboot your system (`sudo reboot`)

### PyFleX Issues

**Issue:** PyFleX import fails after installation  
**Solution:** 
```bash
source ~/.bashrc
source setup_env.sh
```

**Issue:** "libSDL2-2.0.so.0: cannot open shared object file"  
**Solution:** LD_LIBRARY_PATH not set correctly, run `source setup_env.sh`

### CUDA Issues

**Issue:** "CUDA out of memory"  
**Solution:** Reduce batch size in config/mpc/config.yaml

**Issue:** DGL operations fail with CUDA errors  
**Solution:** Verify CUDA toolkit version matches PyTorch: `nvcc --version` should show 11.8

### OpenGL / PyFleX Rendering Issues

**Issue:** Segmentation fault or shader compilation errors like:
```
error: `gl_PositionIn' undeclared
error: `gl_TexCoordIn' undeclared
warning: extension `GL_EXT_geometry_shader4' unsupported
```

**Cause:** System is using Intel integrated graphics (Mesa) instead of NVIDIA GPU for rendering. This is common on laptops with hybrid graphics (Optimus/PRIME).

**Diagnosis:**
```bash
glxinfo | grep "OpenGL renderer"
# If it shows "Intel" or "Mesa", you're not using NVIDIA GPU
```

**Solution:** Force NVIDIA GPU rendering by setting environment variables:
```bash
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
```

These are automatically set in `setup_env.sh`, so make sure to source it:
```bash
source setup_env.sh
```

**Verify Fix:**
```bash
# After sourcing setup_env.sh, check renderer
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia glxinfo | grep "OpenGL renderer"
# Should now show "NVIDIA"
```

**Alternative for Headless Systems:** If you don't have a display or need to run headless:
```bash
# Install xvfb if not present
sudo apt-get install xvfb

# Run with virtual framebuffer
xvfb-run -a python visualize_mpc.py
```

**Note:** PyFleX headless mode has known issues with OpenGL shader compilation. For reliable operation, run with a display (`headless: False` in config) or use `xvfb-run`.

### PyFleX Render Size Mismatch

**Issue:** `ValueError: cannot reshape array of size X into shape (720,720,5)` 

**Cause:** PyFleX may return a different render buffer size than requested, possibly due to display scaling, retina/HiDPI displays, or window manager settings.

**Diagnosis:**
```bash
# Run the diagnostic script
python test_pyflex_render.py
```

**Solutions:**
1. Check if config has correct screen dimensions matching your display
2. Modify `flex_env.py` to dynamically detect render output size
3. Ensure `headless: False` in config for windowed mode (more reliable than headless)

### Debugging Tools

A debug wrapper script `debug_visualize.py` has been created to provide better diagnostics:

```bash
conda activate dyn-res-pile-manip
source setup_env.sh
python debug_visualize.py
```

This script:
- Enables Python `faulthandler` for segfault stack traces
- Catches and prints full exception tracebacks
- Logs progress through execution
- Helps identify where crashes occur

For even more detailed debugging with core dumps:
```bash
ulimit -c unlimited  # Enable core dumps
python debug_visualize.py
# If crash, analyze with: gdb python core
```

## Notes

- The uncolored imports in VS Code were due to missing/incompatible packages
- ModuleNotFoundError for torchdata.datapipes was due to version mismatch
- CUDA runtime errors were due to missing CUDA library paths
- NumPy 2.0 compatibility breaks many scientific packages compiled against 1.x
- PyFleX requires Docker because it has complex C++ dependencies that are pre-configured in the xingyu/softgym container
- NVIDIA driver/library mismatches typically require a system reboot to resolve
