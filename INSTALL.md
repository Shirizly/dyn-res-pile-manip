# Installation Guide

## Quick Installation (Recommended)

Run the automated installation script:

```bash
chmod +x install_all.sh
./install_all.sh
```

This single script handles:
- ✓ Docker installation and configuration
- ✓ NVIDIA Container Toolkit setup
- ✓ Conda environment creation
- ✓ All Python packages (PyTorch, DGL, etc.)
- ✓ PyFleX compilation
- ✓ Data download
- ✓ Environment variable setup

**Important:** If the script prompts you to reboot (for Docker or NVIDIA driver issues), please reboot and run the script again.

## Usage After Installation

```bash
# Activate the environment
conda activate dyn-res-pile-manip

# Load environment variables
source setup_env.sh

# Run your code
python visualize_mpc.py
```

## What's Installed

- **Python 3.9** with conda
- **PyTorch 2.0.1** with CUDA 11.8 support
- **DGL 2.1.0** with CUDA 11.8 bindings
- **PyFleX** compiled from source
- **All dependencies** from requirements-fixed.txt
- **Pre-trained models** and data files

## Files Created

- `setup_env.sh` - Environment configuration script (must be sourced each session)
- `requirements-fixed.txt` - Python package requirements
- `ENVIRONMENT_FIXES.md` - Detailed troubleshooting guide

## Troubleshooting

If you encounter issues, see [ENVIRONMENT_FIXES.md](ENVIRONMENT_FIXES.md) for:
- Manual installation steps
- Common error solutions
- Detailed explanation of all changes
- Docker and NVIDIA setup troubleshooting

## Manual Installation

If you prefer manual installation or need to troubleshoot, follow the detailed steps in [ENVIRONMENT_FIXES.md](ENVIRONMENT_FIXES.md).

## System Requirements

- Ubuntu/Debian Linux
- NVIDIA GPU with CUDA support
- At least 10GB free disk space (for Docker image and packages)
- Conda/Anaconda installed
- For hybrid graphics systems (Intel + NVIDIA): NVIDIA drivers must be installed

**Note for Hybrid Graphics Systems (Laptops):**  
If you have both Intel and NVIDIA GPUs, the `setup_env.sh` script automatically configures the system to use the NVIDIA GPU for rendering, which is required for PyFleX. Without this, you'll get OpenGL shader errors.
