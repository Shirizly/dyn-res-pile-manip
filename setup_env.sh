#!/bin/bash
# Environment setup for dyn-res-pile-manip
# Source this file before running the code: source setup_env.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Force NVIDIA GPU rendering (for hybrid graphics systems)
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only

# CUDA libraries (for DGL)
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH}

# PyFleX
export PYFLEXROOT=${SCRIPT_DIR}/PyFleX
export PYTHONPATH=${PYFLEXROOT}/bindings/build:${PYTHONPATH}
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:${LD_LIBRARY_PATH}

echo "✓ Environment configured for dyn-res-pile-manip"
echo "  - NVIDIA GPU rendering enabled"
echo "  - CUDA libraries loaded"
echo "  - PyFleX bindings loaded"
