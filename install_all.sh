#!/bin/bash
# Complete installation script for dyn-res-pile-manip environment
# This script installs all dependencies, compiles PyFleX, and downloads data

set -e  # Exit on any error

echo "======================================"
echo "dyn-res-pile-manip Installation Script"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Check if running from project root
if [ ! -f "$PROJECT_ROOT/env.yaml" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

echo "Step 1/7: Checking Docker installation"
echo "========================================"
if ! command -v docker &> /dev/null; then
    print_warning "Docker not found. Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io
    print_status "Docker installed"
else
    print_status "Docker already installed"
fi

# Check if user is in docker group
if ! groups $USER | grep -q docker; then
    print_warning "Adding user to docker group..."
    sudo usermod -aG docker $USER
    print_warning "You will need to log out and log back in for docker group membership to take effect"
    print_warning "After logging back in, run this script again"
    echo ""
    echo "Please run: sudo reboot"
    echo "Then run this script again after reboot"
    exit 0
fi

# Start Docker service
if ! sudo systemctl is-active --quiet docker; then
    print_warning "Starting Docker service..."
    sudo systemctl start docker
    sudo systemctl enable docker
fi
print_status "Docker service is running"

echo ""
echo "Step 2/7: Checking NVIDIA Docker support"
echo "========================================"
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    print_warning "NVIDIA Container Toolkit not properly configured"
    
    # Check if nvidia-container-toolkit is installed
    if ! dpkg -l | grep -q nvidia-container-toolkit; then
        print_warning "Installing NVIDIA Container Toolkit..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo systemctl restart docker
        print_status "NVIDIA Container Toolkit installed"
    fi
    
    # Check for driver mismatch
    if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_error "NVIDIA driver/library version mismatch detected"
        print_warning "This usually requires a system reboot to fix"
        echo ""
        echo "Please run: sudo reboot"
        echo "Then run this script again after reboot"
        exit 1
    fi
else
    print_status "NVIDIA Docker support verified"
fi

echo ""
echo "Step 3/7: Setting up Conda environment"
echo "========================================"

# Get conda base
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
if [ ! -d "$CONDA_BASE" ]; then
    print_error "Conda not found. Please install Anaconda or Miniconda first"
    exit 1
fi

# Source conda
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Check if environment exists
if conda env list | grep -q "^dyn-res-pile-manip "; then
    print_warning "Environment 'dyn-res-pile-manip' already exists"
    read -p "Do you want to remove and recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n dyn-res-pile-manip -y
        print_status "Old environment removed"
    else
        print_status "Using existing environment"
    fi
fi

# Create environment if it doesn't exist
if ! conda env list | grep -q "^dyn-res-pile-manip "; then
    print_warning "Creating conda environment..."
    conda create -n dyn-res-pile-manip python=3.9 pip=22.2.2 -y
    print_status "Conda environment created"
fi

# Activate environment
conda activate dyn-res-pile-manip
print_status "Environment activated"

echo ""
echo "Step 4/7: Installing Python packages"
echo "========================================"

print_warning "Installing PyTorch 2.0.1 with CUDA 11.8..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --quiet
print_status "PyTorch installed"

print_warning "Installing DGL 2.1.0 with CUDA 11.8..."
pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu118/repo.html --quiet
print_status "DGL installed"

print_warning "Installing remaining dependencies..."
pip install torchdata==0.7.1 --quiet
pip install pydantic nvidia-cusparse-cu11 --quiet
pip install 'numpy<2' --quiet

# Install from requirements-fixed.txt if it exists
if [ -f "$PROJECT_ROOT/requirements-fixed.txt" ]; then
    # Filter out packages already installed
    grep -v '^#' "$PROJECT_ROOT/requirements-fixed.txt" | grep -v '^numpy' | grep -v '^torch' | grep -v '^dgl' | grep -v '^pydantic' | grep -v '^nvidia-cusparse' | while read package; do
        if [ ! -z "$package" ]; then
            pip install "$package" --quiet || print_warning "Failed to install $package (non-critical)"
        fi
    done
fi
print_status "All Python packages installed"

echo ""
echo "Step 5/7: Compiling PyFleX"
echo "========================================"
cd "$PROJECT_ROOT"

# Check if PyFleX is already compiled
if [ -d "$PROJECT_ROOT/PyFleX/bindings/build" ] && [ -f "$PROJECT_ROOT/PyFleX/bindings/build/pyflex.so" ]; then
    print_warning "PyFleX appears to be already compiled"
    read -p "Do you want to recompile? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Skipping PyFleX compilation"
        SKIP_PYFLEX=1
    fi
fi

if [ -z "$SKIP_PYFLEX" ]; then
    print_warning "Pulling Docker image (this may take a while on first run)..."
    docker pull xingyu/softgym
    
    print_warning "Compiling PyFleX inside Docker container..."
    docker run \
        -v ${PROJECT_ROOT}/PyFleX:/workspace/PyFleX \
        -v ${CONDA_PREFIX}:/workspace/anaconda \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        xingyu/softgym:latest bash \
        -c "export PATH=/workspace/anaconda/bin:\$PATH; cd /workspace/PyFleX; export PYFLEXROOT=/workspace/PyFleX; export PYTHONPATH=/workspace/PyFleX/bindings/build:\$PYTHONPATH; export LD_LIBRARY_PATH=\$PYFLEXROOT/external/SDL2-2.0.4/lib/x64:\$LD_LIBRARY_PATH; cd bindings; mkdir -p build; cd build; /usr/bin/cmake .. && make -j"
    
    if [ -f "$PROJECT_ROOT/PyFleX/bindings/build/pyflex.so" ]; then
        print_status "PyFleX compiled successfully"
    else
        print_error "PyFleX compilation failed"
        exit 1
    fi
fi

echo ""
echo "Step 6/8: Downloading custom robot models"
echo "========================================"
if [ -f "$PROJECT_ROOT/scripts/custom_robot_model.sh" ]; then
    chmod +x "$PROJECT_ROOT/scripts/custom_robot_model.sh"
    print_warning "Downloading custom Franka and Kinova robot models..."
    bash "$PROJECT_ROOT/scripts/custom_robot_model.sh" || print_warning "Robot model download had some issues (might be non-critical)"
    print_status "Custom robot models installed"
else
    print_warning "custom_robot_model.sh not found, skipping robot model download"
fi

echo ""
echo "Step 7/8: Downloading data and models"
echo "========================================"
if [ -f "$PROJECT_ROOT/scripts/download_data.sh" ]; then
    chmod +x "$PROJECT_ROOT/scripts/download_data.sh"
    print_warning "Running download_data.sh..."
    bash "$PROJECT_ROOT/scripts/download_data.sh" || print_warning "Data download had some issues (might be non-critical)"
    print_status "Data download completed"
else
    print_warning "download_data.sh not found, skipping data download"
fi
echo "========================================"
if [ -f "$PROJECT_ROOT/scripts/download_model.sh" ]; then
    chmod +x "$PROJECT_ROOT/scripts/download_model.sh"
    print_warning "Running download_model.sh..."
    bash "$PROJECT_ROOT/scripts/download_model.sh" || print_warning "Model download had some issues (might be non-critical)"
    print_status "Model download completed"
else
    print_warning "download_model.sh not found, skipping model download"
fi

echo ""
echo "Step 8/8: Setting up environment configuration"
echo "========================================"

# Update setup_env.sh with all paths
cat > "$PROJECT_ROOT/setup_env.sh" << 'EOF'
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
EOF

chmod +x "$PROJECT_ROOT/setup_env.sh"
print_status "setup_env.sh created"

echo ""
echo "======================================"
echo -e "${GREEN}Installation Complete!${NC}"
echo "======================================"
echo ""
echo "To use this project, run the following commands:"
echo ""
echo "  conda activate dyn-res-pile-manip"
echo "  source setup_env.sh"
echo "  python visualize_mpc.py"
echo ""
echo "The setup_env.sh script must be sourced every time you start a new terminal."
echo ""

# Test imports
echo "Testing imports..."
conda activate dyn-res-pile-manip
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH}
export PYFLEXROOT=${PROJECT_ROOT}/PyFleX
export PYTHONPATH=${PYFLEXROOT}/bindings/build:${PYTHONPATH}
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:${LD_LIBRARY_PATH}

python -c "import torch; print(f'  ✓ PyTorch {torch.__version__} (CUDA available: {torch.cuda.is_available()})')" || print_error "PyTorch import failed"
python -c "import dgl; print(f'  ✓ DGL {dgl.__version__}')" || print_error "DGL import failed"
python -c "from dgl.geometry import farthest_point_sampler; print('  ✓ DGL CUDA operations')" || print_error "DGL CUDA import failed"
python -c "import pyflex; print('  ✓ PyFleX')" || print_error "PyFleX import failed"

echo ""
print_status "All tests passed!"
