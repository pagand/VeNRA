#!/bin/bash
# VeNRA Training Environment Setup Script
# For Ubuntu 24.04.3 LTS with Python 3.11.x

set -e  # Exit on error

echo "=========================================="
echo "VeNRA Training Environment Setup"
echo "Ubuntu 24.04.3 LTS + Python 3.11.x"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Ubuntu 24.04
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$VERSION_ID" != "24.04" ]]; then
        echo -e "${YELLOW}Warning: Detected $NAME $VERSION_ID instead of Ubuntu 24.04${NC}"
        echo "This script is optimized for Ubuntu 24.04.3 LTS. Continue? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Check Python version
echo -e "${GREEN}[1/8] Checking Python 3.11...${NC}"
if command -v python3.11 &> /dev/null; then
    PYTHON_VERSION=$(python3.11 --version | awk '{print $2}')
    echo "✓ Found Python $PYTHON_VERSION"
else
    echo -e "${RED}✗ Python 3.11 not found${NC}"
    echo "Installing Python 3.11..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
fi

# Install system dependencies
echo -e "${GREEN}[2/8] Installing system dependencies...${NC}"
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3.11-dev \
    python3.11-venv \
    libssl-dev \
    libffi-dev \
    wget \
    curl

# Check NVIDIA driver
echo -e "${GREEN}[3/8] Checking NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "✓ NVIDIA driver detected"
else
    echo -e "${RED}✗ NVIDIA driver not found${NC}"
    echo "Please install NVIDIA drivers first:"
    echo "  sudo ubuntu-drivers autoinstall"
    echo "  sudo reboot"
    exit 1
fi

# Check CUDA
echo -e "${GREEN}[4/8] Checking CUDA...${NC}"

# Get driver CUDA version from nvidia-smi
if nvidia-smi &> /dev/null; then
    DRIVER_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "✓ NVIDIA Driver supports CUDA: $DRIVER_CUDA"
    
    # Explain backward compatibility
    if [[ "$DRIVER_CUDA" > "12.0" ]]; then
        echo "  Note: Driver CUDA $DRIVER_CUDA is backward compatible with PyTorch CUDA 12.1"
        echo "  We'll use PyTorch built for CUDA 12.1 (most stable)"
    fi
fi

# Check if CUDA toolkit is installed (optional)
if command -v nvcc &> /dev/null; then
    TOOLKIT_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo "✓ CUDA Toolkit $TOOLKIT_VERSION installed"
else
    echo "  CUDA toolkit not installed (not required - PyTorch includes runtime)"
fi

# Create virtual environment
echo -e "${GREEN}[5/8] Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}venv directory already exists. Remove? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf venv
    else
        echo "Using existing venv..."
    fi
fi

if [ ! -d "venv" ]; then
    python3.11 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
echo -e "${GREEN}[6/8] Upgrading pip, setuptools, wheel...${NC}"
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA
echo -e "${GREEN}[7/8] Installing PyTorch with CUDA...${NC}"

# Determine best CUDA version
echo "Your driver supports CUDA $DRIVER_CUDA"
echo "Attempting to install PyTorch with CUDA 13.0 (closest to your driver)..."

# Try cu130 first, fall back to cu128 if it fails
if pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 2>/dev/null; then
    echo "✓ Successfully installed PyTorch with CUDA 13.0"
else
    echo "CUDA 13.0 build not available, falling back to CUDA 12.8 (latest stable)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi

# Verify CUDA in PyTorch
python << 'EOF'
import torch
if not torch.cuda.is_available():
    print("\033[0;31m✗ CUDA not available in PyTorch!\033[0m")
    print("This is critical. Check NVIDIA driver installation.")
    exit(1)
else:
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
EOF

# Install remaining dependencies
echo -e "${GREEN}[8/8] Installing remaining dependencies...${NC}"
pip install -r requirements_training.txt

# Create .env template if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${GREEN}Creating .env template...${NC}"
    cat > .env << 'EOF'
# Hugging Face Token
# Get from: https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_token_here

# Weights & Biases API Key
# Get from: https://wandb.ai/authorize
WANDB_API_KEY=your_wandb_key_here

# CUDA Configuration (RTX 3090)
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
EOF
    echo "✓ Created .env template"
    echo -e "${YELLOW}⚠ IMPORTANT: Edit .env and add your HF_TOKEN and WANDB_API_KEY${NC}"
else
    echo "✓ .env file already exists"
fi

# Run verification
echo ""
echo "=========================================="
echo "Running Environment Verification..."
echo "=========================================="

python << 'EOF'
import sys
import torch
import transformers
import peft
import trl
import bitsandbytes as bnb
from dotenv import load_dotenv
import os

print("\n" + "="*60)
print("VeNRA Environment Verification")
print("="*60)

# Python version
py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"✓ Python: {py_version}")

# PyTorch + CUDA
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Libraries
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ PEFT: {peft.__version__}")
print(f"✓ TRL: {trl.__version__}")
print(f"✓ bitsandbytes: {bnb.__version__}")

# Environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
wandb_key = os.getenv("WANDB_API_KEY")

if hf_token and hf_token != "hf_your_token_here":
    print(f"✓ HF_TOKEN: Loaded")
else:
    print(f"⚠ HF_TOKEN: NOT SET (edit .env file)")

if wandb_key and wandb_key != "your_wandb_key_here":
    print(f"✓ WANDB_API_KEY: Loaded")
else:
    print(f"⚠ WANDB_API_KEY: NOT SET (optional, edit .env file)")

print("="*60)
print("\n✅ Setup Complete!")
print("\nNext Steps:")
print("1. Edit .env file with your HF_TOKEN and WANDB_API_KEY")
print("2. Activate environment: source venv/bin/activate")
print("3. Run training: python train_venra_v3_corrected.py")
print("="*60 + "\n")
EOF

echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "To start training:"
echo "  1. Edit .env with your tokens"
echo "  2. source venv/bin/activate"
echo "  3. python train_venra_v3_corrected.py"
echo ""