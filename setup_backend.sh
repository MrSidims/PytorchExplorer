#!/bin/bash
set -e

echo "Installing LLVM 21 and MLIR tools..."
wget -qO- https://apt.llvm.org/llvm.sh | sudo bash -s -- 21
sudo apt-get install -y libmlir-21-dev mlir-21-tools

echo "Exporting LLVM 21 tools path..."
export PATH=/usr/lib/llvm-21/bin:$PATH

echo "Installing Python tools..."
sudo apt install -y python3-pip python3.11-venv

echo "Creating virtual environment..."
python3.11 -m venv mlir_venv
source mlir_venv/bin/activate

echo "Installing torch-mlir and dependencies..."
pip install --upgrade pip
pip install --pre torch-mlir torchvision \
	  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
	    -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels

echo "Installing FastAPI and Uvicorn..."
pip install fastapi uvicorn

echo "Installing pytest and httpx..."
pip install pytest httpx

echo "Backend setup complete."

