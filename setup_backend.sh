#!/bin/bash
set -e

echo "Installing LLVM 22 and MLIR tools..."
#!/bin/bash
set -e

export LLVM_VERSION=22
echo "Installing LLVM $LLVM_VERSION"

curl -L "https://apt.llvm.org/llvm-snapshot.gpg.key" | sudo apt-key add -
echo "deb https://apt.llvm.org/jammy/ llvm-toolchain-jammy main" | sudo tee -a /etc/apt/sources.list

sudo apt-get update
sudo apt-get update
sudo apt-get -y install \
    llvm-22-dev \
    llvm-22-tools \
    mlir-22-tools

echo "Exporting LLVM 22 tools path..."
export PATH=/usr/lib/llvm-22/bin:$PATH

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

