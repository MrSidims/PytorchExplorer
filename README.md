# PyTorch IR Explorer

[![Nightly](https://github.com/MrSidims/PytorchExplorer/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MrSidims/PytorchExplorer/actions?query=workflow%3ACI+event%3Aschedule)

An interactive web-based tool for exploring intermediate representations (IRs) of PyTorch and Triton models.
Designed to help developers, researchers, and students visualize and understand compilation pipelines by
tracing models through various IR stages and transformations.

## Features

- **Live editing of PyTorch, Triton models and raw IR input**
- **Pre-defined lowering IR support**:
  - TorchScript Graph IR
  - Torch MLIR (and TOSA, Linalg, StableHLO dialects)
  - LLVM MLIR and LLVM IR
  - Triton IRs (TTIR, TTGIR, LLVM IR, NVPTX)
- **Customizable compiler pipelines** with toolchain steps like:
  - `torch-mlir-opt`
  - `mlir-opt`
  - `mlir-translate`
  - `opt`, `llc`, or any external tool via `$PATH`
- **Visual pipeline builder** to control and inspect transformation flow
- **IR viewer** with syntax highlighting
- **Side-by-side IR windows**
- **"Print after all opts"** toggle to inspect intermediate outputs

## Known issues and limitations

- (PyTorch) The model and input tensor must be initialized in the provided code. If multiple models are defined, it is recommended to explicitly pair each model and its input tensor using the internal `__explore__(model, input)` function.

- (Triton) The current implementation runs Triton kernels and retrieves IR dumps from the Triton cache directory. Timeout is set to 20s.

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js + npm
- PyTorch
- Torch-MLIR
- Triton
- LLVM with mlir-opt

To setup PyTorch and Torch-MLIR it's a good idea to visit https://github.com/llvm/torch-mlir repository and follow instructions from there.

Current version of the application is tested on Ubuntu 22.04 windows subsystem using LLVM 21 dev.

### Install dependencies

Clone the repository:
```bash
git clone https://github.com/MrSidims/PytorchExplorer.git
cd PytorchExplorer
```

Install frontend dependencies:
```bash
source setup_frontend.sh
```

Set up backend (Torch, MLIR, etc.):
```bash
source setup_backend.sh
```

If you already have a working venv for Torch-MLIR, you can just install FastAPI and testing dependencies:
```bash
pip install fastapi uvicorn pytest httpx
```

To use custom builds of `torch-mlir-opt`, `mlir-opt`, etc. without placing them in your `$PATH`, configure the following environment variables:
- `TORCH_MLIR_OPT_PATH`
- `LLVM_BIN_PATH`
- `TRITON_OPT_PATH`

### Run the application

#### Development mode (local)
```bash
npm run dev:all
```
Then open http://localhost:3000/

#### Production mode (local)
```bash
npm run build
npm run start:all
```

Then open http://localhost:3000/ in your browser and enjoy!

#### Backend and frontend on different machines

Start the backend on the machine that has all compiler tools installed:

```bash
npm run start:api      # or npm run dev:api for development
```

On the machine running the UI, point the frontend to that backend via the
`NEXT_PUBLIC_BACKEND_URL` environment variable and start only the UI part:

```bash
export NEXT_PUBLIC_BACKEND_URL=http://<backend-host>:8000
npm run dev:ui         # or npm run start:ui after `npm run build`
```

#### Run in a container (Docker or Podman)

Build the single image (change `APP_ENV` between development/production, default is production):
```bash
docker build -t pytorch_explorer --build-arg APP_ENV=development .
```

Alternatively build dedicated images for the UI and API:
```bash
docker build -f Dockerfile.backend -t pytorch_explorer_backend .
docker build -f Dockerfile.frontend -t pytorch_explorer_frontend .
```

Run the container in **production mode**:
```bash
docker run -p 3000:3000 -p 8000:8000 pytorch_explorer
```
Then inside the container:
```bash
npm run build
npm run start:all
```

To run in **development mode**:
```bash
docker run -it --rm \
  -e NODE_ENV=development \
  -p 3000:3000 -p 8000:8000 \
  pytorch_explorer
```
Then inside the container:
```bash
npm run dev:all
```

To run the UI and API in separate containers using docker compose:
```bash
docker compose build
docker compose up
```

Secure run (in cases, when you don't trust tested samples):
```bash
podman run --rm -it \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --tmpfs /app/.next:rw,size=256m \
  -v stored_sessions:/app/StoredSessions:rw \
  -p8000:8000 -p3000:3000 \
  -e NODE_ENV=production \
  pytorch_explorer
```

### Run the tests

With the backend running you can execute the Python tests. Point them at the
backend via the optional `API_URL` environment variable if it isn't on
`localhost:8000`:

```bash
API_URL=http://<backend-host>:8000 pytest tests -v
```

## User manual

TBD

## Implementation details

The app uses `fx.export_and_import` under the hood to inpect IR output for PyTorch, therefore for pre-defined lowering paths it's required for a module to have `forward` method.

Lowering to LLVM IR goes through:

```bash
module = fx.export_and_import(model, example_input, output_type=OutputType.LINALG_ON_TENSORS)
mlir-opt --one-shot-bufferize="bufferize-function-boundaries"
         -convert-linalg-to-loops
         -convert-scf-to-cf
         -convert-cf-to-llvm
         -lower-affine
         -finalize-memref-to-llvm
         -convert-math-to-llvm
         -convert-math-to-llvm
         -convert-func-to-llvm
         -reconcile-unrealized-casts
         str(module) -o output.mlir
mlir-translate --mlir-to-llvmir output.mlir
```

For more details about IR lowering, please see [PyTorch Lowerings](docs/pytorch_lowering.md).

## Integration with your frontend or backend

Refer to the [Integration Guide](docs/integration_guide.md) for details on the API contracts and communication between the frontend and backend used in this project.

