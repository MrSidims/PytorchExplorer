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
- LLVM with mlir-opt
- Triton

To setup PyTorch and Torch-MLIR it's a good idea to visit https://github.com/llvm/torch-mlir repository and follow instructions from there.

Current version of the application is tested on Ubuntu 22.04 windows subsystem using LLVM 22 dev.

Triton requires that PyTorch be compiled with CUDA or ROCm support. When
installing PyTorch, pick the desired accelerator build. For example, to install
a CUDA 12.8 wheel you can run (note: this is not included in scripts and dockerfiles) (at least this works with my Blackwell GPU):

```bash
pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
```

### Install dependencies

Clone the repository:
```bash
git clone https://github.com/MrSidims/PytorchExplorer.git
cd PytorchExplorer
```

To use custom builds of `torch-mlir-opt`, `mlir-opt`, etc. without placing them in your `$PATH`, configure the following environment variables:
- `TORCH_MLIR_OPT_PATH`
- `LLVM_BIN_PATH`
- `TRITON_OPT_PATH`
- `PYTORCH_INDEX` – Index URL for installing PyTorch. Defaults to nightly CPU wheels.

For example, to install CUDA-enabled nightly wheels (CUDA 12.8):
```bash
PYTORCH_INDEX=https://download.pytorch.org/whl/nightly/cu128 \
  source setup_backend.sh
```

Install frontend dependencies:
```bash
source setup_frontend.sh
```

Set up backend (Torch, MLIR, etc.) (note, unless `PYTORCH_INDEX` is set the script will install CPU wheels):
```bash
source setup_backend.sh
```

If you already have a working venv for Torch-MLIR, you can just install FastAPI and testing dependencies:
```bash
pip install fastapi uvicorn pytest httpx
```

### Run the application

If you are reused `setup_backend.sh` script - activate the environment with

```bash
source mlir_venv/bin/activate
```

#### Development mode (local)
```bash
npm run dev:all
```

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

To run in **development mode**:
```bash
docker run -it --rm \
  -e NODE_ENV=development \
  -p 3000:3000 -p 8000:8000 \
  pytorch_explorer
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

The interface features a code editor on the left and one or more **IR windows**
on the right.

1. Choose **PyTorch**, **Triton** or **Raw IR** from the language selector above
   the editor and enter your code. Use **Add Source** to work with multiple
   snippets at once.
2. Each window on the right picks a target IR from its drop‑down. Create extra
   windows via **Add IR Window** and switch between vertical or horizontal layout
   using the layout selector.
3. Click **Add Pass** inside a window to build a custom pipeline with tools such
   as `torch-mlir-opt`, `mlir-opt`, `mlir-translate`, `opt`, `llc` or a
   user-specified tool. Toggle **Print IR after opts** to see intermediate IR and
   use the magnifying glass button to inspect a single stage.
4. Press **Generate IR on All Windows** to compile the active source and fill
   each window with the resulting IR. Windows can be collapsed or closed
   individually.
5. Hit **Store Session** to save your work. The backend returns a short ID which
   can be appended to the URL (e.g. `/abc123`) to reload the same session later.

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