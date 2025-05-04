# PyTorch IR Explorer

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

- (PyTorch) The model and input tensor must be initialized in the provided code. If multiple models are defined, it is recommended to explicitly pair each model and its input tensor using the internal __explore__(model, input) function.

- (PyTorch) The current version does not recognize or capture user attempts to dump IR inside the input PyTorch module. It is planned that, in the future, if the user manually calls fx.export_and_import() (or similar IR-producing APIs), the app will use that IR as the base and apply the user-defined custom toolchain.

- (Triton) The current implementation runs Triton kernels and retrieves IR dumps from the Triton cache directory. Timeout is set to 20s.

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js + npm
- PyTorch
- Torch-MLIR
- Triton
- LLVM with mlir-opt

Current version is tested on Ubuntu 22.04 windows subsystem using LLVM 21 dev.

### Install dependencies

```bash
git clone https://github.com/MrSidims/PytorchExplorer.git
cd PytorchExplorer
bash setup_frontend.sh
bash setup_backend.sh
```

If you want to use your builds of the tools like `torch-mlir-opt`, `mlir-opt` etc without placing them in `PATH` please setup `TORCH_MLIR_OPT_PATH` and `LLVM_BIN_PATH` environment variables.

### Run the application

```bash
npm run start:all
```

### Run the tests

With the application (or just backend) started, run:

```bash
pytest tests -v
```

## Implementation details

The app uses fx.export_and_import to inpect IR output for PyTorch.
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


