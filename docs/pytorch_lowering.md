# Supported Lowerings for PyTorch in PyTorch Explorer

This document describes the various IR lowerings currently supported by the project's backend for PyTorch models. Each lowering represents a different compilation target or dialect, potentially with further downstream passes or translation pipelines.

---

## 1. **TorchScript Graph IR**

- **IR Type**: `torch_script_graph_ir`
- **Source**: Generated via `torch.jit.trace`
- **Characteristics**:
  - Produces the TorchScript graph as a string representation.
  - Does not use MLIR.
  - Useful for inspecting traced PyTorch IR directly.

---

## 2. **Torch MLIR**

- **IR Type**: `torch_mlir`
- **Source**: `torch_mlir.fx.export_and_import` with `OutputType.TORCH`
- **Characteristics**:
  - Emits MLIR in the Torch dialect.
  - Used as the starting point for further dialect conversions.

---

## 3. **TOSA MLIR**

- **IR Type**: `tosa_mlir`
- **Source**: `torch_mlir.fx.export_and_import` with `OutputType.TOSA`
- **Characteristics**:
  - Emits MLIR using the TOSA (Tensor Operator Set Architecture) dialect.
  - Suitable for hardware backends that understand TOSA.

---

## 4. **Linalg on Tensors MLIR**

- **IR Type**: `linalg_on_tensors_mlir`
- **Source**: `torch_mlir.fx.export_and_import` with `OutputType.LINALG_ON_TENSORS`
- **Characteristics**:
  - Lowers PyTorch to Linalg dialect over tensors.
  - Common precursor for bufferization and lowering to LLVM.

---

## 5. **StableHLO MLIR**

- **IR Type**: `stablehlo_mlir`
- **Source**: `torch_mlir.fx.export_and_import` with `OutputType.STABLEHLO`
- **Characteristics**:
  - Emits IR using the StableHLO dialect.
  - Intended for portability across accelerators.

---

## 6. **LLVM MLIR**

- **IR Type**: `llvm_mlir`
- **Source**:
  - First generate Linalg-on-Tensors MLIR via `torch_mlir.fx.export_and_import`
  - Then apply a fixed MLIR lowering pipeline including:
    - Bufferization
    - Convert Linalg/SCF/CF to LLVM
    - Lower affine, finalize memref
    - Convert arithmetic, math, func to LLVM
    - Reconcile casts
- **Characteristics**:
  - Produces LLVM dialect MLIR suitable for final lowering.

---

## 7. **LLVM IR**

- **IR Type**: `llvm_ir`
- **Source**:
  - Generate LLVM MLIR as above.
  - Translate to LLVM IR via `mlir-translate --mlir-to-llvmir`
- **Characteristics**:
  - Produces textual `.ll` IR.
  - Suitable for feeding into `opt`, `llc`, or GPU backends.

---

## 8. **GPU Target IRs**

These are produced by first generating LLVM IR and then applying a specific backend lowering.

### a. NVPTX
- **IR Type**: `nvptx`
- **Pipeline**:
  - LLVM IR -> `opt -O2` -> `llc -mtriple=nvptx64-nvidia-cuda`

### b. AMDGPU
- **IR Type**: `amdgpu`
- **Pipeline**:
  - LLVM IR -> `opt -O2` -> `llc -mtriple=amdgcn-amd-amdhsa`

### c. SPIR-V
- **IR Type**: `spirv`
- **Pipeline**:
  - LLVM IR -> `opt -O2` -> `llc -mtriple=spirv64-unknown-unknown`

---

## 9. **Raw IR (PyTorch / MLIR)**

- **IR Type**: `raw_ir`
- **Special Case**:
  - If `selected_language` is `pytorch`, user code is executed, and its `stdout` is captured as IR.
  - Otherwise, it's treated as plain MLIR input.
- **Use Case**:
  - Manual input of custom IR for testing custom pipelines.

---

## 10. **Custom Compilation Pipelines**

All of the above IR types (except `torch_script_graph_ir`) support optional custom compilation pipelines composed of tools:

- `torch-mlir-opt`
- `mlir-opt`
- `mlir-translate`
- `opt`
- `llc`
- `triton-opt`, `triton-llvm-opt`
- User-defined tools (`user-tool`)

Each stage can optionally be configured to **dump intermediate IR**, making the lowering process fully transparent.

---

For Triton models, see the separate section on `compile_triton_ir`, which handles IR extraction from the Triton JIT cache. Supported
`ir_type` values include:

- `triton_ir` – Triton compiler dump (`*.ttir`)
- `triton_gpu_ir` – `*.ttgir`
- `triton_llvm_ir` – `*.llir`
- `triton_nvptx` – `*.ptx`
- `triton_amdgpu` – `*.hsaco`
