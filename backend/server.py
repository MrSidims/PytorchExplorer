import subprocess
import tempfile
import os
import glob
import uuid
import hashlib
import shutil
import atexit
import logging
from typing import List, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import torch.nn as nn
from torch_mlir import fx
from torch_mlir.fx import OutputType

from .errors import (
    IRGenerationError,
    PytorchExecutionError,
    TritonCompilationError,
    TritonExecutionError,
    CompilerPipelineError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cached_triton_runs = {}

TORCH_MLIR_OPT_PATH = os.environ.get("TORCH_MLIR_OPT_PATH", "")
LLVM_BIN_PATH = os.environ.get("LLVM_BIN_PATH", "")
TRITON_OPT_PATH = os.environ.get("TRITON_OPT_PATH", "")


class CodeRequest(BaseModel):
    code: str
    ir_type: str
    selected_language: Optional[str] = "pytorch"
    custom_pipeline: Optional[List[str]] = []
    torch_mlir_opt: Optional[str] = ""
    mlir_opt: Optional[str] = ""
    mlir_translate: Optional[str] = ""
    llvm_opt: Optional[str] = ""
    llc: Optional[str] = ""
    triton_opt: Optional[str] = ""
    triton_llvm_opt: Optional[str] = ""
    user_tool: Optional[str] = ""
    dump_after_each_opt: Optional[bool] = False


class FreeIRCacheRequest(BaseModel):
    code: str


# Get model-tensor pairs to process.
def extract_model_input_pairs(code: str):
    explore_pairs = []

    def __explore__(model, input_tensor):
        explore_pairs.append((model, input_tensor))

    exec_globals = {"__explore__": __explore__}

    # Suppress stdout and stderr.
    with open(os.devnull, "w") as devnull:
        try:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                exec(code, exec_globals)
        except Exception as e:
            logger.exception("User code execution failed in extract_model_input_pairs")
            raise PytorchExecutionError("Code raised an exception during exploration.")

    # If no __explore__ calls found, try matching models and tensors heuristically.
    if not explore_pairs:
        models = [v for v in exec_globals.values() if isinstance(v, nn.Module)]
        tensors = [v for v in exec_globals.values() if isinstance(v, torch.Tensor)]

        for model in models:
            for tensor in tensors:
                try:
                    model.eval()
                    with torch.no_grad():
                        model(tensor)
                    explore_pairs.append((model, tensor))
                    break
                except Exception:
                    continue  # Bad model, tensor combos can happen, continue silently

    return explore_pairs


# TODO: reuse it for pytorch?
def hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def fix_input(input_obj):
    input_obj = (input_obj,) if not isinstance(input_obj, tuple) else input_obj
    if len(input_obj) == 1:
        return input_obj[0]
    return input_obj


# Run torch-mlir-opt and/or mlir-opt and/or opt etc.
def run_external_opt_tool_file(
    input_path: str, cmd: str, tool: str, output_path: str
) -> Tuple[bool, str]:
    try:
        args = [tool] + cmd.split() + [input_path, "-o", output_path]
        result = subprocess.run(args, capture_output=True, text=True)
        return (result.returncode == 0, result.stderr if result.stderr else "")
    except Exception as e:
        logger.error(f"Failed to run tool '{tool}': {e}")
        raise CompilerPipelineError(f"Failed to run compiler tool '{tool}'")


# Utility for custom pipeline.
def apply_optional_passes(
    ir: str, pipeline: List[Tuple[str, str]], dump_each: bool = False
) -> str:
    uid = uuid.uuid4().hex
    output = ""
    temp_files = []

    # Step 1: Write initial IR to a file.
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(ir)
        f.flush()
        prev_path = f.name
        temp_files.append(prev_path)

    if dump_each:
        output += f"\n\n===== Initial IR =====\n{ir}"

    # Step 2: Apply pipeline stages.
    for index, (tool, flags) in enumerate(pipeline):
        tool_path = None

        if tool == "torch-mlir-opt":
            tool_path = TORCH_MLIR_OPT_PATH + "torch-mlir-opt"
        elif tool == "mlir-opt":
            tool_path = LLVM_BIN_PATH + "mlir-opt"
        elif tool == "mlir-translate":
            tool_path = LLVM_BIN_PATH + "mlir-translate"
        elif tool == "opt":
            flags += " -S"
            tool_path = LLVM_BIN_PATH + "opt"
        elif tool == "llc":
            tool_path = LLVM_BIN_PATH + "llc"
        elif tool == "triton-opt":
            tool_path = TRITON_OPT_PATH + "triton-opt"
        elif tool == "triton-llvm-opt":
            tool_path = TRITON_OPT_PATH + "triton-llvm-opt"
        elif tool == "user-tool":
            tokens = flags.strip().split()
            if not tokens:
                raise CompilerPipelineError("Empty user-tool invocation")
            tool_path = tokens[0]
            flags = " ".join(tokens[1:])
        else:
            raise CompilerPipelineError(f"Unknown pipeline tool: '{tool}'")

        out_path = os.path.join(tempfile.gettempdir(), f"ir_step_{index}_{uid}")
        temp_files.append(out_path)

        success, stderr = run_external_opt_tool_file(
            prev_path, flags, tool_path, out_path
        )
        if not success:
            raise CompilerPipelineError(f"{tool} failed: {stderr}")

        if dump_each:
            with open(out_path, "r") as f:
                stage_output = f.read()
            output += f"\n\n===== IR after {tool} {flags} =====\n{stage_output}"

        prev_path = out_path

    if not dump_each:
        with open(prev_path, "r") as f:
            output = f.read()

    # Cleanup.
    for path in temp_files:
        try:
            os.remove(path)
        except Exception:
            pass

    return output


# Torch graph IR.
def generate_torch_script_graph_ir(model, example_input, pipeline, dump_each):
    try:
        traced_model = torch.jit.trace(model, example_input)
        return apply_optional_passes(str(traced_model.graph), pipeline, dump_each)
    except Exception as e:
        logger.exception("Failed to generate TorchScript Graph IR.")
        raise IRGenerationError("Failed to generate TorchScript Graph IR.")


# Torch MLIR dialect.
def generate_torch_mlir(
    model, example_input, pipeline: List[Tuple[str, str]], dump_each: bool
) -> str:
    try:
        example_input = fix_input(example_input)
        module = fx.export_and_import(
            model, example_input, output_type=OutputType.TORCH
        )
        return apply_optional_passes(str(module), pipeline, dump_each)
    except Exception as e:
        logger.exception("Error generating Torch MLIR.")
        raise IRGenerationError("Failed to generate Torch MLIR.")


# TOSA MLIR dialect, uses FX backend.
def generate_tosa_mlir(
    model, example_input, pipeline: List[Tuple[str, str]], dump_each: bool
) -> str:
    try:
        example_input = fix_input(example_input)
        module = fx.export_and_import(model, example_input, output_type=OutputType.TOSA)
        return apply_optional_passes(str(module), pipeline, dump_each)
    except Exception as e:
        logger.exception("Error generating TOSA MLIR.")
        raise IRGenerationError("Failed to generate TOSA MLIR.")


# Linalg on tensors, uses FX backend.
def generate_linalg_on_tensors_mlir(
    model, example_input, pipeline: List[Tuple[str, str]], dump_each: bool
) -> str:
    try:
        example_input = fix_input(example_input)
        module = fx.export_and_import(
            model, example_input, output_type=OutputType.LINALG_ON_TENSORS
        )
        return apply_optional_passes(str(module), pipeline, dump_each)
    except Exception as e:
        logger.exception("Error generating Linalg on Tensors MLIR.")
        raise IRGenerationError("Failed to generate Linalg on Tensors MLIR.")


# StableHLO, uses FX backend.
def generate_stablehlo_mlir(
    model, example_input, pipeline: List[Tuple[str, str]], dump_each: bool
) -> str:
    try:
        example_input = fix_input(example_input)
        module = fx.export_and_import(
            model, example_input, output_type=OutputType.STABLEHLO
        )
        return apply_optional_passes(str(module), pipeline, dump_each)
    except Exception as e:
        logger.exception("Error generating StableHLO MLIR.")
        raise IRGenerationError("Failed to generate StableHLO MLIR.")


# First generate linalg on tensors, then run conversion to LLVM MLIR pipeline.
def lower_to_llvm_mlir(model, example_input) -> str:
    example_input = fix_input(example_input)
    module = fx.export_and_import(
        model, example_input, output_type=OutputType.LINALG_ON_TENSORS
    )

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".mlir", delete=False) as f:
        f.write(str(module))
        f.flush()
        input_path = f.name

    result = subprocess.run(
        [
            LLVM_BIN_PATH + "mlir-opt",
            '--one-shot-bufferize="bufferize-function-boundaries"',
            "-convert-linalg-to-loops",
            "-convert-scf-to-cf",
            "-convert-cf-to-llvm",
            "-lower-affine",
            "-finalize-memref-to-llvm",
            "-convert-math-to-llvm",
            "-convert-arith-to-llvm",
            "-convert-func-to-llvm",
            "-reconcile-unrealized-casts",
            input_path,
        ],
        capture_output=True,
        text=True,
    )

    os.remove(input_path)

    if result.returncode != 0:
        raise CompilerPipelineError(f"mlir-opt failed: {result.stderr}")

    return result.stdout


# Generate LLVM MLIR.
def generate_llvm_mlir(
    model, example_input, pipeline: List[Tuple[str, str]], dump_each: bool
) -> str:
    try:
        base_ir = lower_to_llvm_mlir(model, example_input)
        return apply_optional_passes(base_ir, pipeline, dump_each)
    except Exception as e:
        logger.exception("Error generating LLVM MLIR.")
        raise IRGenerationError("Failed to generate LLVM MLIR.")


# First generate LLVM MLIR and then translate it to LLVM IR.
def generate_llvm_ir(
    model, example_input, pipeline: List[Tuple[str, str]], dump_each: bool
):
    try:
        lowered_mlir = lower_to_llvm_mlir(model, example_input)

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".mlir", delete=False) as f:
            f.write(lowered_mlir)
            f.flush()
            input_path = f.name

        result = subprocess.run(
            [LLVM_BIN_PATH + "mlir-translate", "--mlir-to-llvmir", input_path],
            capture_output=True,
            text=True,
        )

        os.remove(input_path)

        if result.returncode != 0:
            raise CompilerPipelineError(f"mlir-translate failed: {result.stderr}")

        llvm_ir = result.stdout
        return apply_optional_passes(llvm_ir, pipeline, dump_each)
    except Exception as e:
        logger.exception("Error generating LLVM IR.")
        raise IRGenerationError("Failed to generate LLVM IR.")


# TODO: Figure out static compilation.
def compile_triton_ir(
    code: str, ir_type: str, pipeline: List[Tuple[str, str]], dump_each: bool
) -> str:
    try:
        code_hash = hash_code(code)
        cache_info = cached_triton_runs.get(code_hash)

        if cache_info:
            cache_dir = cache_info["cache_dir"]
        else:
            cache_dir = os.path.join("/tmp", f"triton_cache_{uuid.uuid4().hex}")
            os.makedirs(cache_dir, exist_ok=True)

            env = os.environ.copy()
            env["TRITON_CACHE_DIR"] = cache_dir
            env["TRITON_DUMP_IR"] = "1"
            env["TRITON_DISABLE_COMPILE_TIME_ASSERTS"] = "1"

            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".py", delete=False
            ) as tmp_file:
                tmp_file.write(code)
                tmp_file.flush()
                tmp_path = tmp_file.name

            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                env=env,
                timeout=20,
            )

            os.remove(tmp_path)

            if result.returncode != 0:
                shutil.rmtree(cache_dir, ignore_errors=True)
                logger.exception("User code execution failed.")
                raise TritonExecutionError("Triton code execution raised an exception.")

            cached_triton_runs[code_hash] = {"cache_dir": cache_dir, "active_users": 0}

        pattern_map = {
            "triton_ir": "*.ttir",
            "triton_gpu_ir": "*.ttgir",
            "triton_llvm_ir": "*.llir",
            "triton_nvptx": "*.ptx",
        }

        pattern = pattern_map.get(ir_type)
        if not pattern:
            raise IRGenerationError(f"Unsupported Triton IR type: '{ir_type}'")

        files = glob.glob(os.path.join(cache_dir, "**", pattern), recursive=True)

        if not files:
            raise TritonExecutionError("Triton code execution failed.")

        output_parts = []
        for file_path in sorted(files):
            with open(file_path, "r", errors="ignore") as f:
                output_parts.append(f.read())

        cached_triton_runs[code_hash]["active_users"] += 1

        ir_dump = "\n\n".join(output_parts)
        return apply_optional_passes(ir_dump, pipeline, dump_each)

    except Exception as e:
        logger.exception("Error generating Triton IR.")
        raise TritonCompilationError("Failed to compile Triton IR.")


# Helper for custom pipeline.
def build_pipeline(request: CodeRequest) -> List[Tuple[str, str]]:
    pipeline = []
    if request.torch_mlir_opt:
        for stage in request.torch_mlir_opt.split("&&"):
            pipeline.append(("torch-mlir-opt", stage.strip()))
    if request.mlir_opt:
        for stage in request.mlir_opt.split("&&"):
            pipeline.append(("mlir-opt", stage.strip()))
    if request.mlir_translate:
        for stage in request.mlir_translate.split("&&"):
            pipeline.append(("mlir-translate", stage.strip()))
    if request.llvm_opt:
        for stage in request.llvm_opt.split("&&"):
            pipeline.append(("opt", stage.strip()))
    if request.llc:
        for stage in request.llc.split("&&"):
            pipeline.append(("llc", stage.strip()))
    if request.llc:
        for stage in request.triton_opt.split("&&"):
            pipeline.append(("triton_opt", stage.strip()))
    if request.llc:
        for stage in request.triton_llvm_opt.split("&&"):
            pipeline.append(("triton_llvm_opt", stage.strip()))
    if request.user_tool:
        for stage in request.user_tool.split("&&"):
            pipeline.append(("user-tool", stage.strip()))
    return pipeline


# Dispatcher.
def process_model(request: CodeRequest) -> str:
    try:
        if request.ir_type.startswith("triton"):
            pipeline = build_pipeline(request)
            return compile_triton_ir(
                request.code, request.ir_type, pipeline, request.dump_after_each_opt
            )

        if request.ir_type == "raw_ir" and request.selected_language == "pytorch":
            # Execute user Python, capture stdout.
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    stdout_path = os.path.join(tmpdir, "captured_output.txt")
                    with open(stdout_path, "w") as f, redirect_stdout(
                        f
                    ), redirect_stderr(f):
                        exec_globals = {}
                        exec(request.code, exec_globals)

                    with open(stdout_path, "r") as f:
                        captured = f.read()

                return apply_optional_passes(
                    captured, build_pipeline(request), request.dump_after_each_opt
                )
            except Exception as e:
                logger.exception("User code with manual IR print execution failed.")
                raise PytorchExecutionError(
                    "Code raised an exception during execution."
                )

        if request.ir_type == "raw_ir":
            return apply_optional_passes(
                request.code,
                build_pipeline(request),
                request.dump_after_each_opt,
            )

        model_input_pairs = extract_model_input_pairs(request.code)

        if not model_input_pairs:
            raise IRGenerationError("No model–tensor pairs could be extracted.")

        combined_output = ""
        pipeline = build_pipeline(request)

        for model, example_input in model_input_pairs:
            model.eval()

            if request.ir_type == "torch_script_graph_ir":
                combined_output += generate_torch_script_graph_ir(
                    model, example_input, pipeline, request.dump_after_each_opt
                )
            elif request.ir_type == "torch_mlir":
                combined_output += generate_torch_mlir(
                    model, example_input, pipeline, request.dump_after_each_opt
                )
            elif request.ir_type == "tosa_mlir":
                combined_output += generate_tosa_mlir(
                    model, example_input, pipeline, request.dump_after_each_opt
                )
            elif request.ir_type == "linalg_on_tensors_mlir":
                combined_output += generate_linalg_on_tensors_mlir(
                    model, example_input, pipeline, request.dump_after_each_opt
                )
            elif request.ir_type == "stablehlo_mlir":
                combined_output += generate_stablehlo_mlir(
                    model, example_input, pipeline, request.dump_after_each_opt
                )
            elif request.ir_type == "llvm_mlir":
                combined_output += generate_llvm_mlir(
                    model, example_input, pipeline, request.dump_after_each_opt
                )
            elif request.ir_type == "llvm_ir":
                combined_output += generate_llvm_ir(
                    model, example_input, pipeline, request.dump_after_each_opt
                )
            else:
                combined_output += "IR type not supported yet."

        return combined_output.strip()

    except Exception as e:
        logger.exception("Unhandled error in process_model")
        raise IRGenerationError("Unhandled exception while processing request.")


@app.post("/free_ir_cache")
def free_ir_cache(request: FreeIRCacheRequest):
    code_hash = hash_code(request.code)
    cache_info = cached_triton_runs.get(code_hash)

    if not cache_info:
        return {"status": "ok"}

    cache_info["active_users"] -= 1

    if cache_info["active_users"] <= 0:
        shutil.rmtree(cache_info["cache_dir"], ignore_errors=True)
        del cached_triton_runs[code_hash]

    return {"status": "ok"}


@atexit.register
def cleanup_triton_caches():
    for info in cached_triton_runs.values():
        shutil.rmtree(info["cache_dir"], ignore_errors=True)


@app.post("/generate_ir")
def generate_ir(request: CodeRequest):
    try:
        output = process_model(request)
        return {"status": "ok", "output": output}
    except IRGenerationError as e:
        logger.warning(f"IR generation failed: {e}")
        return {"status": "error", "message": str(e), "detail": None}
    except Exception as e:
        logger.exception("Unexpected internal error")
        return {"status": "error", "message": "Internal backend error", "detail": None}
