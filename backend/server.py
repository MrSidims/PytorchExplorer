import subprocess
import tempfile
import os
import linecache
import glob
import uuid
import hashlib
import shutil
import atexit
import logging
import traceback
import base64
from typing import List, Optional, Tuple
import re
from pathlib import Path

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

try:
    from PyPDF2 import PdfMerger
except Exception:
    PdfMerger = None

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

# Where to store per-request temporary artifacts (DOT/PDF/IR)
# Default: <project_root>/session_temps, override with PE_SESSION_TEMPS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_TEMPS_ROOT = os.environ.get(
    "PE_SESSION_TEMPS", str(PROJECT_ROOT / "session_temps")
)
os.makedirs(SESSION_TEMPS_ROOT, exist_ok=True)

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
            raise PytorchExecutionError(
                f"Code raised an exception during exploration: {e}"
            ) from e

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


def split_cmd_arguments(cmd: str) -> List[str]:
    # Split the command string into arguments, handling quoted strings.
    cmd_split = re.split(r""" (?=(?:[^'"]|'[^']*'|"[^"]*")*$)""", cmd.strip())
    # Remove quotes from each argument.
    cmd_split = [arg.replace('"', "").replace("'", "") for arg in cmd_split]
    return cmd_split


# Run torch-mlir-opt and/or mlir-opt and/or opt etc.
def run_external_opt_tool_file(
    input_path: str, cmd: str, tool: str, output_path: str, cwd: Optional[str] = None
) -> Tuple[bool, str]:
    args = [tool] + split_cmd_arguments(cmd) + [input_path, "-o", output_path]
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, check=True, cwd=cwd
        )
        return (True, result.stderr or "")
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Tool '{tool}' failed with return code {e.returncode}:\n{e.stderr}"
        )
        return (False, e.stderr or f"{tool} failed unexpectedly.")
    except FileNotFoundError as e:
        logger.error(f"Tool not found: {tool}", exc_info=True)
        raise CompilerPipelineError(f"Compiler tool '{tool}' not found.")
    except Exception as e:
        logger.error(f"Unexpected error running tool '{tool}': {e}", exc_info=True)
        raise CompilerPipelineError(f"Unexpected error while running '{tool}': {e}")


def _read_file_safe(path: str) -> Tuple[str, bool]:
    # Read a file returning its text or base64 if binary.
    # Returns a tuple of (content, is_binary). If the file cannot be decoded as
    # UTF-8, it is assumed to be binary and returned base64-encoded.

    with open(path, "rb") as f:
        data = f.read()
    try:
        return data.decode("utf-8"), False
    except UnicodeDecodeError:
        encoded = base64.b64encode(data).decode("utf-8")
        return encoded, True


# Utility for custom pipeline.
def apply_optional_passes(
    ir: str, pipeline: List[Tuple[str, str]], dump_each: bool = False
) -> str:
    uid = uuid.uuid4().hex
    output = ""
    pdf_blocks: List[Tuple[str, str]] = []

    # Make one session dir under PytorchExplorer/session_temps.
    session_dir = tempfile.mkdtemp(dir=SESSION_TEMPS_ROOT, prefix=f"sess_{uid}_")
    try:
        # Step 1: Write initial IR into the session dir.
        prev_path = os.path.join(session_dir, f"ir_init_{uid}.ll")
        with open(prev_path, "w") as f:
            f.write(ir)

        if dump_each:
            output += f"\n\n===== Initial IR =====\n{ir}"

        # Step 2: Apply pipeline stages (all I/O + cwd inside session_dir).
        for index, (tool, flags) in enumerate(pipeline):
            if tool == "torch-mlir-opt":
                tool_path = os.path.join(TORCH_MLIR_OPT_PATH, "torch-mlir-opt")
            elif tool == "mlir-opt":
                tool_path = os.path.join(LLVM_BIN_PATH, "mlir-opt")
            elif tool == "mlir-translate":
                tool_path = os.path.join(LLVM_BIN_PATH, "mlir-translate")
            elif tool == "opt":
                flags += " -S"
                tool_path = os.path.join(LLVM_BIN_PATH, "opt")
            elif tool == "llc":
                tool_path = os.path.join(LLVM_BIN_PATH, "llc")
            elif tool == "triton-opt":
                tool_path = os.path.join(TRITON_OPT_PATH, "triton-opt")
            elif tool == "triton-llvm-opt":
                tool_path = os.path.join(TRITON_OPT_PATH, "triton-llvm-opt")
            elif tool == "user-tool":
                tokens = split_cmd_arguments(flags)
                if not tokens:
                    raise CompilerPipelineError("Empty user-tool invocation")
                tool_path = tokens[0]
                flags = " ".join(tokens[1:])
                if os.path.basename(tool_path) == "opt" and "-S" not in flags.split():
                    flags += " -S"
            else:
                raise CompilerPipelineError(f"Unknown pipeline tool: '{tool}'")

            out_path = os.path.join(session_dir, f"ir_step_{index}_{uid}.ll")

            if "dot-cfg" in flags and "--dot-cfg-dir=" not in flags:
                flags += f" --dot-cfg-dir={session_dir}"

            # Run the tool with cwd=session_dir so DOTs land here.
            success, stderr = run_external_opt_tool_file(
                prev_path, flags, tool_path, out_path, cwd=session_dir
            )
            if not success:
                raise CompilerPipelineError(f"{tool} failed: {stderr}")

            # Collect DOTs from session_dir -> convert to PDFs -> merge -> attach.
            dot_files = sorted(
                set(
                    glob.glob(os.path.join(session_dir, "*.dot"))
                    + glob.glob(os.path.join(session_dir, ".*.dot"))
                )
            )
            pdf_paths: List[str] = []

            # Only warn if user requested dot-cfg but no DOTs appeared.
            if "dot-cfg" in flags and not dot_files:
                logger.warning(
                    "No *.dot emitted by -passes=dot-cfg; checked %s", session_dir
                )

            # Convert DOT -> PDF.
            if dot_files:
                if not shutil.which("dot"):
                    logger.error(
                        "'dot' (graphviz) not found on PATH; cannot render CFG PDFs."
                    )
                else:
                    for df in sorted(set(dot_files)):
                        pdf_path = os.path.splitext(df)[0] + ".pdf"
                        try:
                            subprocess.run(
                                ["dot", "-Tpdf", df, "-o", pdf_path], check=True
                            )
                            pdf_paths.append(pdf_path)
                        except Exception as e:
                            logger.error(f"Failed to convert {df} to PDF: {e}")

                # Remove DOTs after conversion (keep PDFs).
                for df in dot_files:
                    try:
                        os.remove(df)
                    except Exception:
                        pass

            def _encode_and_attach(path_to_pdf: str):
                try:
                    with open(path_to_pdf, "rb") as pf:
                        encoded = base64.b64encode(pf.read()).decode("utf-8")
                    pdf_blocks.append((os.path.basename(path_to_pdf), encoded))
                    if dump_each:
                        nonlocal output
                        output += f"\n\n===== DOT PDF {os.path.basename(path_to_pdf)} =====\n{encoded}"
                except Exception as e:
                    logger.error(f"Failed to read PDF {path_to_pdf}: {e}")

            # Merge PDFs if we have more than one.
            if pdf_paths:
                merged_ok = False
                merged_path = os.path.join(session_dir, f"cfg-merged-stage-{index}.pdf")

                if PdfMerger is not None:
                    try:
                        merger = PdfMerger()
                        for p in sorted(pdf_paths):
                            merger.append(p)
                        with open(merged_path, "wb") as mf:
                            merger.write(mf)
                        merger.close()
                        _encode_and_attach(merged_path)
                        merged_ok = True
                    except Exception as e:
                        logger.error(f"PDF merge via PyPDF2 failed: {e}")

                if not merged_ok and shutil.which("pdfunite"):
                    try:
                        cmd = ["pdfunite"] + sorted(pdf_paths) + [merged_path]
                        subprocess.run(cmd, check=True)
                        _encode_and_attach(merged_path)
                        merged_ok = True
                    except Exception as e:
                        logger.error(f"PDF merge via pdfunite failed: {e}")

                if not merged_ok:
                    # Fall back to attaching individual PDFs.
                    for p in sorted(pdf_paths):
                        _encode_and_attach(p)

            # Handle analysis-only stages (no output file produced).
            wrote_output = os.path.exists(out_path)

            if dump_each:
                path_to_show = out_path if wrote_output else prev_path
                stage_output, is_binary = _read_file_safe(path_to_show)
                if not wrote_output:
                    output += f"\n\n===== IR after {tool} {flags} (no new output; IR unchanged) =====\n"
                else:
                    output += f"\n\n===== IR after {tool} {flags} =====\n"
                output += stage_output

            if wrote_output:
                prev_path = out_path

        # Final assembly.
        if not dump_each:
            with open(prev_path, "rb") as f:
                data = f.read()
            try:
                output = data.decode("utf-8")
            except UnicodeDecodeError:
                encoded = base64.b64encode(data).decode("utf-8")
                if data.startswith(b"%PDF"):
                    pdf_blocks.insert(
                        0, (os.path.basename(prev_path) + ".pdf", encoded)
                    )
                    output = ""
                else:
                    output = f"===== BINARY OUTPUT {os.path.basename(prev_path)} =====\n{encoded}"

            for name, encoded in pdf_blocks:
                if output:
                    output += "\n\n"
                output += f"===== DOT PDF {name} =====\n{encoded}"

        return output

    finally:
        # Nuke the whole session dir; PDFs/DOTs/IR intermediates are all ephemeral.
        shutil.rmtree(session_dir, ignore_errors=True)


# Torch graph IR.
def generate_torch_script_graph_ir(model, example_input, pipeline, dump_each):
    try:
        traced_model = torch.jit.trace(model, example_input)
        return apply_optional_passes(str(traced_model.graph), pipeline, dump_each)
    except Exception as e:
        logger.exception("Failed to generate TorchScript Graph IR.")
        raise IRGenerationError(f"Failed to generate TorchScript Graph IR: {e}") from e


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
        raise IRGenerationError(f"Failed to generate Torch MLIR: {e}") from e


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
        raise IRGenerationError(f"Failed to generate TOSA MLIR: {e}") from e


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
        raise IRGenerationError(
            f"Failed to generate Linalg on Tensors MLIR: {e}"
        ) from e


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
        raise IRGenerationError(f"Failed to generate StableHLO MLIR: {e}") from e


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

    cmd = [
        os.path.join(LLVM_BIN_PATH, "mlir-opt"),
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
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise CompilerPipelineError(
            f"mlir-opt failed with code {e.returncode}: {e.stderr}"
        )
    except FileNotFoundError:
        raise CompilerPipelineError(f"'mlir-opt' not found at path: {cmd[0]}")
    finally:
        # Prevent tmp leaks
        try:
            os.remove(input_path)
        except Exception:
            pass


# Generate LLVM MLIR.
def generate_llvm_mlir(
    model, example_input, pipeline: List[Tuple[str, str]], dump_each: bool
) -> str:
    try:
        base_ir = lower_to_llvm_mlir(model, example_input)
        return apply_optional_passes(base_ir, pipeline, dump_each)
    except Exception as e:
        logger.exception("Error generating LLVM MLIR.")
        raise IRGenerationError(f"Failed to generate LLVM MLIR: {e}") from e


# First generate LLVM MLIR and then translate it to LLVM IR.
def generate_llvm_ir(
    model, example_input, pipeline: List[Tuple[str, str]], dump_each: bool
) -> str:
    try:
        lowered_mlir = lower_to_llvm_mlir(model, example_input)

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".mlir", delete=False) as f:
            f.write(lowered_mlir)
            f.flush()
            input_path = f.name

        result = subprocess.run(
            [
                os.path.join(LLVM_BIN_PATH, "mlir-translate"),
                "--mlir-to-llvmir",
                input_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        llvm_ir = result.stdout
        return apply_optional_passes(llvm_ir, pipeline, dump_each)

    except subprocess.CalledProcessError as e:
        logger.error(f"mlir-translate failed: {e.stderr}", exc_info=True)
        raise CompilerPipelineError(f"mlir-translate failed: {e.stderr}")

    except Exception as e:
        logger.exception("Error generating LLVM IR.")
        raise IRGenerationError(f"Failed to generate LLVM IR: {e}") from e

    finally:
        # Prevent tmp leaks
        try:
            os.remove(input_path)
        except Exception:
            pass


# Generate NVPTX, AMDGPU or SPIR-V.
def generate_target_gpu_ir(model, example_input, target: str) -> str:
    try:
        llvm_ir_module = generate_llvm_ir(model, example_input, [], False)
        pipeline: list[tuple[str, str]] = [("opt", "-O2")]
        if target == "nvptx":
            pipeline.append(("llc", "-mtriple=nvptx64-nvidia-cuda"))
        elif target == "amdgpu":
            pipeline.append(("llc", "-mtriple amdgcn-amd-amdhsa"))
        elif target == "spirv":
            pipeline.append(("llc", "-mtriple=spirv64-unknown-unknown"))
        return apply_optional_passes(llvm_ir_module, pipeline, False)
    except Exception as e:
        logger.exception("Error generating LLVM IR.")
        raise IRGenerationError(f"Failed to generate LLVM IR: {e}") from e


# Compile Triton IR.
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

            try:
                result = subprocess.run(
                    ["python3", tmp_path],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=60,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                shutil.rmtree(cache_dir, ignore_errors=True)
                logger.error(
                    f"Triton code execution failed:\n{e.stderr}", exc_info=True
                )
                raise TritonExecutionError(
                    f"Triton code execution raised an exception: {e.stderr}"
                ) from e
            finally:
                # Prevent tmp leaks
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            cached_triton_runs[code_hash] = {"cache_dir": cache_dir, "active_users": 0}

        pattern_map = {
            "triton_ir": "*.ttir",
            "triton_gpu_ir": "*.ttgir",
            "triton_llvm_ir": "*.llir",
            "triton_nvptx": "*.ptx",
            "triton_amdgpu": "*.hsaco",
        }

        pattern = pattern_map.get(ir_type)
        if not pattern:
            raise IRGenerationError(f"Unsupported Triton IR type: '{ir_type}'")

        files = glob.glob(os.path.join(cache_dir, "**", pattern), recursive=True)

        if not files:
            raise TritonExecutionError(
                "Triton code execution failed: IR files not found"
            )

        output_parts = []
        for file_path in sorted(files):
            with open(file_path, "r", errors="ignore") as f:
                output_parts.append(f.read())

        cached_triton_runs[code_hash]["active_users"] += 1

        ir_dump = "\n\n".join(output_parts)
        return apply_optional_passes(ir_dump, pipeline, dump_each)

    except Exception as e:
        logger.exception("Error generating Triton IR.")
        raise TritonCompilationError(f"Failed to compile Triton IR: {e}") from e


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
    if request.triton_opt:
        for stage in request.triton_opt.split("&&"):
            pipeline.append(("triton-opt", stage.strip()))
    if request.triton_llvm_opt:
        for stage in request.triton_llvm_opt.split("&&"):
            pipeline.append(("triton-llvm-opt", stage.strip()))
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

        if request.ir_type == "raw_ir" and request.selected_language in (
            "pytorch",
            "triton",
        ):
            # If raw IR is requested, we execute the user code directly.
            # Prepare a fake file for linecache to make
            # inspect.getsourcelines() work.
            fake_name = "<string>"
            source_code = request.code
            lines = [ln + "\n" for ln in source_code.splitlines()]
            linecache.cache[fake_name] = (len(source_code), None, lines, fake_name)
            # Execute user code, capture stdout.
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
                if request.selected_language == "pytorch":
                    raise PytorchExecutionError(
                        f"Code raised an exception during execution: {e}"
                    ) from e
                else:
                    raise TritonExecutionError(
                        f"Triton code execution raised an exception: {e}"
                    ) from e

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
            elif request.ir_type in ("nvptx", "amdgpu", "spirv"):
                # FIXME?: it could really be just generate_llvm_ir with the pipeline.
                # Yet I prefered a dedicated function in case of some smart things has
                # to be done before lowering. For example for SPIR-V it's nice idea
                # to create a kernel first aka write a pass and execute it here.
                combined_output += generate_target_gpu_ir(
                    model, example_input, request.ir_type
                )
            else:
                combined_output += "IR type not supported yet."

        return combined_output.strip()

    except Exception as e:
        logger.exception("Unhandled error in process_model")
        raise IRGenerationError(
            f"Unhandled exception while processing request: {e}"
        ) from e


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
        tb = traceback.format_exc()
        return {
            "status": "error",
            "message": str(e),
            "detail": tb,
        }
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Unexpected internal error:\n%s", tb)
        return {
            "status": "error",
            "message": "Internal backend error",
            "detail": tb,
        }
