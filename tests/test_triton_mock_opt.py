import os
import subprocess
import tempfile
import shutil
import pytest
import httpx

@pytest.fixture(scope="session")
def mock_opt_path():
    cpp_src = os.path.abspath("tests/cpp_sources/mock-opt.cpp")
    build_dir = tempfile.mkdtemp()
    exe_path = os.path.join(build_dir, "mock-opt")

    subprocess.check_call(["g++", cpp_src, "-o", exe_path])

    yield exe_path

    shutil.rmtree(build_dir)

def test_triton_mock_opt(mock_opt_path):
    code = """
import triton
import triton.language as tl
import torch

BLOCK_SIZE = tl.constexpr(1024)

@triton.jit
def add_kernel(X, Y, Z, N):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    z = x + y
    tl.store(Z + offsets, z, mask=mask)

N = 4096
x = torch.randn(N, device="cuda", dtype=torch.float32)
y = torch.randn(N, device="cuda", dtype=torch.float32)
z = torch.empty_like(x)

grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
add_kernel[grid](x, y, z, N)

"""

    payload = {
        "code": code,
        "ir_type": "triton_gpu_ir",
        "user_tool": f"{mock_opt_path}",
        "dump_after_each_opt": False,
    }

    response = httpx.post("http://localhost:8000/generate_ir", json=payload)
    assert response.status_code == 200
    assert "tt.func public @add_kernel" in response.json()["output"]
    assert "test mock_opt 42" in response.json()["output"]
