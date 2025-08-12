import os
import httpx
import pytest

BASE_URL = os.environ.get("API_URL", "http://localhost:8000/generate_ir")

pytorch_code = """
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.linear(x))

model = MyModel()
example_input = torch.randn(4, 4)
"""


@pytest.mark.parametrize("ir_type", ["llvm_mlir"])
def test_wrong_command_line(ir_type):
    payload = {
        "code": pytorch_code,
        "ir_type": ir_type,
        "torch_mlir_opt": "",
        "mlir_opt": "",
        "mlir_translate": "--mlir-to-llvmir",
        "llvm_opt": "",
        "llc": "abcde",
        "user_tool": "",
        "dump_after_each_opt": True,
    }

    r = httpx.post(BASE_URL, json=payload)
    assert r.status_code == 200

    data = r.json()
    assert data["status"] == "error"

    assert data["message"].startswith(
        "Unhandled exception while processing request: Failed to generate LLVM MLIR"
    )
    assert "llc" in data["message"] or "Compiler tool" in data["message"]
    assert "Traceback (most recent call last)" in data["detail"]
    assert "CompilerPipelineError" in data["detail"]
    assert "llc" in data["detail"]
