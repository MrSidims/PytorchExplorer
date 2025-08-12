import os
import pytest
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000/generate_ir")


def test_torch_user_controlled_ir_print():
    code = """
import torch
import torch.nn as nn
from torch_mlir import fx
from torch_mlir.compiler_utils import run_pipeline_with_repro_report
from torch_mlir.fx import OutputType

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.linear(x))

model = MyModel()
example_input = torch.randn(4, 4)
module = fx.export_and_import(model, example_input, output_type=OutputType.LINALG_ON_TENSORS)
print(module)
"""

    payload = {
        "code": code,
        "ir_type": "raw_ir",
        "custom_pipeline": [],
        "torch_mlir_opt": "",
        "mlir_opt": '--one-shot-bufferize="bufferize-function-boundaries"',
        "mlir_translate": "",
        "llvm_opt": "",
        "llc": "",
        "user_tool": "",
        "dump_after_each_opt": False,
    }

    response = httpx.post(API_URL, json=payload)
    assert response.status_code == 200

    ir = response.json()["output"]

    assert "affine_map" in ir
    assert "memref.global" in ir
    assert "arith.constant" in ir
    assert "linalg.matmul" in ir
