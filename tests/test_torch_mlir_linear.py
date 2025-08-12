import os
import pytest
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000/generate_ir")


def test_torch_mlir_linear():
    code = """
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

    payload = {
        "code": code,
        "ir_type": "torch_mlir",
        "custom_pipeline": [],
        "torch_mlir_opt": "",
        "mlir_opt": "",
        "mlir_translate": "",
        "llvm_opt": "",
        "llc": "",
        "user_tool": "",
        "dump_after_each_opt": False,
    }

    response = httpx.post(API_URL, json=payload)
    assert response.status_code == 200

    ir = response.json()["output"]

    assert "torch.aten.mm" in ir
    assert "torch.aten.relu" in ir
    assert "torch.vtensor.literal" in ir
