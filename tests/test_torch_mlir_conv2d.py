import os
import pytest
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000/generate_ir")


def test_torch_mlir_conv2d():
    code = """
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

model = MyModel()
example_input = torch.randn(1, 1, 1, 1)
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

    assert "torch.prim.ListConstruct" in ir
    assert "torch.aten.convolution" in ir
    assert "torch.vtensor.literal" in ir
