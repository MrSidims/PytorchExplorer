import pytest
import httpx

API_URL = "http://localhost:8000/generate_ir"


def test_simple_linear_relu_model():
    code = """
import torch
import torch.nn as nn

class MyModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.linear(x))

model1 = MyModel1()
example_input1 = torch.randn(4, 4)
__explore__(model1, example_input1)

class MyModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

model2 = MyModel2()
example_input2 = torch.randn(1, 1, 1, 1)
__explore__(model2, example_input2)
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

    assert "torch.prim.ListConstruct" in ir
    assert "torch.aten.convolution" in ir
    assert "torch.vtensor.literal" in ir
