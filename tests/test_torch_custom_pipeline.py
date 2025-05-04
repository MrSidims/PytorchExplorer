import httpx
import pytest

BASE_URL = "http://localhost:8000/generate_ir"

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
def test_custom_pipeline(ir_type):
    payload = {
        "code": pytorch_code,
        "ir_type": ir_type,
        "torch_mlir_opt": "",  # no torch-mlir-opt
        "mlir_opt": "",  # no mlir-opt
        "mlir_translate": "--mlir-to-llvmir",
        "llvm_opt": "",  # no opt
        "llc": "-mtriple=nvptx64",
        "user_tool": "",
        "dump_after_each_opt": True,
    }

    response = httpx.post(BASE_URL, json=payload)
    assert response.status_code == 200
    output = response.json()["output"]

    assert "===== Initial IR =====" in output
    assert "llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64)"
    assert "===== IR after mlir-translate --mlir-to-llvmir =====" in output
    assert "define { ptr, ptr, i64, [2 x i64], [2 x i64] } @main" in output
    assert "===== IR after llc -mtriple=nvptx64 =====" in output
    assert ".visible .func  (.param .align 8 .b8 func_retval0[56]) main" in output
