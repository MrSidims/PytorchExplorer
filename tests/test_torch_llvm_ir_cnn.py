import os
import pytest
import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000/generate_ir")


def test_torch_cnn():
    code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = SimpleCNN(num_classes=10)
x = torch.randn(8, 1, 28, 28)

"""

    payload = {
        "code": code,
        "ir_type": "llvm_ir",
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

    assert "@main" in ir
