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

def test_torch_mock_opt(mock_opt_path):
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
        "ir_type": "torch_script_graph_ir",
        "user_tool": f"{mock_opt_path}",
        "dump_after_each_opt": True,
    }

    response = httpx.post(os.environ.get("API_URL", "http://localhost:8000/generate_ir"), json=payload)
    assert response.status_code == 200
    assert "graph(%self.1 : __torch__.builtins.MyModel," in response.json()["output"]
    assert "test mock_opt 42" in response.json()["output"]
