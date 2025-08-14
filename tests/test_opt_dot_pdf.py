import os
import httpx
import pytest

BASE_URL = os.environ.get("API_URL", "http://localhost:8000/generate_ir")

LLVM_IR = """
define i32 @foo(i32 %x) {
entry:
  %0 = add i32 %x, 1
  ret i32 %0
}
"""


def test_opt_dot_generates_pdf():
    payload = {
        "code": LLVM_IR,
        "ir_type": "raw_ir",
        "selected_language": "raw_ir",
        "torch_mlir_opt": "",
        "mlir_opt": "",
        "mlir_translate": "",
        "llvm_opt": "-passes=dot-cfg",
        "llc": "",
        "triton_opt": "",
        "triton_llvm_opt": "",
        "user_tool": "",
        "dump_after_each_opt": False,
    }
    resp = httpx.post(BASE_URL, json=payload)
    assert resp.status_code == 200, resp.text
    out = resp.json()["output"]

    # IR is still printed
    assert "define i32 @foo" in out

    # At least one embedded PDF block from the DOT conversion
    assert "===== DOT PDF " in out
    assert "JVBERi0" in out  # base64 PDF header
