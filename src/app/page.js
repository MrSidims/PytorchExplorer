"use client";

import React from "react";
import { useState, useEffect } from "react";
import Editor, { loader } from "@monaco-editor/react";

const defaultPyTorchCode = `import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.linear(x))

model = MyModel()
example_input = torch.randn(4, 4)
# If you have multiple models, wrap each model and input tensor pair using:
# __explore__(model, input_tensor)
`;

const defaultTritonCode = `import triton
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
`;

const defaultRawIRCode = `module {
  func.func @main(%arg0: !torch.vtensor<[1,1,5,5],f32>) -> !torch.vtensor<[1,1,3,3],f32> {
    %false = torch.constant.bool false
    %int0 = torch.constant.int 0
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_1_3_3_torch.float32> : tensor<1x1x3x3xf32>) : !torch.vtensor<[1,1,3,3],f32>
    %1 = torch.vtensor.literal(dense<-0.296203673> : tensor<1xf32>) : !torch.vtensor<[1],f32>
    %int1 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %6 = torch.aten.convolution %arg0, %0, %1, %2, %3, %4, %false, %5, %int1 : !torch.vtensor<[1,1,5,5],f32>, !torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,3,3],f32>
    return %6 : !torch.vtensor<[1,1,3,3],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_1_1_3_3_torch.float32: "0x040000008074FFBD60CFA9BE8048673CDB4721BE88D26BBE36C07B3DCB82693E5634363C502184BE"
    }
  }
#-}`;

const pytorchIROptions = [
  { value: "torch_script_graph_ir", label: "TorchScript Graph IR" },
  { value: "torch_mlir", label: "Torch MLIR" },
  { value: "tosa_mlir", label: "TOSA MLIR" },
  { value: "linalg_on_tensors_mlir", label: "Linalg on Tensors MLIR" },
  { value: "stablehlo_mlir", label: "StableHLO MLIR" },
  { value: "llvm_mlir", label: "LLVM MLIR" },
  { value: "llvm_ir", label: "LLVM IR" },
  { value: "raw_ir", label: "Raw IR Output" },
];

const tritonIROptions = [
  { value: "triton_ir", label: "Triton IR" },
  { value: "triton_gpu_ir", label: "Triton GPU IR" },
  { value: "triton_llvm_ir", label: "LLVM IR" },
  { value: "triton_nvptx", label: "NVPTX" },
];

const rawIROptions = [{ value: "raw_ir", label: "Raw IR Output" }];

export default function PyTorchTritonExplorer() {
  const [selectedLanguage, setSelectedLanguage] = useState("pytorch");
  const [code, setCode] = useState(defaultPyTorchCode);
  const [irWindows, setIrWindows] = useState([
    {
      id: 1,
      selectedIR: "torch_script_graph_ir",
      output: "Select IR and Generate",
      collapsed: false,
      loading: false,
      pipeline: [],
      dumpAfterEachOpt: false,
    },
  ]);
  const [globalLoading, setGlobalLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [currentWindowId, setCurrentWindowId] = useState(null);
  const [currentTool, setCurrentTool] = useState("");
  const [currentFlags, setCurrentFlags] = useState("");
  const [editPassWindowId, setEditPassWindowId] = useState(null);
  const [editPassIndex, setEditPassIndex] = useState(null);
  const [editTool, setEditTool] = useState("");
  const [editFlags, setEditFlags] = useState("");
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [sourceCollapsed, setSourceCollapsed] = useState(false);

  const isTriton = selectedLanguage === "triton";
  const isPytorch = selectedLanguage === "pytorch";

  useEffect(() => {
    loader.init().then((monaco) => {
      if (!monaco) return;

      monaco.languages.register({ id: "mlir" });

      monaco.languages.setMonarchTokensProvider("mlir", {
        tokenizer: {
          root: [
            [/%[a-zA-Z0-9_.]+/, "variable"],
            [/[(),]/, "delimiter"],
            [/->/, "operator"],
            [
              /\b(func|module|return|scf|affine|loop|br|if|else|while|for|global|memref|tensor|vector|llvm|cf|arith|select|unroll)\b/,
              "keyword",
            ],
            [/\b(i[0-9]+|f[0-9]+|tensor|memref|index|vector|none)\b/, "type"],
            [/[0-9]+(\.[0-9]+)?/, "number"],
            [/".*?"/, "string"],
            [/\/\/.*$/, "comment"],
            [/;/, "comment"],
          ],
        },
      });

      monaco.editor.defineTheme("mlirTheme", {
        base: "vs",
        inherit: true,
        rules: [
          { token: "variable", foreground: "1F618D" },
          { token: "keyword", foreground: "D35400" },
          { token: "type", foreground: "8E44AD" },
          { token: "number", foreground: "27AE60" },
          { token: "string", foreground: "D81B60" },
          { token: "comment", foreground: "7F8C8D" },
        ],
        colors: {
          "editor.foreground": "#000000",
          "editor.background": "#FFFFFF",
        },
      });
    });
  }, []);

  const handleAddPass = (id, tool) => {
    const flags = prompt(`Enter flags for ${tool}`);
    if (flags !== null) {
      setIrWindows((prev) =>
        prev.map((w) =>
          w.id === id ? { ...w, pipeline: [...w.pipeline, { tool, flags }] } : w
        )
      );
    }
  };

  const handleSubmitFlags = () => {
    setIrWindows((prev) =>
      prev.map((w) =>
        w.id === currentWindowId
          ? {
              ...w,
              pipeline: [
                ...w.pipeline,
                { tool: currentTool, flags: currentFlags },
              ],
            }
          : w
      )
    );
    setModalVisible(false);
  };

  const handleEditPass = (windowId, index, tool, flags) => {
    setEditPassWindowId(windowId);
    setEditPassIndex(index);
    setEditTool(tool);
    setEditFlags(flags);
    setEditModalVisible(true);
  };

  const handleUpdatePass = () => {
    setIrWindows((prev) =>
      prev.map((w) =>
        w.id === editPassWindowId
          ? {
              ...w,
              pipeline: w.pipeline.map((p, i) =>
                i === editPassIndex ? { tool: editTool, flags: editFlags } : p
              ),
            }
          : w
      )
    );
    setEditModalVisible(false);
  };

  const handleRemovePass = () => {
    setIrWindows((prev) =>
      prev.map((w) =>
        w.id === editPassWindowId
          ? {
              ...w,
              pipeline: w.pipeline.filter((_, i) => i !== editPassIndex),
            }
          : w
      )
    );
    setEditModalVisible(false);
  };

  const handleLanguageChange = (e) => {
    const lang = e.target.value;
    setSelectedLanguage(lang);
    const defaultIR =
      lang === "pytorch"
        ? "torch_script_graph_ir"
        : lang === "triton"
          ? "triton_ir"
          : "raw_ir";

    const defaultCode =
      lang === "pytorch"
        ? defaultPyTorchCode
        : lang === "triton"
          ? defaultTritonCode
          : defaultRawIRCode;

    setCode(defaultCode);
    setIrWindows([
      {
        id: 1,
        selectedIR: defaultIR,
        output: "Select IR and Generate",
        collapsed: false,
        loading: false,
        pipeline: [],
      },
    ]);
  };

  const addWindow = () => {
    const nextId = Math.max(...irWindows.map((w) => w.id)) + 1;
    setIrWindows([
      ...irWindows,
      {
        id: nextId,
        selectedIR: isTriton
          ? "triton_ir"
          : isPytorch
            ? "torch_script_graph_ir"
            : "raw_ir",
        output: "Select IR and Generate",
        collapsed: false,
        loading: false,
        pipeline: [],
        dumpAfterEachOpt: false,
      },
    ]);
  };

  const removeWindow = (id) => {
    setIrWindows(irWindows.filter((w) => w.id !== id));
  };

  const toggleCollapse = (id) => {
    setIrWindows((prev) =>
      prev.map((w) => (w.id === id ? { ...w, collapsed: !w.collapsed } : w))
    );
  };

  const handleIRChange = (id, event) => {
    const value = event.target.value;
    setIrWindows((prev) =>
      prev.map((w) =>
        w.id === id ? { ...w, selectedIR: value, pipeline: [] } : w
      )
    );
  };

  const generateIR = async (id) => {
    setIrWindows((prev) =>
      prev.map((w) => (w.id === id ? { ...w, loading: true } : w))
    );

    const irWin = irWindows.find((w) => w.id === id);
    if (!irWin) return;

    const body = {
      code,
      ir_type: irWin.selectedIR,
      selected_language: selectedLanguage,
      custom_pipeline: [],
      torch_mlir_opt: irWin.pipeline
        .filter((p) => p.tool === "torch-mlir-opt")
        .map((p) => p.flags)
        .join(" && "),
      mlir_opt: irWin.pipeline
        .filter((p) => p.tool === "mlir-opt")
        .map((p) => p.flags)
        .join(" && "),
      mlir_translate: irWin.pipeline
        .filter((p) => p.tool === "mlir-translate")
        .map((p) => p.flags)
        .join(" && "),
      llvm_opt: irWin.pipeline
        .filter((p) => p.tool === "opt")
        .map((p) => p.flags)
        .join(" && "),
      llc: irWin.pipeline
        .filter((p) => p.tool === "llc")
        .map((p) => p.flags)
        .join(" && "),
      user_tool: irWin.pipeline
        .filter((p) => p.tool === "user-tool")
        .map((p) => p.flags)
        .join(" && "),
      dump_after_each_opt: irWin.dumpAfterEachOpt,
    };

    const response = await fetch("http://" + window.location.hostname + ":8000/generate_ir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await response.json();
    setIrWindows((prev) =>
      prev.map((w) =>
        w.id === id ? { ...w, output: data.output, loading: false } : w
      )
    );
  };

  const generateAllIR = async () => {
    setGlobalLoading(true);
    for (const w of irWindows) {
      await generateIR(w.id);
    }
    setGlobalLoading(false);
  };

  const getIROptions = () =>
    isTriton ? tritonIROptions : isPytorch ? pytorchIROptions : rawIROptions;

  const getLabelForIR = (value) => {
    const found = getIROptions().find((opt) => opt.value === value);
    return found ? found.label : value;
  };

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        padding: "0.2%",
        paddingRight: "0.2%",
        backgroundImage: "url('katze.png')",
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        gap: "0.2%",
        overflow: "hidden",
      }}
    >
      {/* Left Panel */}
      {!sourceCollapsed && (
        <div
          style={{
            width: "34%",
            resize: "horizontal",
            overflow: "auto",
            minWidth: "20%",
            maxWidth: "60%",
            backgroundColor: "white",
            opacity: 0.9,
            borderRadius: "8px",
            padding: "10px",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <h2>Source Code</h2>
            <button
              onClick={() => setSourceCollapsed(true)}
              style={{
                padding: "5px 10px",
                backgroundColor: "#ccc",
                border: "none",
                borderRadius: "5px",
                cursor: "pointer",
              }}
            >
              Collapse
            </button>
          </div>

          <select
            value={selectedLanguage}
            onChange={handleLanguageChange}
            style={{ marginBottom: "10px" }}
          >
            <option value="pytorch">PyTorch</option>
            <option value="raw_ir">Raw IR Input</option>
            <option value="triton">Triton (experimental support)</option>
          </select>

          <Editor
            height="80vh"
            defaultLanguage="python"
            value={code}
            onChange={(value) => setCode(value)}
            theme="vs-light"
          />
        </div>
      )}

      {sourceCollapsed && (
        <div
          style={{
            width: "auto",
            padding: "10px",
          }}
        >
          <button
            onClick={() => setSourceCollapsed(false)}
            style={{
              padding: "5px 10px",
              backgroundColor: "#ccc",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
          >
            Expand Source Panel
          </button>
        </div>
      )}
      {/* Right Panel */}
      <div
        style={{
          flexGrow: 1,
          overflow: "auto",
          display: "flex",
          flexDirection: "column",
          gap: "6px",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            gap: "6px",
            marginBottom: "10px",
          }}
        >
          <button
            onClick={addWindow}
            style={{
              padding: "10px",
              backgroundColor: "#5fa",
              border: "none",
              borderRadius: "5px",
              fontWeight: "bold",
              cursor: "pointer",
            }}
          >
            ➕ Add IR Window
          </button>

          <button
            onClick={generateAllIR}
            style={{
              padding: "10px",
              backgroundColor: "#5fa",
              border: "none",
              borderRadius: "5px",
              fontWeight: "bold",
              cursor: "pointer",
            }}
            disabled={globalLoading}
          >
            {globalLoading ? "🔄 Generating..." : "Generate IR on All Windows"}
          </button>
        </div>

        {irWindows.map((irWin) => (
          <div
            key={irWin.id}
            style={{
              backgroundColor: "white",
              borderRadius: "8px",
              padding: "10px",
              opacity: 0.9,
              boxShadow: "0px 0px 10px rgba(0,0,0,0.2)",
              position: "relative",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <h3>Output {getLabelForIR(irWin.selectedIR)}</h3>
              <div>
                <button
                  onClick={() => toggleCollapse(irWin.id)}
                  style={{
                    marginRight: "8px",
                    padding: "5px",
                    backgroundColor: "#ccc",
                    border: "none",
                    borderRadius: "5px",
                    cursor: "pointer",
                  }}
                >
                  {irWin.collapsed ? "Expand" : "Collapse"}
                </button>
                <button
                  onClick={() => removeWindow(irWin.id)}
                  style={{
                    backgroundColor: "#f66",
                    border: "none",
                    borderRadius: "5px",
                    cursor: "pointer",
                    padding: "5px 10px",
                  }}
                >
                  ×
                </button>
              </div>
            </div>

            {!irWin.collapsed && (
              <>
                <select
                  value={irWin.selectedIR}
                  onChange={(e) => handleIRChange(irWin.id, e)}
                  style={{ marginBottom: "10px" }}
                >
                  {getIROptions().map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
                {(selectedLanguage === "pytorch" ||
                  selectedLanguage == "raw_ir") &&
                  (() => {
                    const allowTorchMlirOpt = ["torch_script_graph_ir", "torch_mlir", "raw_ir"].includes(
                      irWin.selectedIR
                    );
                    const allowMlirOpt = [
                      "torch_script_graph_ir",
                      "torch_mlir",
                      "tosa_mlir",
                      "linalg_on_tensors_mlir",
                      "stablehlo_mlir",
                      "llvm_mlir",
                      "raw_ir",
                    ].includes(irWin.selectedIR);
                    const allowMlirTranslate = [
                      "torch_script_graph_ir",
                      "torch_mlir",
                      "tosa_mlir",
                      "linalg_on_tensors_mlir",
                      "stablehlo_mlir",
                      "llvm_mlir",
                      "raw_ir",
                    ].includes(irWin.selectedIR);
                    const allowLlvmOpt = true;
                    const allowLLC = true;
                    const allowUserTool = true;

                    if (
                      !allowTorchMlirOpt &&
                      !allowMlirOpt &&
                      !allowMlirTranslate &&
                      !allowLlvmOpt &&
                      !allowUserTool
                    )
                      return null;

                    return (
                      <div
                        style={{
                          display: "flex",
                          gap: "8px",
                          marginBottom: "10px",
                          alignItems: "center",
                        }}
                      >
                        {allowTorchMlirOpt && (
                          <button
                            onClick={() =>
                              handleAddPass(irWin.id, "torch-mlir-opt")
                            }
                            style={{
                              padding: "6px 10px",
                              backgroundColor: "#add8e6",
                              border: "none",
                              borderRadius: "5px",
                              fontWeight: "bold",
                              cursor: "pointer",
                              fontSize: "0.65em",
                              whiteSpace: "nowrap",
                              minHeight: "32px",
                              flexShrink: 0,
                            }}
                          >
                            torch-mlir-opt from this
                          </button>
                        )}
                        {allowMlirOpt && (
                          <button
                            onClick={() => handleAddPass(irWin.id, "mlir-opt")}
                            style={{
                              padding: "6px 10px",
                              backgroundColor: "#add8e6",
                              border: "none",
                              borderRadius: "5px",
                              fontWeight: "bold",
                              cursor: "pointer",
                              fontSize: "0.65em",
                              whiteSpace: "nowrap",
                              minHeight: "32px",
                              flexShrink: 0,
                            }}
                          >
                            mlir-opt from this
                          </button>
                        )}
                        {allowMlirTranslate && (
                          <button
                            onClick={() =>
                              handleAddPass(irWin.id, "mlir-translate")
                            }
                            style={{
                              padding: "6px 10px",
                              backgroundColor: "#add8e6",
                              border: "none",
                              borderRadius: "5px",
                              fontWeight: "bold",
                              cursor: "pointer",
                              fontSize: "0.65em",
                              whiteSpace: "nowrap",
                              minHeight: "32px",
                              flexShrink: 0,
                            }}
                          >
                            mlir-translate from this
                          </button>
                        )}
                        {allowLlvmOpt && (
                          <button
                            onClick={() => handleAddPass(irWin.id, "opt")}
                            style={{
                              padding: "6px 10px",
                              backgroundColor: "#add8e6",
                              border: "none",
                              borderRadius: "5px",
                              fontWeight: "bold",
                              cursor: "pointer",
                              fontSize: "0.65em",
                              whiteSpace: "nowrap",
                              minHeight: "32px",
                              flexShrink: 0,
                            }}
                          >
                            opt from this
                          </button>
                        )}
                        {allowLLC && (
                          <button
                            onClick={() => handleAddPass(irWin.id, "llc")}
                            style={{
                              padding: "6px 10px",
                              backgroundColor: "#add8e6",
                              border: "none",
                              borderRadius: "5px",
                              fontWeight: "bold",
                              cursor: "pointer",
                              fontSize: "0.65em",
                              whiteSpace: "nowrap",
                              minHeight: "32px",
                              flexShrink: 0,
                            }}
                          >
                            llc from this
                          </button>
                        )}
                        {allowUserTool && (
                          <button
                            onClick={() =>
                              handleAddPass(irWin.id, "user-tool")
                            }
                            style={{
                              padding: "6px 10px",
                              backgroundColor: "#add8e6",
                              border: "none",
                              borderRadius: "5px",
                              fontWeight: "bold",
                              cursor: "pointer",
                              fontSize: "0.65em",
                              whiteSpace: "nowrap",
                              minHeight: "32px",
                              flexShrink: 0,
                            }}
                          >
                            %your tool% in $PATH from this
                          </button>
                        )}
                        <button
                          onClick={() =>
                            setIrWindows((prev) =>
                              prev.map((w) =>
                                w.id === irWin.id
                                  ? {
                                      ...w,
                                      dumpAfterEachOpt: !w.dumpAfterEachOpt,
                                    }
                                  : w
                              )
                            )
                          }
                          style={{
                            padding: "6px 10px",
                            backgroundColor: irWin.dumpAfterEachOpt
                              ? "#5fa"
                              : "#ccc",
                            border: "none",
                            borderRadius: "5px",
                            fontWeight: "bold",
                            cursor: "pointer",
                            fontSize: "0.65em",
                            whiteSpace: "nowrap",
                            minHeight: "32px",
                            flexShrink: 0,
                          }}
                        >
                          {irWin.dumpAfterEachOpt
                            ? "✓ Print IR after opts"
                            : "Print IR after opts"}
                        </button>
                      </div>
                    );
                  })()}

                {(selectedLanguage === "pytorch" ||
                  selectedLanguage == "raw_ir") &&
                  irWin.pipeline.length > 0 && (
                    <div
                      style={{
                        marginBottom: "10px",
                        backgroundColor: "#eef",
                        padding: "6px",
                        borderRadius: "6px",
                        fontSize: "0.95em",
                      }}
                    >
                      <div
                        style={{
                          marginBottom: "10px",
                          backgroundColor: "#eee",
                          padding: "6px 10px",
                          borderRadius: "6px",
                          fontSize: "0.95em",
                          display: "flex",
                          flexWrap: "wrap",
                          alignItems: "center",
                          gap: "6px",
                        }}
                      >
                        <span
                          style={{
                            backgroundColor: "#ccc",
                            padding: "2px 8px",
                            borderRadius: "4px",
                            fontWeight: "bold",
                          }}
                        >
                          Compilation pipeline:
                        </span>

                        <span
                          style={{
                            backgroundColor: "#ffeb3b",
                            padding: "2px 8px",
                            borderRadius: "4px",
                            fontWeight: "bold",
                          }}
                        >
                          {getLabelForIR(irWin.selectedIR)}
                        </span>
                        {irWin.pipeline.map((p, i) => {
                          const preview =
                            p.flags.length <= 25
                              ? p.flags
                              : `${p.flags.slice(0, 15)}...${p.flags.slice(-10)}`;
                          return (
                            <React.Fragment key={i}>
                              <span
                                style={{ fontSize: "1.2em", color: "#666" }}
                              >
                                →
                              </span>
                              <span
                                onClick={() =>
                                  handleEditPass(irWin.id, i, p.tool, p.flags)
                                }
                                style={{
                                  backgroundColor: "#a5d6a7",
                                  padding: "2px 8px",
                                  borderRadius: "4px",
                                  cursor: "pointer",
                                  fontWeight: "bold",
                                  display: "flex",
                                  flexDirection: "column",
                                  lineHeight: "1.2em",
                                }}
                              >
                                <span>{p.tool}</span>
                                <span
                                  style={{
                                    fontSize: "0.8em",
                                    color: "#555",
                                  }}
                                >
                                  {preview}
                                </span>
                              </span>
                            </React.Fragment>
                          );
                        })}
                      </div>
                    </div>
                  )}

                <button
                  onClick={() => generateIR(irWin.id)}
                  style={{
                    marginBottom: "6px",
                    padding: "4px",
                    backgroundColor: "#5fa",
                    border: "none",
                    borderRadius: "5px",
                    fontWeight: "bold",
                    cursor: "pointer",
                  }}
                  disabled={irWin.loading}
                >
                  {irWin.loading ? "Generating..." : "Generate IR"}
                </button>

                <Editor
                  height="70vh"
                  language="mlir"
                  value={irWin.output}
                  onChange={() => {}}
                  theme="mlirTheme"
                  options={{ readOnly: true }}
                />
              </>
            )}
          </div>
        ))}
      </div>
      {editModalVisible && (
        <div
          style={{
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            backgroundColor: "#fff",
            border: "1px solid #ccc",
            padding: "50px",
            borderRadius: "8px",
            boxShadow: "0 2px 10px rgba(0,0,0,0.3)",
            zIndex: 1000,
          }}
        >
          <h3>Edit Compilation Pass</h3>
          <p>
            <strong>{editTool}</strong>
          </p>
          <input
            type="text"
            value={editFlags}
            onChange={(e) => setEditFlags(e.target.value)}
            style={{ width: "100%", marginBottom: "10px" }}
          />
          <div
            style={{
              display: "flex",
              gap: "200px",
              justifyContent: "flex-end",
            }}
          >
            <button onClick={handleUpdatePass}>Update</button>
            <button onClick={handleRemovePass} style={{ color: "red" }}>
              Remove
            </button>
            <button onClick={() => setEditModalVisible(false)}>Cancel</button>
          </div>
        </div>
      )}
    </div>
  );
}
