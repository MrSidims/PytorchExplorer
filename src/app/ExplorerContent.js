"use client";

import React, { useEffect, useState } from "react";
import { useSession } from "./SessionContext";
import Editor, { loader } from "@monaco-editor/react";
import { ConfigProvider, Splitter, Collapse, Input, Tabs } from "antd";

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
  { value: "nvptx", label: "NVPTX" },
  { value: "amdgpu", label: "AMDGPU" },
  { value: "spirv", label: "SPIR-V" },
  { value: "raw_ir", label: "Raw IR Output" },
];

const tritonIROptions = [
  { value: "triton_ir", label: "Triton IR" },
  { value: "triton_gpu_ir", label: "Triton GPU IR" },
  { value: "triton_llvm_ir", label: "LLVM IR" },
  { value: "triton_nvptx", label: "NVPTX" },
];

const rawIROptions = [{ value: "raw_ir", label: "Raw IR Output" }];

export default function ExplorerContent() {
  const { sources, setSources, activeSourceId, setActiveSourceId } =
    useSession();

  function updateActiveSource(updater) {
    setSources((prev) =>
      prev.map((s) => (s.id === activeSourceId ? updater(s) : s)),
    );
  }

  const activeSource = sources.find((s) => s.id === activeSourceId);
  const { selectedLanguage, code, irWindows, customToolCmd } = activeSource;

  const [globalLoading, setGlobalLoading] = useState(false);
  const [editPassWindowId, setEditPassWindowId] = useState(null);
  const [editPassIndex, setEditPassIndex] = useState(null);
  const [editTool, setEditTool] = useState("");
  const [editFlags, setEditFlags] = useState("");
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [layout, setLayout] = useState("vertical");
  const [exploreStage, setExploreStage] = useState({
    windowId: null,
    stageIdx: null,
  });

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

  const addSource = () => {
    const newId = sources.length + 1;
    const template = {
      id: newId,
      name: `Source ${newId}`,
      selectedLanguage: "pytorch",
      code: defaultPyTorchCode,
      irWindows: [
        {
          id: 1,
          selectedIR: "torch_script_graph_ir",
          output: "Select IR and Generate",
          collapsed: false,
          loading: false,
          pipeline: [],
          dumpAfterEachOpt: false,
        },
      ],
      customToolCmd: {},
    };
    setSources((prev) => [...prev, template]);
    setActiveSourceId(newId);
  };

  const handleAddPass = (id, tool) => {
    const flags = prompt(`Enter flags for ${tool}`);
    if (flags !== null) {
      updateActiveSource((s) => ({
        ...s,
        irWindows: s.irWindows.map((w) =>
          w.id === id
            ? { ...w, pipeline: [...w.pipeline, { tool, flags }] }
            : w,
        ),
      }));
    }
  };

  const handleAddCustomTool = (windowId, flags) => {
    updateActiveSource((s) => ({
      ...s,
      irWindows: s.irWindows.map((w) =>
        w.id === windowId
          ? { ...w, pipeline: [...w.pipeline, { tool: "user-tool", flags }] }
          : w,
      ),
      customToolCmd: { ...s.customToolCmd, [windowId]: flags },
    }));
  };

  const handleEditPass = (windowId, index, tool, flags) => {
    setEditPassWindowId(windowId);
    setEditPassIndex(index);
    setEditTool(tool);
    setEditFlags(flags);
    setEditModalVisible(true);
  };

  const handleUpdatePass = () => {
    updateActiveSource((s) => ({
      ...s,
      irWindows: s.irWindows.map((w) => {
        if (w.id !== editPassWindowId) return w;

        // replace only the single pass in w.pipeline
        return {
          ...w,
          pipeline: w.pipeline.map((pass, idx) =>
            idx === editPassIndex ? { tool: editTool, flags: editFlags } : pass,
          ),
        };
      }),
    }));
    setEditModalVisible(false);
  };

  const handleRemovePass = () => {
    updateActiveSource((s) => ({
      ...s,
      irWindows: s.irWindows.map((w) => {
        if (w.id !== editPassWindowId) return w;

        // drop just that one pass in w.pipeline
        return {
          ...w,
          pipeline: w.pipeline.filter((_, idx) => idx !== editPassIndex),
        };
      }),
    }));
    setEditModalVisible(false);
  };

  const handleLanguageChange = (e) => {
    const lang = e.target.value;
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
    updateActiveSource((s) => ({
      ...s,
      selectedLanguage: lang,
      code: defaultCode,
      irWindows: [
        {
          id: 1,
          selectedIR: defaultIR,
          output: "Select IR and Generate",
          collapsed: false,
          loading: false,
          pipeline: [],
          dumpAfterEachOpt: false,
        },
      ],
      customToolCmd: {},
    }));
  };

  const addWindow = () => {
    updateActiveSource((s) => {
      const nextId = s.irWindows.length
        ? Math.max(...s.irWindows.map((w) => w.id)) + 1
        : 1;
      return {
        ...s,
        irWindows: [
          ...s.irWindows,
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
        ],
      };
    });
  };

  const removeWindow = (id) => {
    updateActiveSource((s) => ({
      ...s,
      irWindows: s.irWindows.filter((w) => w.id !== id),
    }));
  };

  const toggleCollapse = (id) => {
    updateActiveSource((s) => ({
      ...s,
      irWindows: s.irWindows.map((w) =>
        w.id === id ? { ...w, collapsed: !w.collapsed } : w,
      ),
    }));
  };

  const handleIRChange = (id, event) => {
    const value = event.target.value;
    updateActiveSource((s) => ({
      ...s,
      irWindows: s.irWindows.map((w) =>
        w.id === id ? { ...w, selectedIR: value, pipeline: [] } : w,
      ),
    }));
  };

  function filterToStage(fullDump, stageIdx) {
    const slices = fullDump.split(/===== /).slice(1);
    const block = slices[stageIdx + 1];
    if (!block) {
      // out-of-range - fallback to dump after each stage
      return fullDump.replace(/^===== .* =====\n/gm, "").trim();
    }
    const lines = block.split("\n");
    return lines.slice(1).join("\n").trim();
  }

  const generateIR = async (id) => {
    const exploring = exploreStage.windowId === id;
    updateActiveSource((s) => ({
      ...s,
      irWindows: s.irWindows.map((w) =>
        w.id === id ? { ...w, loading: true } : w,
      ),
    }));

    const irWin = activeSource.irWindows.find((w) => w.id === id);
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
      dump_after_each_opt: exploring ? true : irWin.dumpAfterEachOpt,
    };

    const apiBase =
      process.env.NEXT_PUBLIC_BACKEND_URL ||
      (typeof window !== "undefined"
        ? "http://" + window.location.hostname + ":8000"
        : "http://localhost:8000");
    const response = await fetch(
      `${apiBase}/generate_ir`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    );

    const data = await response.json();
    updateActiveSource((s) => ({
      ...s,
      irWindows: s.irWindows.map((w) =>
        w.id === id
          ? {
              ...w,
              output:
                data.status === "ok"
                  ? data.output
                  : `${data.message}${data.detail ? "\n\n" + data.detail : ""}`,
              loading: false,
            }
          : w,
      ),
    }));
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

  const sourceTabs = sources.map((src) => ({
    label: src.name,
    key: `${src.id}`,
  }));

  return (
    <ConfigProvider
      theme={{
        components: {
          Splitter: {
            splitBarSize: 4,
            splitBarDraggableSize: 24,
          },
        },
      }}
    >
      <Splitter
        layout="horizontal"
        lazy={false}
        style={{
          height: "calc(100vh - 8px)",
          margin: "4px 0",
          boxSizing: "border-box",
          backgroundImage: "url('katze.png')",
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
        }}
      >
        {/* Left Panel */}
        <Splitter.Panel collapsible defaultSize="40%" min="200px">
          <div
            style={{
              margin: "4px 0",
              display: "flex",
              flexDirection: "column",
              height: "98%",
              overflow: "hidden",
              backgroundColor: "white",
              opacity: 0.9,
              borderRadius: "8px",
              padding: "10px",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <h2 style={{ margin: 0 }}>Source code tabs</h2>
            </div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                padding: "4px 10px",
              }}
            >
              <Tabs
                activeKey={String(activeSourceId)}
                type="editable-card"
                hideAdd
                onChange={(key) => setActiveSourceId(Number(key))}
                onEdit={(targetKey, action) => {
                  if (action === "remove") {
                    setSources((prev) => {
                      const next = prev.filter(
                        (src) => String(src.id) !== targetKey,
                      );
                      if (String(activeSourceId) === targetKey) {
                        setActiveSourceId(next[0]?.id ?? null);
                      }
                      return next;
                    });
                  }
                }}
                items={sources.map((src) => ({
                  key: String(src.id),
                  label: `${src.name} (${src.selectedLanguage})`,
                  closable: sources.length > 1,
                }))}
              />
              <button
                onClick={addSource}
                style={{
                  display: "flex",
                  marginLeft: "auto",
                  backgroundColor: "#5fa",
                  fontSize: "0.8rem",
                  color: "black",
                  border: "none",
                  borderRadius: "4px",
                  padding: "4px 8px",
                  cursor: "pointer",
                }}
              >
                Add Source
              </button>
            </div>

            <select
              value={selectedLanguage}
              onChange={handleLanguageChange}
              style={{ margin: "10px 0" }}
            >
              <option value="pytorch">PyTorch</option>
              <option value="raw_ir">Raw IR Input</option>
              <option value="triton">Triton (experimental support)</option>
            </select>
            <div
              style={{
                flex: "1 1 auto",
                overflow: "auto",
              }}
            >
              <Editor
                height="100%"
                defaultLanguage="python"
                value={code}
                onChange={(value) =>
                  updateActiveSource((s) => ({ ...s, code: value || "" }))
                }
                theme="vs-light"
              />
            </div>
          </div>
        </Splitter.Panel>

        {/* Right Panel */}
        <Splitter.Panel defaultSize="60%" min="300px">
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
              height: "100%",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "10px",
                padding: "0 10px",
              }}
            >
              <select
                value={layout}
                onChange={(e) => setLayout(e.target.value)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  padding: "2px",
                  fontSize: "0.7rem",
                  height: "30px",
                  borderRadius: "5px",
                  backgroundColor: "#ccc",
                  border: "none",
                  fontWeight: "bold",
                }}
              >
                <option value="vertical">Vertical Layout</option>
                <option value="horizontal">Horizontal Layout</option>
              </select>
              <div style={{ display: "flex", gap: "6px" }}>
                <button
                  onClick={addWindow}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: "2px",
                    fontSize: "0.7rem",
                    height: "30px",
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
                  onClick={async () => {
                    try {
                      const resp = await fetch("/api/sessions", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ sources, activeSourceId }),
                      });
                      if (!resp.ok) {
                        throw new Error("Failed to save session");
                      }
                      const { id } = await resp.json();
                      window.history.replaceState(null, "", `/${id}`);
                      alert(`Saved session ID: ${id}`);
                    } catch (e) {
                      alert("Failed to save session");
                      console.error(e);
                    }
                  }}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: "2px",
                    fontSize: "0.7rem",
                    height: "30px",
                    backgroundColor: "#5fa",
                    border: "none",
                    borderRadius: "5px",
                    fontWeight: "bold",
                    cursor: "pointer",
                  }}
                >
                  Store Session
                </button>
                <button
                  onClick={generateAllIR}
                  disabled={globalLoading}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: "2px",
                    fontSize: "0.7rem",
                    height: "30px",
                    backgroundColor: "#5fa",
                    border: "none",
                    borderRadius: "5px",
                    fontWeight: "bold",
                    cursor: "pointer",
                  }}
                >
                  {globalLoading
                    ? "🔄 Generating..."
                    : "Generate IR on All Windows"}
                </button>
              </div>
            </div>

            {/* IR windows container */}
            <div
              style={{
                flex: "1 1 0",
                minHeight: 0,
                overflowY: "auto",
                padding: "0 10px 10px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  flexDirection: layout === "vertical" ? "column" : "row",
                  flexWrap: layout === "horizontal" ? "wrap" : "nowrap",
                  gap: "6px",
                  alignItems: "stretch",
                  height: "100%",
                }}
              >
                {irWindows.map((irWin) => (
                  <div
                    key={irWin.id}
                    style={{
                      backgroundColor: "white",
                      borderRadius: "8px",
                      padding: "10px",
                      opacity: 0.9,
                      boxShadow: "0 0 10px rgba(0,0,0,0.2)",
                      display: "flex",
                      flexDirection: "column",
                      flex:
                        irWin.collapsed
                          ? "0 0 auto"
                          : layout === "horizontal"
                            ? "1 1 calc(50% - 6px)"
                            : "0 0 auto",
                      minWidth: layout === "horizontal" ? "30%" : "auto",
                      height: irWin.collapsed ? "auto" : "100%",
                      boxSizing: "border-box",
                      minHeight: 0,
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        marginBottom: "10px",
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
                        {(() => {
                          // determine which built-in tool buttons to show
                          const allowTorchMlirOpt = [
                            "torch_script_graph_ir",
                            "torch_mlir",
                            "raw_ir",
                          ].includes(irWin.selectedIR);
                          const allowMlirOpt = [
                            "torch_script_graph_ir",
                            "torch_mlir",
                            "tosa_mlir",
                            "linalg_on_tensors_mlir",
                            "stablehlo_mlir",
                            "llvm_mlir",
                            "raw_ir",
                          ].includes(irWin.selectedIR);
                          const allowMlirTranslate = allowMlirOpt;
                          const allowTritonOpt = [
                            "triton_ir",
                            "triton_gpu_ir",
                          ].includes(irWin.selectedIR);
                          const allowTritonLLVMOpt = [
                            "triton_ir",
                            "triton_gpu_ir",
                            "triton_llvm_ir",
                          ].includes(irWin.selectedIR);
                          const allowLlvmOpt = ![
                            "triton_nvptx",
                            "nvptx",
                            "amdgpu",
                            "spirv",
                          ].includes(irWin.selectedIR);
                          const allowLLC = allowLlvmOpt;
                          const allowUserTool = true;

                          if (
                            !allowTorchMlirOpt &&
                            !allowMlirOpt &&
                            !allowMlirTranslate &&
                            !allowTritonOpt &&
                            !allowTritonLLVMOpt &&
                            !allowLlvmOpt &&
                            !allowLLC &&
                            !allowUserTool
                          ) {
                            return null;
                          }

                          return (
                            <>
                              <Collapse
                                defaultActiveKey={["1"]}
                                style={{ marginBottom: "10px" }}
                                items={[
                                  {
                                    key: "tools",
                                    label: (
                                      <span
                                        style={{
                                          fontSize: "0.85rem",
                                          padding: "4px 8px",
                                        }}
                                      >
                                        Use predefined toolbar
                                      </span>
                                    ),
                                    children: (
                                      <div
                                        style={{
                                          display: "flex",
                                          flexWrap: "wrap",
                                          gap: "20px",
                                        }}
                                      >
                                        {allowTorchMlirOpt && (
                                          <button
                                            onClick={() =>
                                              handleAddPass(
                                                irWin.id,
                                                "torch-mlir-opt",
                                              )
                                            }
                                            className="tool-btn"
                                          >
                                            torch-mlir-opt
                                          </button>
                                        )}
                                        {allowMlirOpt && (
                                          <button
                                            onClick={() =>
                                              handleAddPass(
                                                irWin.id,
                                                "mlir-opt",
                                              )
                                            }
                                            className="tool-btn"
                                          >
                                            mlir-opt
                                          </button>
                                        )}
                                        {allowMlirTranslate && (
                                          <button
                                            onClick={() =>
                                              handleAddPass(
                                                irWin.id,
                                                "mlir-translate",
                                              )
                                            }
                                            className="tool-btn"
                                          >
                                            mlir-translate
                                          </button>
                                        )}
                                        {allowTritonOpt && (
                                          <button
                                            onClick={() =>
                                              handleAddPass(
                                                irWin.id,
                                                "triton-opt",
                                              )
                                            }
                                            className="tool-btn"
                                          >
                                            triton-opt
                                          </button>
                                        )}
                                        {allowTritonLLVMOpt && (
                                          <button
                                            onClick={() =>
                                              handleAddPass(
                                                irWin.id,
                                                "triton-llvm-opt",
                                              )
                                            }
                                            className="tool-btn"
                                          >
                                            triton-llvm-opt
                                          </button>
                                        )}
                                        {allowLlvmOpt && (
                                          <button
                                            onClick={() =>
                                              handleAddPass(irWin.id, "opt")
                                            }
                                            className="tool-btn"
                                          >
                                            opt
                                          </button>
                                        )}
                                        {allowLLC && (
                                          <button
                                            onClick={() =>
                                              handleAddPass(irWin.id, "llc")
                                            }
                                            className="tool-btn"
                                          >
                                            llc
                                          </button>
                                        )}
                                      </div>
                                    ),
                                  },
                                ]}
                              />
                              <Input
                                placeholder="Call any tool in PATH + flags, e.g. `mlir-opt -convert-scf-to-cf` or `opt -O2 -S` and press `Enter`"
                                value={customToolCmd[irWin.id] || ""}
                                onChange={(e) =>
                                  updateActiveSource((s) => ({
                                    ...s,
                                    customToolCmd: {
                                      ...s.customToolCmd,
                                      [irWin.id]: e.target.value,
                                    },
                                  }))
                                }
                                onPressEnter={() => {
                                  const flags = (
                                    customToolCmd[irWin.id] || ""
                                  ).trim();
                                  if (!flags) return;
                                  handleAddCustomTool(irWin.id, flags);
                                  updateActiveSource((s) => ({
                                    ...s,
                                    customToolCmd: {
                                      ...s.customToolCmd,
                                      [irWin.id]: "",
                                    },
                                  }));
                                }}
                                style={{
                                  width: "100%",
                                  marginBottom: "10px",
                                }}
                              />
                            </>
                          );
                        })()}
                        {irWin.pipeline.length > 0 && (
                          <div
                            style={{
                              marginBottom: "10px",
                              backgroundColor: "#eef",
                              padding: "6px",
                              borderRadius: "6px",
                              fontSize: "0.8em",
                            }}
                          >
                            <div
                              style={{
                                marginBottom: "4px",
                                backgroundColor: "#eee",
                                padding: "4px 4px",
                                borderRadius: "6px",
                                display: "flex",
                                flexWrap: "wrap",
                                alignItems: "center",
                                gap: "4px",
                              }}
                            >
                              <span
                                style={{
                                  backgroundColor: "#ccc",
                                  padding: "2px 4px",
                                  borderRadius: "4px",
                                  fontWeight: "bold",
                                }}
                              >
                                Compilation pipeline:
                              </span>
                              <span
                                style={{
                                  backgroundColor: "#ffeb3b",
                                  padding: "2px 4px",
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
                                    : `${p.flags.slice(0, 15)}…${p.flags.slice(
                                        -10,
                                      )}`;
                                return (
                                  <React.Fragment key={i}>
                                    <span
                                      style={{
                                        fontSize: "1.2em",
                                        color: "#666",
                                      }}
                                    >
                                      →
                                    </span>
                                    <span
                                      onClick={() =>
                                        handleEditPass(
                                          irWin.id,
                                          i,
                                          p.tool,
                                          p.flags,
                                        )
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
                                      {p.tool !== "user-tool" && (
                                        <span>{p.tool}</span>
                                      )}
                                      <span
                                        style={{
                                          fontSize: "0.8em",
                                          color: "#555",
                                        }}
                                      >
                                        {preview}
                                      </span>
                                    </span>
                                    <button
                                      onClick={() => {
                                        setExploreStage({
                                          windowId: irWin.id,
                                          stageIdx: i,
                                        });
                                        generateIR(irWin.id);
                                      }}
                                      style={{
                                        marginLeft: 4,
                                        padding: "2px 6px",
                                        cursor: "pointer",
                                      }}
                                    >
                                      🔍
                                    </button>
                                  </React.Fragment>
                                );
                              })}
                            </div>
                          </div>
                        )}
                        <div
                          style={{
                            display: "flex",
                            gap: "6px",
                            marginBottom: "6px",
                          }}
                        >
                          <button
                            onClick={() => generateIR(irWin.id)}
                            style={{
                              flex: 3,
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
                          <button
                            onClick={() =>
                              updateActiveSource((s) => ({
                                ...s,
                                irWindows: s.irWindows.map((w) =>
                                  w.id === irWin.id
                                    ? {
                                        ...w,
                                        dumpAfterEachOpt: !w.dumpAfterEachOpt,
                                      }
                                    : w,
                                ),
                              }))
                            }
                            style={{
                              flex: 1,
                              padding: "4px",
                              backgroundColor: irWin.dumpAfterEachOpt
                                ? "#5fa"
                                : "#ccc",
                              border: "none",
                              borderRadius: "5px",
                              fontWeight: "bold",
                              cursor: "pointer",
                            }}
                          >
                            {irWin.dumpAfterEachOpt
                              ? "✓ Print IR after opts"
                              : "Print IR after opts"}
                          </button>
                          {exploreStage.windowId === irWin.id && (
                            <button
                              onClick={() =>
                                setExploreStage({
                                  windowId: null,
                                  stageIdx: null,
                                })
                              }
                              style={{
                                flex: 1,
                                padding: "4px",
                                backgroundColor: "#ccc",
                                border: "none",
                                borderRadius: "5px",
                                fontWeight: "bold",
                                cursor: "pointer",
                              }}
                            >
                              Reset Explore At Stage
                            </button>
                          )}
                        </div>
                        <div
                          style={{
                            flex: 1,
                            minHeight: 0,
                            overflow: "auto",
                          }}
                        >
                          <Editor
                            height="100%"
                            language="mlir"
                            value={
                              exploreStage.windowId === irWin.id
                                ? filterToStage(
                                    irWin.output,
                                    exploreStage.stageIdx,
                                  )
                                : irWin.output
                            }
                            onChange={() => {}}
                            theme="mlirTheme"
                            options={{ readOnly: true }}
                          />
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
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
                  <button onClick={() => setEditModalVisible(false)}>
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        </Splitter.Panel>
      </Splitter>
    </ConfigProvider>
  );
}
