"use client";

import React, { createContext, useContext, useState } from "react";

const SessionContext = createContext({
  sources: [],
  setSources: () => {},
  activeSourceId: null,
  setActiveSourceId: () => {},
});

export function SessionProvider({ children }) {
  const [sources, setSources] = useState([
    {
      id: 1,
      name: "Source 1",
      selectedLanguage: "pytorch",
      code: `import torch
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
# __explore__(model, example_input)
`,
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
    },
  ]);
  const [activeSourceId, setActiveSourceId] = useState(1);

  return (
    <SessionContext.Provider
      value={{ sources, setSources, activeSourceId, setActiveSourceId }}
    >
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  return useContext(SessionContext);
}
