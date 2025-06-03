"use client";
import React from "react";
import { SessionProvider } from "./SessionContext";
import ExplorerContent from "./ExplorerContent";

export default function HomePage() {
  return (
    <SessionProvider>
      <ExplorerContent />
    </SessionProvider>
  );
}
