"use client";
import React, { useEffect } from "react";
import { useParams } from "next/navigation";
import { SessionProvider, useSession } from "../SessionContext";
import ExplorerContent from "../ExplorerContent";

function SessionLoader({ sessionId }) {
  const { setSources, setActiveSourceId } = useSession();

  useEffect(() => {
    if (!sessionId) return;
    fetch(`/api/sessions/${sessionId}`)
      .then((res) => {
        if (!res.ok) {
          throw new Error("No such session");
        }
        return res.json();
      })
      .then((data) => {
        if (data.sources) {
          setSources(data.sources);
          setActiveSourceId(data.activeSourceId);
        }
      })
      .catch((err) => {
        console.error("Could not load session:", err);
      });
  }, [sessionId, setSources, setActiveSourceId]);

  return null;
}

export default function SessionPage() {
  const { sessionId } = useParams();

  return (
    <SessionProvider>
      <SessionLoader sessionId={sessionId} />
      <ExplorerContent />
    </SessionProvider>
  );
}
