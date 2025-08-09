"use client";

import React, { useRef } from "react";

/**
 * Lightweight draggable/resizable wrapper used when the react-rnd package
 * is unavailable. It supports dragging via an element with class "drag-title"
 * and exposes onDrag/onResize callbacks that receive updated geometry.
 */
export function DraggableWindow({
  id,
  zIndex,
  pos,
  size,
  collapsed,
  onFocus,
  onDrag,
  onResize,
  children,
}) {
  const ref = useRef(null);
  const height = collapsed ? 40 : size.h;

  const clamp = (value, min, max) => Math.max(min, Math.min(value, max));

  const startDrag = (e) => {
    const handle = e.target.closest(".drag-title");
    if (!handle) return;
    onFocus && onFocus();
    const startX = e.clientX;
    const startY = e.clientY;
    const startPos = { ...pos };
    const parentRect = ref.current?.parentElement?.getBoundingClientRect();

    const onMove = (ev) => {
      const dx = ev.clientX - startX;
      const dy = ev.clientY - startY;
      let x = startPos.x + dx;
      let y = startPos.y + dy;
      if (parentRect) {
        const maxX = parentRect.width - size.w;
        const maxY = parentRect.height - height;
        x = clamp(x, 0, maxX);
        y = clamp(y, 0, maxY);
      }
      onDrag({ x, y });
    };

    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  const startResize = (e, dir) => {
    e.stopPropagation();
    onFocus && onFocus();
    const startX = e.clientX;
    const startY = e.clientY;
    const start = { w: size.w, h: size.h, x: pos.x, y: pos.y };
    const parentRect = ref.current?.parentElement?.getBoundingClientRect();

    const onMove = (ev) => {
      const dx = ev.clientX - startX;
      const dy = ev.clientY - startY;
      let { w, h, x, y } = start;
      if (dir.includes("right")) w = start.w + dx;
      if (dir.includes("left")) {
        w = start.w - dx;
        x = start.x + dx;
      }
      if (dir.includes("bottom")) h = start.h + dy;
      if (dir.includes("top")) {
        h = start.h - dy;
        y = start.y + dy;
      }
      w = Math.max(320, w);
      h = Math.max(180, h);
      if (parentRect) {
        const maxX = parentRect.width - w;
        const maxY = parentRect.height - h;
        x = clamp(x, 0, maxX);
        y = clamp(y, 0, maxY);
      }
      onResize({ w, h }, { x, y });
    };

    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  const handleStyles = collapsed
    ? []
    : [
        {
          dir: "top",
          style: { top: -4, left: "50%", marginLeft: -4, cursor: "n-resize" },
        },
        {
          dir: "right",
          style: { right: -4, top: "50%", marginTop: -4, cursor: "e-resize" },
        },
        {
          dir: "bottom",
          style: {
            bottom: -4,
            left: "50%",
            marginLeft: -4,
            cursor: "s-resize",
          },
        },
        {
          dir: "left",
          style: { left: -4, top: "50%", marginTop: -4, cursor: "w-resize" },
        },
        { dir: "top-left", style: { top: -4, left: -4, cursor: "nw-resize" } },
        {
          dir: "top-right",
          style: { top: -4, right: -4, cursor: "ne-resize" },
        },
        {
          dir: "bottom-left",
          style: { bottom: -4, left: -4, cursor: "sw-resize" },
        },
        {
          dir: "bottom-right",
          style: { bottom: -4, right: -4, cursor: "se-resize" },
        },
      ];

  return (
    <div
      ref={ref}
      onMouseDown={startDrag}
      style={{
        position: "absolute",
        left: pos.x,
        top: pos.y,
        width: size.w,
        height: height,
        zIndex,
      }}
    >
      {children}
      {handleStyles.map((h) => (
        <div
          key={h.dir}
          onMouseDown={(e) => startResize(e, h.dir)}
          style={{
            position: "absolute",
            width: 8,
            height: 8,
            background: "transparent",
            ...h.style,
          }}
        />
      ))}
    </div>
  );
}

export default DraggableWindow;
