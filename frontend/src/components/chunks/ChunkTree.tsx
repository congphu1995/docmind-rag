import { useState } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { useChunkStore } from "@/stores/chunkStore";
import type { ChunkNode, ChunkChild } from "@/api/chunks";

const TYPE_COLORS: Record<string, string> = {
  text: "bg-slate-500/15 text-slate-600 dark:text-slate-400",
  table: "bg-blue-500/15 text-blue-600 dark:text-blue-400",
  figure: "bg-purple-500/15 text-purple-600 dark:text-purple-400",
  picture: "bg-purple-500/15 text-purple-600 dark:text-purple-400",
  code: "bg-green-500/15 text-green-600 dark:text-green-400",
};

function TypeBadge({ type }: { type: string }) {
  const color = TYPE_COLORS[type] ?? "bg-muted text-muted-foreground";
  return (
    <span
      className={cn(
        "px-2 py-0.5 rounded-md text-[10px] font-semibold uppercase tracking-wide shrink-0",
        color,
      )}
    >
      {type}
    </span>
  );
}

function ParentRow({ node }: { node: ChunkNode }) {
  const [expanded, setExpanded] = useState(false);
  const { selectedChunk, selectChunk } = useChunkStore();
  const isSelected = selectedChunk?.chunk_id === node.chunk_id;

  return (
    <div>
      {/* Parent row */}
      <button
        onClick={() => {
          selectChunk(node);
          setExpanded((prev) => !prev);
        }}
        className={cn(
          "w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-left transition-all duration-150",
          isSelected
            ? "bg-primary/10 border border-primary/30"
            : "hover:bg-muted/50 border border-transparent",
        )}
      >
        {node.children.length > 0 ? (
          expanded ? (
            <ChevronDown className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          )
        ) : (
          <span className="w-3.5 shrink-0" />
        )}

        <TypeBadge type={node.type} />

        <span className="text-sm truncate flex-1 font-medium">
          {node.section || "Untitled section"}
        </span>

        <span className="text-[11px] text-muted-foreground shrink-0">
          p.{node.page}
        </span>
        <span className="text-[11px] text-muted-foreground shrink-0">
          {node.word_count}w
        </span>
      </button>

      {/* Children */}
      {expanded && node.children.length > 0 && (
        <div className="ml-6 mt-0.5 space-y-0.5">
          {node.children.map((child) => (
            <ChildRow key={child.chunk_id} child={child} />
          ))}
        </div>
      )}
    </div>
  );
}

function ChildRow({ child }: { child: ChunkChild }) {
  const { selectedChunk, selectChunk } = useChunkStore();
  const isSelected = selectedChunk?.chunk_id === child.chunk_id;

  // For children we create a minimal ChunkNode-compatible object for selectChunk
  const asNode: ChunkNode = {
    chunk_id: child.chunk_id,
    content_raw: child.content_raw,
    content_markdown: null,
    content_html: null,
    type: child.type,
    page: child.page,
    section: child.section,
    language: "",
    word_count: child.content.split(/\s+/).length,
    children: [],
  };

  return (
    <button
      onClick={() => selectChunk(asNode)}
      className={cn(
        "w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all duration-150",
        isSelected
          ? "bg-primary/10 border border-primary/30"
          : "hover:bg-muted/30 border border-transparent",
      )}
    >
      <TypeBadge type={child.type} />
      <span className="text-xs text-muted-foreground truncate flex-1">
        {child.content.slice(0, 100)}
        {child.content.length > 100 ? "..." : ""}
      </span>
      <span className="text-[10px] text-muted-foreground shrink-0">
        p.{child.page}
      </span>
    </button>
  );
}

export default function ChunkTree() {
  const { chunks } = useChunkStore();

  if (chunks.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
        No chunks to display
      </div>
    );
  }

  return (
    <div className="space-y-0.5 p-2">
      {chunks.map((node) => (
        <ParentRow key={node.chunk_id} node={node} />
      ))}
    </div>
  );
}
