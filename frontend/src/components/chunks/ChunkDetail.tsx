import { useState } from "react";
import { cn } from "@/lib/utils";
import { useChunkStore } from "@/stores/chunkStore";
import { FileText, Hash, Globe, Type, BookOpen, Users } from "lucide-react";

type Tab = "raw" | "markdown" | "html";

export default function ChunkDetail() {
  const { selectedChunk } = useChunkStore();
  const [activeTab, setActiveTab] = useState<Tab>("raw");

  if (!selectedChunk) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center px-6">
        <div className="w-14 h-14 rounded-2xl bg-muted/50 flex items-center justify-center mb-4">
          <FileText className="h-6 w-6 text-muted-foreground" />
        </div>
        <p className="text-sm text-muted-foreground">
          Select a chunk from the tree to view its details
        </p>
      </div>
    );
  }

  const chunk = selectedChunk;

  // Determine available tabs
  const tabs: { key: Tab; label: string }[] = [{ key: "raw", label: "Raw" }];
  if (chunk.content_markdown) tabs.push({ key: "markdown", label: "Markdown" });
  if (chunk.content_html) tabs.push({ key: "html", label: "HTML" });

  // Ensure active tab is valid
  const validTab = tabs.find((t) => t.key === activeTab) ? activeTab : "raw";

  const metadata = [
    { icon: Type, label: "Type", value: chunk.type },
    { icon: Hash, label: "Page", value: String(chunk.page) },
    { icon: BookOpen, label: "Section", value: chunk.section || "--" },
    { icon: Globe, label: "Language", value: chunk.language || "--" },
    {
      icon: FileText,
      label: "Word count",
      value: String(chunk.word_count),
    },
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Header with metadata */}
      <div className="p-4 border-b border-border/30 space-y-3">
        {/* Title row */}
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold truncate flex-1">
            {chunk.section || "Untitled chunk"}
          </h3>
          {chunk.children.length > 0 && (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-md bg-primary/10 text-primary text-[11px] font-medium shrink-0">
              <Users className="h-3 w-3" />
              {chunk.children.length} children
            </span>
          )}
        </div>

        {/* Metadata grid */}
        <div className="flex flex-wrap gap-x-5 gap-y-1.5">
          {metadata.map(({ icon: Icon, label, value }) => (
            <div key={label} className="flex items-center gap-1.5">
              <Icon className="h-3 w-3 text-muted-foreground" />
              <span className="text-[11px] text-muted-foreground">
                {label}:
              </span>
              <span className="text-[11px] font-medium">{value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Tabs */}
      {tabs.length > 1 && (
        <div className="flex items-center gap-1 px-4 pt-3">
          {tabs.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={cn(
                "px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-150",
                validTab === key
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/50",
              )}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {validTab === "raw" && (
          <pre className="text-xs leading-relaxed whitespace-pre-wrap font-mono text-foreground/90 bg-muted/30 rounded-xl p-4">
            {chunk.content_raw}
          </pre>
        )}
        {validTab === "markdown" && chunk.content_markdown && (
          <pre className="text-xs leading-relaxed whitespace-pre-wrap font-mono text-foreground/90 bg-muted/30 rounded-xl p-4">
            {chunk.content_markdown}
          </pre>
        )}
        {validTab === "html" && chunk.content_html && (
          <div
            className="prose prose-sm dark:prose-invert max-w-none text-sm [&_table]:border-collapse [&_td]:border [&_td]:border-border/50 [&_td]:px-2 [&_td]:py-1 [&_th]:border [&_th]:border-border/50 [&_th]:px-2 [&_th]:py-1 [&_th]:bg-muted/30 [&_th]:font-semibold [&_table]:text-xs"
            dangerouslySetInnerHTML={{ __html: chunk.content_html }}
          />
        )}
      </div>
    </div>
  );
}
