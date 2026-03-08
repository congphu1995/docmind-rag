import { useState, useEffect } from "react";
import { Loader2, Layers, ChevronDown } from "lucide-react";
import { useChunks } from "@/hooks/useChunks";
import { useDocumentStore } from "@/stores/documentStore";
import ChunkFilters from "@/components/chunks/ChunkFilters";
import ChunkTree from "@/components/chunks/ChunkTree";
import ChunkDetail from "@/components/chunks/ChunkDetail";

export default function Chunks() {
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const { documents, fetchDocuments } = useDocumentStore();
  const { chunks, isLoading } = useChunks(selectedDocId);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="max-w-7xl mx-auto w-full px-6 py-6 flex flex-col h-full gap-4">
        {/* Header */}
        <div className="animate-slide-up">
          <div className="flex items-center gap-3 mb-1">
            <Layers className="h-5 w-5 text-primary" />
            <h2 className="text-xl font-semibold">Chunk Viewer</h2>
          </div>
          <p className="text-sm text-muted-foreground ml-8">
            Inspect how documents were parsed and chunked
          </p>
        </div>

        {/* Document selector */}
        <div
          className="animate-slide-up"
          style={{ animationDelay: "50ms" }}
        >
          <div className="relative w-full max-w-sm">
            <select
              value={selectedDocId ?? ""}
              onChange={(e) =>
                setSelectedDocId(e.target.value || null)
              }
              className="w-full pl-4 pr-10 py-2.5 rounded-xl border border-border/50 bg-background/50 text-sm outline-none focus:border-primary/40 transition-colors appearance-none cursor-pointer"
            >
              <option value="">Select a document...</option>
              {documents
                .filter((d) => d.status === "ready")
                .map((doc) => (
                  <option key={doc.doc_id} value={doc.doc_id}>
                    {doc.doc_name}
                  </option>
                ))}
            </select>
            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
          </div>
        </div>

        {/* No document selected */}
        {!selectedDocId && (
          <div className="flex-1 flex flex-col items-center justify-center text-center animate-slide-up">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center mb-4">
              <Layers className="h-7 w-7 text-primary" />
            </div>
            <p className="text-sm text-muted-foreground max-w-xs">
              Select a document above to view its parsed chunks and
              parent-child structure
            </p>
          </div>
        )}

        {/* Document selected — show filters + split view */}
        {selectedDocId && (
          <>
            {/* Filters bar */}
            <div
              className="glass rounded-xl border border-border/50 p-3 animate-slide-up"
              style={{ animationDelay: "100ms" }}
            >
              <ChunkFilters />
            </div>

            {/* Loading state */}
            {isLoading && (
              <div className="flex-1 flex items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            )}

            {/* Split view */}
            {!isLoading && (
              <div
                className="flex-1 flex gap-4 min-h-0 animate-slide-up"
                style={{ animationDelay: "150ms" }}
              >
                {/* Left panel — Tree (40%) */}
                <div className="w-[40%] glass rounded-xl border border-border/50 overflow-y-auto">
                  <div className="sticky top-0 z-10 glass border-b border-border/30 px-4 py-2.5">
                    <span className="text-xs font-medium text-muted-foreground">
                      Parent chunks
                    </span>
                    <span className="ml-2 text-[11px] text-muted-foreground/60">
                      ({chunks.length})
                    </span>
                  </div>
                  <ChunkTree />
                </div>

                {/* Right panel — Detail (60%) */}
                <div className="w-[60%] glass rounded-xl border border-border/50 overflow-hidden">
                  <ChunkDetail />
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
