import { Loader2, RefreshCw } from "lucide-react";
import { useDocuments } from "@/hooks/useDocuments";
import UploadZone from "@/components/documents/UploadZone";
import DocCard from "@/components/documents/DocCard";

export default function Documents() {
  const { documents, isLoading, uploadProgress, upload, remove, refresh } =
    useDocuments();

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8 animate-slide-up">
          <div>
            <h2 className="text-xl font-semibold">Documents</h2>
            <p className="text-sm text-muted-foreground mt-0.5">
              {documents.length} document
              {documents.length !== 1 ? "s" : ""} in your knowledge base
            </p>
          </div>
          <button
            onClick={refresh}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
        </div>

        {/* Upload zone */}
        <div
          className="mb-8 animate-slide-up"
          style={{ animationDelay: "50ms" }}
        >
          <UploadZone onUpload={upload} />
        </div>

        {/* Upload progress */}
        {uploadProgress.size > 0 && (
          <div className="mb-6 space-y-2 animate-slide-up">
            {Array.from(uploadProgress.entries()).map(([name, status]) => (
              <div
                key={name}
                className="flex items-center justify-between px-4 py-3 rounded-xl border border-primary/20 bg-primary/[0.03]"
              >
                <span className="text-sm truncate font-medium">{name}</span>
                <div className="flex items-center gap-2 text-xs text-primary font-medium">
                  {status === "processing" && (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  )}
                  {status}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Document grid */}
        {isLoading && documents.length === 0 ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : documents.length === 0 ? (
          <div
            className="text-center py-20 animate-slide-up"
            style={{ animationDelay: "100ms" }}
          >
            <p className="text-sm text-muted-foreground">
              No documents yet. Upload your first document to get started.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {documents.map((doc, i) => (
              <DocCard
                key={doc.doc_id}
                doc={doc}
                onDelete={remove}
                index={i}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
