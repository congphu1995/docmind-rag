import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, Loader2 } from "lucide-react";
import { useDocuments } from "@/hooks/useDocuments";
import UploadZone from "@/components/documents/UploadZone";
import DocCard from "@/components/documents/DocCard";

export default function Documents() {
  const { documents, isLoading, uploadProgress, upload, remove, refresh } =
    useDocuments();

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 bg-card">
        <h2 className="text-sm font-semibold">Documents</h2>
        <Button variant="ghost" size="sm" onClick={refresh} className="h-7 text-xs">
          <RefreshCw className="h-3 w-3 mr-1" />
          Refresh
        </Button>
      </div>
      <Separator />

      <ScrollArea className="flex-1 px-6 py-4">
        <div className="space-y-6">
          {/* Upload zone */}
          <UploadZone onUpload={upload} />

          {/* Upload progress */}
          {uploadProgress.size > 0 && (
            <div className="space-y-2">
              {Array.from(uploadProgress.entries()).map(([name, status]) => (
                <Card
                  key={name}
                  className="flex items-center justify-between px-4 py-2 bg-primary/5"
                >
                  <span className="text-sm truncate">{name}</span>
                  <Badge variant="secondary" className="text-xs gap-1">
                    {status === "processing" && (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    )}
                    {status}
                  </Badge>
                </Card>
              ))}
            </div>
          )}

          {/* Document list */}
          {isLoading && documents.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">
              Loading...
            </p>
          ) : documents.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">
              No documents uploaded yet.
            </p>
          ) : (
            <div className="space-y-2">
              {documents.map((doc) => (
                <DocCard key={doc.doc_id} doc={doc} onDelete={remove} />
              ))}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
