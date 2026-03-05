import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Trash2 } from "lucide-react";
import type { DocumentInfo } from "@/api/documents";

interface Props {
  doc: DocumentInfo;
  onDelete: (docId: string) => void;
}

export default function DocCard({ doc, onDelete }: Props) {
  const ext = doc.doc_name.split(".").pop()?.toUpperCase() || "FILE";

  return (
    <Card className="flex items-center justify-between px-4 py-3">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 bg-primary/10 rounded-lg flex items-center justify-center">
          <span className="text-xs font-semibold text-primary">{ext}</span>
        </div>
        <div>
          <p className="text-sm font-medium truncate max-w-[300px]">
            {doc.doc_name}
          </p>
          <Badge
            variant={doc.status === "ready" ? "default" : "secondary"}
            className="text-xs mt-0.5"
          >
            {doc.status === "ready" ? "Ready" : doc.status}
          </Badge>
        </div>
      </div>
      <Button
        variant="ghost"
        size="icon"
        onClick={() => onDelete(doc.doc_id)}
        className="text-muted-foreground hover:text-destructive"
      >
        <Trash2 className="h-4 w-4" />
      </Button>
    </Card>
  );
}
