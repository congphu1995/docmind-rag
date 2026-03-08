import { cn } from "@/lib/utils";
import { Trash2, FileText, FileSpreadsheet, File } from "lucide-react";
import type { DocumentInfo } from "@/api/documents";

interface Props {
  doc: DocumentInfo;
  onDelete: (docId: string) => void;
  index: number;
}

const typeConfig: Record<
  string,
  { icon: typeof FileText; gradient: string }
> = {
  PDF: { icon: FileText, gradient: "from-rose-500/10 to-orange-500/10" },
  DOCX: {
    icon: FileSpreadsheet,
    gradient: "from-blue-500/10 to-cyan-500/10",
  },
  TXT: { icon: File, gradient: "from-emerald-500/10 to-teal-500/10" },
  MD: { icon: File, gradient: "from-violet-500/10 to-purple-500/10" },
};

export default function DocCard({ doc, onDelete, index }: Props) {
  const ext = doc.doc_name.split(".").pop()?.toUpperCase() || "FILE";
  const config = typeConfig[ext] || {
    icon: File,
    gradient: "from-gray-500/10 to-gray-400/10",
  };
  const Icon = config.icon;

  return (
    <div
      className="group relative rounded-2xl border border-border/50 bg-background/60 hover:bg-background/80 hover:border-border hover:shadow-lg hover:shadow-black/[0.03] transition-all duration-300 hover:-translate-y-0.5 overflow-hidden animate-slide-up"
      style={{ animationDelay: `${Math.min(index * 60, 400)}ms` }}
    >
      {/* Gradient header */}
      <div
        className={cn(
          "h-24 bg-gradient-to-br flex items-center justify-center",
          config.gradient
        )}
      >
        <Icon className="h-10 w-10 text-foreground/20" strokeWidth={1.5} />
      </div>

      {/* Content */}
      <div className="p-4">
        <p className="text-sm font-medium truncate mb-2">{doc.doc_name}</p>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <div
              className={cn(
                "w-1.5 h-1.5 rounded-full",
                doc.status === "ready"
                  ? "bg-emerald-500"
                  : "bg-amber-500 animate-pulse"
              )}
            />
            <span className="text-[11px] text-muted-foreground capitalize">
              {doc.status}
            </span>
          </div>
          <span className="text-[10px] text-muted-foreground/60 font-medium uppercase">
            {ext}
          </span>
        </div>
      </div>

      {/* Delete button — revealed on hover */}
      <button
        onClick={() => onDelete(doc.doc_id)}
        aria-label={`Delete ${doc.doc_name}`}
        className="absolute top-2 right-2 w-7 h-7 rounded-lg bg-background/80 backdrop-blur-sm border border-border/50 flex items-center justify-center opacity-0 group-hover:opacity-100 focus-visible:opacity-100 transition-opacity text-muted-foreground hover:text-destructive hover:border-destructive/30"
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
