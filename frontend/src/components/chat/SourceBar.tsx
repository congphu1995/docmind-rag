import { FileText } from "lucide-react";
import type { ChatMeta } from "@/api/chat";

type Source = ChatMeta["sources"][number];

interface Props {
  sources: Source[];
}

export default function SourceBar({ sources }: Props) {
  return (
    <div className="mt-3 flex items-center gap-2 flex-wrap">
      <span className="text-[10px] font-medium text-muted-foreground/60 uppercase tracking-wider">
        {sources.length} source{sources.length !== 1 ? "s" : ""}
      </span>
      <div className="flex items-center gap-1.5 flex-wrap">
        {sources.map((source) => (
          <div
            key={source.source_num}
            className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-muted/40 border border-border/40 text-[10px] text-muted-foreground hover:bg-muted/60 transition-colors"
          >
            <FileText className="h-2.5 w-2.5" />
            <span className="font-medium text-foreground/70">
              {source.source_num}
            </span>
            <span className="truncate max-w-[100px]">
              {source.doc_name}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
