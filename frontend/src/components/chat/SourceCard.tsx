import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import type { ChatMeta } from "@/api/chat";

type Source = ChatMeta["sources"][number];

interface Props {
  source: Source;
}

export default function SourceCard({ source }: Props) {
  const confidence = Math.round(source.score * 100);

  return (
    <Collapsible>
      <CollapsibleTrigger className="w-full text-left group">
        <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl border border-border/50 bg-muted/30 hover:bg-muted/50 transition-all">
          {/* Source number */}
          <div className="w-6 h-6 rounded-md bg-primary/10 flex items-center justify-center shrink-0">
            <span className="text-[10px] font-bold text-primary">
              {source.source_num}
            </span>
          </div>

          {/* Doc info */}
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium truncate">{source.doc_name}</p>
            <p className="text-[10px] text-muted-foreground">
              Page {source.page}
            </p>
          </div>

          {/* Confidence bar */}
          <div className="flex items-center gap-2 shrink-0">
            <div className="w-16 h-1.5 rounded-full bg-border/50 overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-all",
                  confidence >= 70
                    ? "bg-emerald-500"
                    : confidence >= 40
                      ? "bg-amber-500"
                      : "bg-red-400"
                )}
                style={{ width: `${confidence}%` }}
              />
            </div>
            <span className="text-[10px] font-mono text-muted-foreground w-7 text-right">
              {confidence}%
            </span>
          </div>
        </div>
      </CollapsibleTrigger>

      <CollapsibleContent>
        <div className="mt-1 ml-9 px-3 py-2.5 rounded-xl bg-muted/20 border border-border/30">
          {source.section && (
            <p className="text-[10px] font-medium text-primary/70 mb-1.5 uppercase tracking-wider">
              {source.section}
            </p>
          )}
          <p className="text-xs leading-relaxed text-muted-foreground">
            {source.content_preview}
          </p>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
