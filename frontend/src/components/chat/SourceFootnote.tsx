import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { cn } from "@/lib/utils";
import type { ChatMeta } from "@/api/chat";

type Source = ChatMeta["sources"][number];

interface Props {
  source: Source;
}

export default function SourceFootnote({ source }: Props) {
  const confidence = Math.round(source.score * 100);

  return (
    <HoverCard openDelay={200} closeDelay={100}>
      <HoverCardTrigger asChild>
        <button
          type="button"
          className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-primary/15 text-[9px] font-bold text-primary hover:bg-primary/25 transition-colors align-super -mt-1 mx-0.5 cursor-pointer"
          aria-label={`Source ${source.source_num}: ${source.doc_name}`}
        >
          {source.source_num}
        </button>
      </HoverCardTrigger>
      <HoverCardContent
        side="top"
        align="center"
        className="w-72 p-3"
      >
        <div className="space-y-2">
          {/* Header */}
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-md bg-primary/10 flex items-center justify-center shrink-0">
              <span className="text-[9px] font-bold text-primary">
                {source.source_num}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium truncate">{source.doc_name}</p>
              <p className="text-[10px] text-muted-foreground">
                Page {source.page}
                {source.section && ` · ${source.section}`}
              </p>
            </div>
          </div>

          {/* Content preview */}
          {source.content && (
            <p className="text-[11px] leading-relaxed text-muted-foreground line-clamp-3">
              {source.content}
            </p>
          )}

          {/* Confidence */}
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1 rounded-full bg-border/50 overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full",
                  confidence >= 70
                    ? "bg-emerald-500"
                    : confidence >= 40
                      ? "bg-amber-500"
                      : "bg-red-400"
                )}
                style={{ width: `${confidence}%` }}
              />
            </div>
            <span className="text-[10px] font-mono text-muted-foreground">
              {confidence}%
            </span>
          </div>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}
