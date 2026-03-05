import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import type { ChatMeta } from "@/api/chat";

type Source = ChatMeta["sources"][number];

interface Props {
  source: Source;
}

export default function SourceCard({ source }: Props) {
  const confidence = Math.round(source.score * 100);

  return (
    <Collapsible>
      <CollapsibleTrigger className="w-full text-left bg-muted hover:bg-muted/80 rounded-lg px-3 py-2 transition-colors">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="font-mono text-xs">
              [{source.source_num}]
            </Badge>
            <span className="text-xs font-medium truncate max-w-[180px]">
              {source.doc_name}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">
              p.{source.page}
            </span>
            <Badge
              variant={
                confidence >= 70
                  ? "default"
                  : confidence >= 40
                    ? "secondary"
                    : "destructive"
              }
              className="text-xs"
            >
              {confidence}%
            </Badge>
          </div>
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="px-3 pb-2">
          <Separator className="my-2" />
          {source.section && (
            <p className="text-xs text-muted-foreground mb-1">
              {source.section}
            </p>
          )}
          <p className="text-xs leading-relaxed">{source.content_preview}</p>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
