import { useState } from "react";
import { ChevronDown, ChevronRight, Zap } from "lucide-react";

interface Props {
  trace: string[];
}

export default function AgentTrace({ trace }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mt-4">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-[11px] text-muted-foreground/60 hover:text-muted-foreground transition-colors"
      >
        <Zap className="h-3 w-3" />
        <span className="font-medium">Pipeline trace</span>
        <span className="text-[10px]">({trace.length} steps)</span>
        {expanded ? (
          <ChevronDown className="h-3 w-3" />
        ) : (
          <ChevronRight className="h-3 w-3" />
        )}
      </button>

      {expanded && (
        <div className="mt-3 ml-1 animate-slide-up">
          <div className="flex flex-wrap gap-x-1 gap-y-2 items-center">
            {trace.map((step, i) => (
              <div key={i} className="flex items-center gap-1">
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-primary/[0.06] border border-primary/10">
                  <div className="w-1.5 h-1.5 rounded-full bg-primary/50" />
                  <span className="text-[10px] font-medium text-foreground/70">
                    {step}
                  </span>
                </div>
                {i < trace.length - 1 && (
                  <div className="w-3 h-px bg-border" />
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
