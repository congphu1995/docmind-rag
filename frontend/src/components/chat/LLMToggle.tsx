import { useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import { ChevronDown, Check } from "lucide-react";

interface Props {
  value: "openai" | "claude";
  onChange: (v: "openai" | "claude") => void;
}

const models = [
  { key: "openai" as const, label: "GPT-4o", color: "bg-emerald-500" },
  { key: "claude" as const, label: "Claude", color: "bg-amber-500" },
];

export default function LLMToggle({ value, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const current = models.find((m) => m.key === value)!;

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className={cn(
          "flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors",
          "text-muted-foreground hover:text-foreground hover:bg-muted/50"
        )}
      >
        <div className={cn("w-2 h-2 rounded-full", current.color)} />
        {current.label}
        <ChevronDown className={cn("h-3 w-3 transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div className="absolute bottom-full right-0 mb-2 py-1 min-w-[140px] glass rounded-xl border border-border/50 shadow-lg animate-slide-up">
          {models.map((model) => (
            <button
              key={model.key}
              type="button"
              onClick={() => {
                onChange(model.key);
                setOpen(false);
              }}
              className="flex items-center gap-2 w-full px-3 py-2 text-xs hover:bg-muted/50 transition-colors"
            >
              <div className={cn("w-2 h-2 rounded-full", model.color)} />
              <span className="flex-1 text-left font-medium">{model.label}</span>
              {value === model.key && <Check className="h-3 w-3 text-primary" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
