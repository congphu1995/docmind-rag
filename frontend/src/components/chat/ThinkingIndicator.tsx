export default function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-3 py-1 animate-slide-up">
      {/* Frosted pill with shimmer */}
      <div className="relative flex items-center gap-2.5 px-4 py-2.5 rounded-2xl bg-muted/40 backdrop-blur-sm border border-border/40 overflow-hidden">
        {/* Shimmer overlay */}
        <div
          className="absolute inset-0 animate-shimmer"
          style={{
            background:
              "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.08) 50%, transparent 100%)",
            backgroundSize: "200% 100%",
          }}
        />

        {/* Bouncing dots */}
        <div className="flex items-center gap-1">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-dot-bounce"
              style={{ animationDelay: `${i * 0.16}s` }}
            />
          ))}
        </div>

        <span className="text-xs text-muted-foreground relative">
          Thinking...
        </span>
      </div>
    </div>
  );
}
