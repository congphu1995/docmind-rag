import { cn } from "@/lib/utils";
import type { Message } from "@/stores/chatStore";
import SourceCard from "./SourceCard";
import AgentTrace from "./AgentTrace";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  message: Message;
  index: number;
}

export default function MessageBubble({ message, index }: Props) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn("mb-6 animate-slide-up", isUser && "flex justify-end")}
      style={{ animationDelay: `${Math.min(index * 50, 300)}ms` }}
    >
      {isUser ? (
        <div className="max-w-[80%] px-4 py-2.5 rounded-2xl rounded-br-md bg-primary text-primary-foreground text-sm leading-relaxed">
          {message.content}
        </div>
      ) : (
        <div className="flex gap-3">
          {/* Avatar */}
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-primary/15 to-accent/15 flex items-center justify-center shrink-0 mt-0.5">
            <span className="text-xs font-semibold text-primary">D</span>
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="prose-chat text-sm leading-relaxed">
              <Markdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </Markdown>
              {message.isStreaming && (
                <span className="inline-block w-2 h-4 bg-primary/60 ml-0.5 animate-pulse rounded-sm" />
              )}
            </div>

            {/* Sources */}
            {message.meta?.sources && message.meta.sources.length > 0 && (
              <div className="mt-4 space-y-2">
                <p className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
                  Sources
                </p>
                <div className="grid gap-2">
                  {message.meta.sources.map((source) => (
                    <SourceCard key={source.source_num} source={source} />
                  ))}
                </div>
              </div>
            )}

            {/* Agent trace */}
            {message.meta?.agent_trace &&
              message.meta.agent_trace.length > 0 && (
                <AgentTrace trace={message.meta.agent_trace} />
              )}
          </div>
        </div>
      )}
    </div>
  );
}
