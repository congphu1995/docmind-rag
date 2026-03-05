import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import type { Message } from "@/stores/chatStore";
import SourceCard from "./SourceCard";
import AgentTrace from "./AgentTrace";

interface Props {
  message: Message;
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex mb-4", isUser ? "justify-end" : "justify-start")}>
      <Card
        className={cn(
          "max-w-[75%] px-4 py-3 shadow-sm",
          isUser
            ? "bg-primary text-primary-foreground rounded-2xl rounded-br-sm"
            : "bg-card rounded-2xl rounded-bl-sm"
        )}
      >
        {/* Message content */}
        <div className="text-sm leading-relaxed whitespace-pre-wrap">
          {message.content}
          {message.isStreaming && (
            <span className="inline-block w-1.5 h-4 bg-primary ml-0.5 animate-pulse" />
          )}
        </div>

        {/* Sources */}
        {!isUser && message.meta?.sources && message.meta.sources.length > 0 && (
          <>
            <Separator className="my-3" />
            <p className="text-xs font-medium text-muted-foreground mb-2">
              Sources
            </p>
            <div className="space-y-1.5">
              {message.meta.sources.map((source) => (
                <SourceCard key={source.source_num} source={source} />
              ))}
            </div>
          </>
        )}

        {/* Agent trace */}
        {!isUser &&
          message.meta?.agent_trace &&
          message.meta.agent_trace.length > 0 && (
            <AgentTrace trace={message.meta.agent_trace} />
          )}
      </Card>
    </div>
  );
}
