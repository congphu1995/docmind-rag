import React from "react";
import { cn } from "@/lib/utils";
import type { Message } from "@/stores/chatStore";
import type { ChatMeta } from "@/api/chat";
import ThinkingIndicator from "./ThinkingIndicator";
import SourceFootnote from "./SourceFootnote";
import SourceBar from "./SourceBar";
import AgentTrace from "./AgentTrace";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  message: Message;
  index: number;
}

type Source = ChatMeta["sources"][number];

/**
 * Walk react-markdown children, find string segments containing [N],
 * and replace them with SourceFootnote chips.
 */
function processFootnotes(
  children: React.ReactNode,
  sources: Source[]
): React.ReactNode {
  return React.Children.map(children, (child) => {
    if (typeof child === "string") {
      const parts = child.split(/\[(\d+)\]/g);
      if (parts.length === 1) return child;
      return parts.map((part, i) => {
        if (i % 2 === 1) {
          const num = parseInt(part, 10);
          const source = sources.find((s) => s.source_num === num);
          if (source) return <SourceFootnote key={`fn-${num}-${i}`} source={source} />;
          return `[${part}]`;
        }
        return part || null;
      });
    }
    if (React.isValidElement<{ children?: React.ReactNode }>(child) && child.props.children) {
      return React.cloneElement(child, {}, processFootnotes(child.props.children, sources));
    }
    return child;
  });
}

/**
 * Build react-markdown component overrides that inject footnote chips
 * into rendered paragraph, list-item, and table-cell elements.
 */
function useFootnoteComponents(sources: Source[]) {
  if (!sources.length) return {};
  return {
    p: ({ children, ...props }: React.ComponentProps<"p">) => (
      <p {...props}>{processFootnotes(children, sources)}</p>
    ),
    li: ({ children, ...props }: React.ComponentProps<"li">) => (
      <li {...props}>{processFootnotes(children, sources)}</li>
    ),
    td: ({ children, ...props }: React.ComponentProps<"td">) => (
      <td {...props}>{processFootnotes(children, sources)}</td>
    ),
  };
}

export default function MessageBubble({ message, index }: Props) {
  const isUser = message.role === "user";
  const sources = message.meta?.sources ?? [];
  const footnoteComponents = useFootnoteComponents(sources);
  const isThinking = message.isStreaming && message.content === "";

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
            {isThinking ? (
              <ThinkingIndicator />
            ) : (
              <div className="prose-chat text-sm leading-relaxed">
                <Markdown
                  remarkPlugins={[remarkGfm]}
                  components={footnoteComponents}
                >
                  {message.content}
                </Markdown>
                {message.isStreaming && (
                  <span className="inline-block w-0.5 h-4 bg-primary/70 ml-0.5 animate-blink rounded-full" />
                )}
              </div>
            )}

            {/* Source bar */}
            {!isThinking && sources.length > 0 && (
              <SourceBar sources={sources} />
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
