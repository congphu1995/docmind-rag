import { useRef, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Send, Square, Trash2 } from "lucide-react";
import { useSSEChat } from "@/hooks/useSSEChat";
import MessageBubble from "@/components/chat/MessageBubble";
import LLMToggle from "@/components/chat/LLMToggle";

export default function Chat() {
  const { messages, isLoading, llm, setLLM, send, stop, clear } = useSSEChat();
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    send(input);
    setInput("");
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 bg-card">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold">Chat</h2>
          {messages.length > 0 && (
            <Button variant="ghost" size="sm" onClick={clear} className="h-7 text-xs">
              <Trash2 className="h-3 w-3 mr-1" />
              Clear
            </Button>
          )}
        </div>
        <LLMToggle value={llm} onChange={setLLM} />
      </div>
      <Separator />

      {/* Messages */}
      <ScrollArea className="flex-1 px-6 py-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <h3 className="text-lg font-medium text-muted-foreground mb-2">
              DocMind RAG
            </h3>
            <p className="text-sm text-muted-foreground max-w-sm">
              Upload documents and ask questions. Answers include citations and
              an agent trace showing how the pipeline processed your query.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </ScrollArea>

      {/* Input */}
      <Separator />
      <form onSubmit={handleSubmit} className="px-6 py-4 bg-card">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your documents..."
            disabled={isLoading}
            className="flex-1"
          />
          {isLoading ? (
            <Button type="button" variant="destructive" size="icon" onClick={stop}>
              <Square className="h-4 w-4" />
            </Button>
          ) : (
            <Button type="submit" size="icon" disabled={!input.trim()}>
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}
