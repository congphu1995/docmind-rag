import { useRef, useEffect, useState } from "react";
import { Send, Square, Sparkles, BookOpen, MessageCircle } from "lucide-react";
import { useSSEChat } from "@/hooks/useSSEChat";
import MessageBubble from "@/components/chat/MessageBubble";
import LLMToggle from "@/components/chat/LLMToggle";

export default function Chat() {
  const { messages, isLoading, llm, setLLM, send, stop, clear } = useSSEChat();
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    send(input);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex flex-col h-full relative">
      {/* Scrollable messages area */}
      <div className="flex-1 overflow-y-auto scroll-smooth">
        <div className="max-w-3xl mx-auto px-6 py-8">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center min-h-[calc(100vh-16rem)] text-center animate-slide-up">
              {/* Floating icon */}
              <div className="relative mb-8">
                <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center animate-float">
                  <Sparkles className="h-9 w-9 text-primary" />
                </div>
                <div
                  className="absolute -bottom-1 -right-1 w-6 h-6 rounded-lg bg-accent/30 animate-float"
                  style={{ animationDelay: "0.5s" }}
                />
              </div>

              <h2 className="text-2xl font-semibold mb-3">
                What would you like to know?
              </h2>
              <p className="text-muted-foreground max-w-md leading-relaxed mb-8">
                Ask questions about your documents. Every answer includes
                citations and a pipeline trace showing how the answer was
                constructed.
              </p>

              {/* Suggested queries */}
              <div className="flex flex-wrap gap-2 justify-center max-w-lg">
                {[
                  { icon: BookOpen, text: "Summarize the key findings" },
                  { icon: MessageCircle, text: "What does the paper conclude?" },
                ].map(({ icon: Icon, text }) => (
                  <button
                    key={text}
                    onClick={() => {
                      setInput(text);
                      textareaRef.current?.focus();
                    }}
                    className="flex items-center gap-2 px-4 py-2 rounded-full border border-border/60 bg-background/50 text-sm text-muted-foreground hover:text-foreground hover:border-primary/30 hover:bg-primary/5 transition-all"
                  >
                    <Icon className="h-3.5 w-3.5" />
                    {text}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <MessageBubble key={msg.id} message={msg} index={i} />
              ))}
              <div ref={bottomRef} />
            </>
          )}
        </div>
      </div>

      {/* Floating input bar */}
      <div className="sticky bottom-0 pb-6 pt-2 pointer-events-none">
        <div className="max-w-3xl mx-auto px-6 pointer-events-auto">
          <form
            onSubmit={handleSubmit}
            className="glass rounded-2xl shadow-lg shadow-black/[0.05] border border-border/50 p-3 transition-shadow focus-within:shadow-xl focus-within:shadow-primary/[0.05]"
          >
            <div className="flex items-end gap-3">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about your documents..."
                disabled={isLoading}
                rows={1}
                className="flex-1 bg-transparent text-sm placeholder:text-muted-foreground/60 resize-none outline-none min-h-[36px] max-h-[160px] py-2 px-1"
              />
              <div className="flex items-center gap-2 shrink-0">
                <LLMToggle value={llm} onChange={setLLM} />
                {isLoading ? (
                  <button
                    type="button"
                    onClick={stop}
                    aria-label="Stop generation"
                    className="w-9 h-9 rounded-xl bg-destructive text-white flex items-center justify-center hover:bg-destructive/90 transition-colors"
                  >
                    <Square className="h-3.5 w-3.5" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
                    aria-label="Send message"
                    className="w-9 h-9 rounded-xl bg-primary text-primary-foreground flex items-center justify-center hover:bg-primary/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <Send className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            </div>

            {messages.length > 0 && (
              <div className="flex items-center justify-between mt-2 pt-2 border-t border-border/30">
                <span className="text-[11px] text-muted-foreground/50">
                  {messages.filter((m) => m.role === "user").length} messages
                </span>
                <button
                  type="button"
                  onClick={clear}
                  className="text-[11px] text-muted-foreground/50 hover:text-destructive transition-colors"
                >
                  Clear chat
                </button>
              </div>
            )}
          </form>
        </div>
      </div>
    </div>
  );
}
