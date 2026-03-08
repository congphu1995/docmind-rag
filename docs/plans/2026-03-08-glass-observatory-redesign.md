# Glass Observatory Frontend Redesign

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform DocMind RAG frontend from vanilla shadcn boilerplate into a luminous "Glass Observatory" interface with frosted-glass panels, top navigation, floating chat input, markdown rendering, pipeline stepper, grid document layout, and dark mode toggle.

**Architecture:** Replace the sidebar layout with a frosted-glass top nav bar. Chat gets a centered conversation thread (max-w-3xl) with a floating input bar at the bottom. Documents use a responsive grid of cards with file-type gradients. All panels use a `.glass` utility (semi-transparent background + backdrop-blur). CSS-only animations (no new animation library). `react-markdown` + `remark-gfm` for rich assistant message rendering.

**Tech Stack:** React 19, Tailwind CSS 4, shadcn/ui (radix), react-markdown, remark-gfm, Zustand, Lucide icons, Outfit font (Google Fonts)

**Design System:**
| Token | Light | Dark |
|-------|-------|------|
| Background | warm off-white `oklch(0.975 0.003 80)` | deep navy `oklch(0.14 0.02 265)` |
| Primary | violet-blue `oklch(0.50 0.22 270)` | lighter violet `oklch(0.65 0.22 270)` |
| Accent | warm peach `oklch(0.85 0.10 55)` | `oklch(0.75 0.10 55)` |
| Glass | `rgba(255,255,255,0.6)` + `blur(20px)` | `rgba(20,20,40,0.6)` + `blur(20px)` |
| Border radius | 20px cards, 14px buttons, 9999px pills | same |
| Font | Outfit (300–800) from Google Fonts | same |

**Dependency Graph:**
```
Task 1 (foundation) → Task 2 (app shell) → ┬─ Task 3 (chat + toggle)
                                             ├─ Task 4 (message components)  [parallel]
                                             └─ Task 5 (documents)           [parallel]
                                           → Task 6 (build verify + commit)
```

---

### Task 1: Foundation — Font, Theme & Dependencies

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/src/index.css`

**Step 1: Verify react-markdown is installed**

Run: `cd frontend && npm ls react-markdown remark-gfm 2>/dev/null | head -5`
Expected: both packages listed. If not, run: `npm install react-markdown remark-gfm`

**Step 2: Add Outfit font to `frontend/index.html`**

Replace entire file with:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>DocMind</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap"
      rel="stylesheet"
    />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

**Step 3: Rewrite `frontend/src/index.css`**

Replace entire file with:

```css
@import "tailwindcss";
@import "tw-animate-css";
@import "shadcn/tailwind.css";

@custom-variant dark (&:is(.dark *));

@theme inline {
    --radius-sm: calc(var(--radius) - 4px);
    --radius-md: calc(var(--radius) - 2px);
    --radius-lg: var(--radius);
    --radius-xl: calc(var(--radius) + 4px);
    --radius-2xl: calc(var(--radius) + 8px);
    --radius-3xl: calc(var(--radius) + 12px);
    --radius-4xl: calc(var(--radius) + 16px);
    --font-sans: 'Outfit', ui-sans-serif, system-ui, -apple-system, sans-serif;
    --color-background: var(--background);
    --color-foreground: var(--foreground);
    --color-card: var(--card);
    --color-card-foreground: var(--card-foreground);
    --color-popover: var(--popover);
    --color-popover-foreground: var(--popover-foreground);
    --color-primary: var(--primary);
    --color-primary-foreground: var(--primary-foreground);
    --color-secondary: var(--secondary);
    --color-secondary-foreground: var(--secondary-foreground);
    --color-muted: var(--muted);
    --color-muted-foreground: var(--muted-foreground);
    --color-accent: var(--accent);
    --color-accent-foreground: var(--accent-foreground);
    --color-destructive: var(--destructive);
    --color-border: var(--border);
    --color-input: var(--input);
    --color-ring: var(--ring);
    --color-chart-1: var(--chart-1);
    --color-chart-2: var(--chart-2);
    --color-chart-3: var(--chart-3);
    --color-chart-4: var(--chart-4);
    --color-chart-5: var(--chart-5);
    --color-sidebar: var(--sidebar);
    --color-sidebar-foreground: var(--sidebar-foreground);
    --color-sidebar-primary: var(--sidebar-primary);
    --color-sidebar-primary-foreground: var(--sidebar-primary-foreground);
    --color-sidebar-accent: var(--sidebar-accent);
    --color-sidebar-accent-foreground: var(--sidebar-accent-foreground);
    --color-sidebar-border: var(--sidebar-border);
    --color-sidebar-ring: var(--sidebar-ring);
    --animate-slide-up: slide-up 0.4s ease-out both;
    --animate-float: float 4s ease-in-out infinite;
}

:root {
    --radius: 0.75rem;
    --background: oklch(0.975 0.003 80);
    --foreground: oklch(0.18 0.02 265);
    --card: oklch(0.99 0.002 80);
    --card-foreground: oklch(0.18 0.02 265);
    --popover: oklch(0.99 0.002 80);
    --popover-foreground: oklch(0.18 0.02 265);
    --primary: oklch(0.50 0.22 270);
    --primary-foreground: oklch(0.98 0 0);
    --secondary: oklch(0.94 0.04 60);
    --secondary-foreground: oklch(0.25 0.02 265);
    --muted: oklch(0.95 0.003 80);
    --muted-foreground: oklch(0.50 0.02 265);
    --accent: oklch(0.85 0.10 55);
    --accent-foreground: oklch(0.25 0.02 265);
    --destructive: oklch(0.58 0.22 27);
    --border: oklch(0.91 0.003 80);
    --input: oklch(0.91 0.003 80);
    --ring: oklch(0.50 0.22 270);
    --chart-1: oklch(0.50 0.22 270);
    --chart-2: oklch(0.65 0.15 160);
    --chart-3: oklch(0.75 0.15 55);
    --chart-4: oklch(0.60 0.20 305);
    --chart-5: oklch(0.65 0.20 15);
    --sidebar: oklch(0.98 0.002 80);
    --sidebar-foreground: oklch(0.18 0.02 265);
    --sidebar-primary: oklch(0.50 0.22 270);
    --sidebar-primary-foreground: oklch(0.98 0 0);
    --sidebar-accent: oklch(0.94 0.04 60);
    --sidebar-accent-foreground: oklch(0.25 0.02 265);
    --sidebar-border: oklch(0.91 0.003 80);
    --sidebar-ring: oklch(0.50 0.22 270);
}

.dark {
    --background: oklch(0.14 0.02 265);
    --foreground: oklch(0.93 0.005 80);
    --card: oklch(0.19 0.02 265);
    --card-foreground: oklch(0.93 0.005 80);
    --popover: oklch(0.19 0.02 265);
    --popover-foreground: oklch(0.93 0.005 80);
    --primary: oklch(0.65 0.22 270);
    --primary-foreground: oklch(0.98 0 0);
    --secondary: oklch(0.25 0.015 265);
    --secondary-foreground: oklch(0.93 0.005 80);
    --muted: oklch(0.22 0.015 265);
    --muted-foreground: oklch(0.60 0.01 265);
    --accent: oklch(0.75 0.10 55);
    --accent-foreground: oklch(0.93 0.005 80);
    --destructive: oklch(0.65 0.22 27);
    --border: oklch(0.28 0.015 265);
    --input: oklch(0.28 0.015 265);
    --ring: oklch(0.65 0.22 270);
    --chart-1: oklch(0.65 0.22 270);
    --chart-2: oklch(0.70 0.15 160);
    --chart-3: oklch(0.80 0.12 55);
    --chart-4: oklch(0.65 0.20 305);
    --chart-5: oklch(0.70 0.20 15);
    --sidebar: oklch(0.19 0.02 265);
    --sidebar-foreground: oklch(0.93 0.005 80);
    --sidebar-primary: oklch(0.65 0.22 270);
    --sidebar-primary-foreground: oklch(0.98 0 0);
    --sidebar-accent: oklch(0.25 0.015 265);
    --sidebar-accent-foreground: oklch(0.93 0.005 80);
    --sidebar-border: oklch(0.28 0.015 265);
    --sidebar-ring: oklch(0.65 0.22 270);
}

/* ---------- Keyframes ---------- */
@keyframes slide-up {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50%      { transform: translateY(-8px); }
}

/* ---------- Glass utility ---------- */
.glass {
  background: rgba(255, 255, 255, 0.6);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
}
.dark .glass {
  background: rgba(20, 20, 40, 0.6);
}

/* ---------- Markdown prose in chat ---------- */
.prose-chat p { margin-bottom: 0.75rem; }
.prose-chat p:last-child { margin-bottom: 0; }
.prose-chat h1,
.prose-chat h2,
.prose-chat h3 { font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem; }
.prose-chat h1 { font-size: 1.125rem; }
.prose-chat h2 { font-size: 1rem; }
.prose-chat h3 { font-size: 0.875rem; }
.prose-chat ul,
.prose-chat ol { margin-bottom: 0.75rem; padding-left: 1.5rem; }
.prose-chat ul { list-style-type: disc; }
.prose-chat ol { list-style-type: decimal; }
.prose-chat li { margin-bottom: 0.25rem; }
.prose-chat code {
  font-size: 0.8125rem;
  padding: 0.125rem 0.375rem;
  border-radius: 0.375rem;
  background: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.prose-chat pre {
  margin-bottom: 0.75rem;
  padding: 0.75rem 1rem;
  border-radius: 0.75rem;
  background: var(--muted);
  overflow-x: auto;
}
.prose-chat pre code {
  padding: 0;
  background: transparent;
}
.prose-chat blockquote {
  border-left: 3px solid var(--primary);
  padding-left: 1rem;
  margin-bottom: 0.75rem;
  color: var(--muted-foreground);
}
.prose-chat table {
  width: 100%;
  margin-bottom: 0.75rem;
  border-collapse: collapse;
}
.prose-chat th,
.prose-chat td {
  padding: 0.375rem 0.75rem;
  border: 1px solid var(--border);
  text-align: left;
  font-size: 0.8125rem;
}
.prose-chat th {
  background: var(--muted);
  font-weight: 600;
}
.prose-chat strong { font-weight: 600; }
.prose-chat a { color: var(--primary); text-decoration: underline; }

/* ---------- Custom scrollbar ---------- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted-foreground); }

/* ---------- Base ---------- */
@layer base {
  * {
    @apply border-border outline-ring/50;
  }
  body {
    @apply bg-background text-foreground font-sans antialiased;
  }
}
```

**Step 4: Verify compilation**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -20`
Expected: No errors.

**Step 5: Commit**

```bash
git add frontend/index.html frontend/src/index.css frontend/package.json frontend/package-lock.json
git commit -m "feat(frontend): add Glass Observatory theme — Outfit font, violet-blue palette, glass utilities, animations"
```

---

### Task 2: App Shell — Top Navigation Bar

**Files:**
- Modify: `frontend/src/App.tsx`

**Step 1: Rewrite `frontend/src/App.tsx`**

Replace entire file with:

```tsx
import { useState } from "react";
import { cn } from "@/lib/utils";
import { MessageSquare, FileText, Moon, Sun } from "lucide-react";
import Chat from "./pages/Chat";
import Documents from "./pages/Documents";

type Page = "chat" | "documents";

export default function App() {
  const [page, setPage] = useState<Page>("chat");
  const [dark, setDark] = useState(false);

  const toggleDark = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
  };

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-300">
      {/* Decorative background orbs */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute -top-[300px] -right-[200px] w-[700px] h-[700px] rounded-full bg-primary/[0.04] blur-[100px]" />
        <div className="absolute -bottom-[200px] -left-[200px] w-[600px] h-[600px] rounded-full bg-accent/[0.08] blur-[100px]" />
      </div>

      {/* Top nav bar */}
      <nav className="fixed top-0 inset-x-0 z-50 h-16 glass border-b border-border/50">
        <div className="max-w-7xl mx-auto px-6 h-full flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-primary/70 flex items-center justify-center shadow-md shadow-primary/20">
              <span className="text-sm font-bold text-primary-foreground">D</span>
            </div>
            <div className="leading-tight">
              <span className="text-base font-semibold tracking-tight block">DocMind</span>
              <span className="text-[10px] text-muted-foreground tracking-widest uppercase block -mt-0.5">
                Intelligence
              </span>
            </div>
          </div>

          {/* Nav pills */}
          <div className="flex items-center gap-1 rounded-full p-1 bg-muted/50 border border-border/50">
            {(
              [
                { key: "chat", label: "Chat", icon: MessageSquare },
                { key: "documents", label: "Documents", icon: FileText },
              ] as const
            ).map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setPage(key)}
                className={cn(
                  "flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200",
                  page === key
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <Icon className="h-4 w-4" />
                {label}
              </button>
            ))}
          </div>

          {/* Dark mode toggle */}
          <button
            onClick={toggleDark}
            className="w-9 h-9 rounded-xl flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
          >
            {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </nav>

      {/* Main content */}
      <main className="pt-16 h-screen">
        {page === "chat" && <Chat />}
        {page === "documents" && <Documents />}
      </main>
    </div>
  );
}
```

**Step 2: Verify compilation**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -20`
Expected: No errors.

**Step 3: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat(frontend): replace sidebar with frosted-glass top nav bar + dark mode toggle"
```

---

### Task 3: Chat Experience — Page Layout + Floating Input + LLM Toggle

**Parallel group:** B (can run simultaneously with Tasks 4 and 5 after Task 2)

**Files:**
- Modify: `frontend/src/pages/Chat.tsx`
- Modify: `frontend/src/components/chat/LLMToggle.tsx`

**Step 1: Rewrite `frontend/src/components/chat/LLMToggle.tsx`**

The LLM toggle becomes a compact dropdown that opens upward, designed to sit inside the floating input bar.

Replace entire file with:

```tsx
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
```

**Step 2: Rewrite `frontend/src/pages/Chat.tsx`**

Features: centered conversation (max-w-3xl), beautiful empty state with gradient icon and suggested queries, floating glass input bar with integrated LLM selector, textarea (supports Shift+Enter for newlines, Enter to send), auto-resize textarea, message counter + clear button inside input bar.

Replace entire file with:

```tsx
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
                    className="w-9 h-9 rounded-xl bg-destructive text-white flex items-center justify-center hover:bg-destructive/90 transition-colors"
                  >
                    <Square className="h-3.5 w-3.5" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
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
```

**Step 3: Verify compilation**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -20`
Expected: May show errors about MessageBubble `index` prop (will be fixed in Task 4). Proceed.

**Step 4: Commit**

```bash
git add frontend/src/pages/Chat.tsx frontend/src/components/chat/LLMToggle.tsx
git commit -m "feat(frontend): floating glass chat input, beautiful empty state, LLM dropdown"
```

---

### Task 4: Message Components — Bubble, Sources & Pipeline Stepper

**Parallel group:** B (can run simultaneously with Tasks 3 and 5 after Task 2)

**Files:**
- Modify: `frontend/src/components/chat/MessageBubble.tsx`
- Modify: `frontend/src/components/chat/SourceCard.tsx`
- Modify: `frontend/src/components/chat/AgentTrace.tsx`

**Step 1: Rewrite `frontend/src/components/chat/MessageBubble.tsx`**

User messages: right-aligned violet pill. Assistant messages: left-aligned with avatar icon, full-width, markdown-rendered content via react-markdown, staggered entrance animation.

Replace entire file with:

```tsx
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
```

**Step 2: Rewrite `frontend/src/components/chat/SourceCard.tsx`**

Redesigned with a horizontal layout: source number badge, doc name + page, confidence bar (colored green/amber/red), and expandable preview.

Replace entire file with:

```tsx
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import type { ChatMeta } from "@/api/chat";

type Source = ChatMeta["sources"][number];

interface Props {
  source: Source;
}

export default function SourceCard({ source }: Props) {
  const confidence = Math.round(source.score * 100);

  return (
    <Collapsible>
      <CollapsibleTrigger className="w-full text-left group">
        <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl border border-border/50 bg-muted/30 hover:bg-muted/50 transition-all">
          {/* Source number */}
          <div className="w-6 h-6 rounded-md bg-primary/10 flex items-center justify-center shrink-0">
            <span className="text-[10px] font-bold text-primary">
              {source.source_num}
            </span>
          </div>

          {/* Doc info */}
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium truncate">{source.doc_name}</p>
            <p className="text-[10px] text-muted-foreground">
              Page {source.page}
            </p>
          </div>

          {/* Confidence bar */}
          <div className="flex items-center gap-2 shrink-0">
            <div className="w-16 h-1.5 rounded-full bg-border/50 overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-all",
                  confidence >= 70
                    ? "bg-emerald-500"
                    : confidence >= 40
                      ? "bg-amber-500"
                      : "bg-red-400"
                )}
                style={{ width: `${confidence}%` }}
              />
            </div>
            <span className="text-[10px] font-mono text-muted-foreground w-7 text-right">
              {confidence}%
            </span>
          </div>
        </div>
      </CollapsibleTrigger>

      <CollapsibleContent>
        <div className="mt-1 ml-9 px-3 py-2.5 rounded-xl bg-muted/20 border border-border/30">
          {source.section && (
            <p className="text-[10px] font-medium text-primary/70 mb-1.5 uppercase tracking-wider">
              {source.section}
            </p>
          )}
          <p className="text-xs leading-relaxed text-muted-foreground">
            {source.content_preview}
          </p>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
```

**Step 3: Rewrite `frontend/src/components/chat/AgentTrace.tsx`**

Replace the accordion with a horizontal pipeline stepper: each step is a pill with a dot connector, collapsible via a toggle button.

Replace entire file with:

```tsx
import { useState } from "react";
import { cn } from "@/lib/utils";
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
```

**Step 4: Verify compilation**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -20`
Expected: No errors.

**Step 5: Commit**

```bash
git add frontend/src/components/chat/MessageBubble.tsx frontend/src/components/chat/SourceCard.tsx frontend/src/components/chat/AgentTrace.tsx
git commit -m "feat(frontend): markdown message bubbles, confidence-bar sources, pipeline stepper trace"
```

---

### Task 5: Documents Experience — Grid Layout, Cards & Upload

**Parallel group:** B (can run simultaneously with Tasks 3 and 4 after Task 2)

**Files:**
- Modify: `frontend/src/pages/Documents.tsx`
- Modify: `frontend/src/components/documents/DocCard.tsx`
- Modify: `frontend/src/components/documents/UploadZone.tsx`

**Step 1: Rewrite `frontend/src/components/documents/UploadZone.tsx`**

Redesigned with rounded-2xl dashed border, centered icon + text, purple primary button, scale-up effect on drag.

Replace entire file with:

```tsx
import { useCallback, useState } from "react";
import { cn } from "@/lib/utils";
import { Upload, Plus } from "lucide-react";

interface Props {
  onUpload: (file: File) => void;
}

export default function UploadZone({ onUpload }: Props) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) onUpload(file);
    },
    [onUpload]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onUpload(file);
      e.target.value = "";
    },
    [onUpload]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={cn(
        "relative rounded-2xl border-2 border-dashed transition-all duration-300 overflow-hidden",
        isDragging
          ? "border-primary bg-primary/[0.03] scale-[1.01]"
          : "border-border/60 hover:border-primary/30 hover:bg-muted/20"
      )}
    >
      <div className="flex flex-col items-center justify-center py-10 px-6">
        <div
          className={cn(
            "w-12 h-12 rounded-2xl flex items-center justify-center mb-4 transition-colors",
            isDragging ? "bg-primary/10" : "bg-muted/50"
          )}
        >
          <Upload
            className={cn(
              "h-5 w-5 transition-colors",
              isDragging ? "text-primary" : "text-muted-foreground"
            )}
          />
        </div>
        <p className="text-sm font-medium mb-1">
          {isDragging ? "Drop to upload" : "Drop files here"}
        </p>
        <p className="text-xs text-muted-foreground mb-4">
          or browse from your computer
        </p>
        <label className="cursor-pointer inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors">
          <Plus className="h-3.5 w-3.5" />
          Choose file
          <input
            type="file"
            accept=".pdf,.docx,.txt,.md"
            onChange={handleFileSelect}
            className="hidden"
          />
        </label>
        <p className="text-[10px] text-muted-foreground/60 mt-3">
          PDF, DOCX, TXT, MD — up to 50 MB
        </p>
      </div>
    </div>
  );
}
```

**Step 2: Rewrite `frontend/src/components/documents/DocCard.tsx`**

Grid card with gradient header area (color varies by file type), file type icon, hover lift effect with shadow, delete button revealed on hover.

Replace entire file with:

```tsx
import { cn } from "@/lib/utils";
import { Trash2, FileText, FileSpreadsheet, File } from "lucide-react";
import type { DocumentInfo } from "@/api/documents";

interface Props {
  doc: DocumentInfo;
  onDelete: (docId: string) => void;
  index: number;
}

const typeConfig: Record<
  string,
  { icon: typeof FileText; gradient: string }
> = {
  PDF: { icon: FileText, gradient: "from-rose-500/10 to-orange-500/10" },
  DOCX: {
    icon: FileSpreadsheet,
    gradient: "from-blue-500/10 to-cyan-500/10",
  },
  TXT: { icon: File, gradient: "from-emerald-500/10 to-teal-500/10" },
  MD: { icon: File, gradient: "from-violet-500/10 to-purple-500/10" },
};

export default function DocCard({ doc, onDelete, index }: Props) {
  const ext = doc.doc_name.split(".").pop()?.toUpperCase() || "FILE";
  const config = typeConfig[ext] || {
    icon: File,
    gradient: "from-gray-500/10 to-gray-400/10",
  };
  const Icon = config.icon;

  return (
    <div
      className="group relative rounded-2xl border border-border/50 bg-background/60 hover:bg-background/80 hover:border-border hover:shadow-lg hover:shadow-black/[0.03] transition-all duration-300 hover:-translate-y-0.5 overflow-hidden animate-slide-up"
      style={{ animationDelay: `${Math.min(index * 60, 400)}ms` }}
    >
      {/* Gradient header */}
      <div
        className={cn(
          "h-24 bg-gradient-to-br flex items-center justify-center",
          config.gradient
        )}
      >
        <Icon className="h-10 w-10 text-foreground/20" strokeWidth={1.5} />
      </div>

      {/* Content */}
      <div className="p-4">
        <p className="text-sm font-medium truncate mb-2">{doc.doc_name}</p>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <div
              className={cn(
                "w-1.5 h-1.5 rounded-full",
                doc.status === "ready"
                  ? "bg-emerald-500"
                  : "bg-amber-500 animate-pulse"
              )}
            />
            <span className="text-[11px] text-muted-foreground capitalize">
              {doc.status}
            </span>
          </div>
          <span className="text-[10px] text-muted-foreground/60 font-medium uppercase">
            {ext}
          </span>
        </div>
      </div>

      {/* Delete button — revealed on hover */}
      <button
        onClick={() => onDelete(doc.doc_id)}
        className="absolute top-2 right-2 w-7 h-7 rounded-lg bg-background/80 backdrop-blur-sm border border-border/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive hover:border-destructive/30"
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
```

**Step 3: Rewrite `frontend/src/pages/Documents.tsx`**

Grid layout (1 / 2 / 3 columns responsive), header with doc count, upload zone, progress cards, staggered card entrance.

Replace entire file with:

```tsx
import { Loader2, RefreshCw } from "lucide-react";
import { useDocuments } from "@/hooks/useDocuments";
import UploadZone from "@/components/documents/UploadZone";
import DocCard from "@/components/documents/DocCard";

export default function Documents() {
  const { documents, isLoading, uploadProgress, upload, remove, refresh } =
    useDocuments();

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8 animate-slide-up">
          <div>
            <h2 className="text-xl font-semibold">Documents</h2>
            <p className="text-sm text-muted-foreground mt-0.5">
              {documents.length} document
              {documents.length !== 1 ? "s" : ""} in your knowledge base
            </p>
          </div>
          <button
            onClick={refresh}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
        </div>

        {/* Upload zone */}
        <div
          className="mb-8 animate-slide-up"
          style={{ animationDelay: "50ms" }}
        >
          <UploadZone onUpload={upload} />
        </div>

        {/* Upload progress */}
        {uploadProgress.size > 0 && (
          <div className="mb-6 space-y-2 animate-slide-up">
            {Array.from(uploadProgress.entries()).map(([name, status]) => (
              <div
                key={name}
                className="flex items-center justify-between px-4 py-3 rounded-xl border border-primary/20 bg-primary/[0.03]"
              >
                <span className="text-sm truncate font-medium">{name}</span>
                <div className="flex items-center gap-2 text-xs text-primary font-medium">
                  {status === "processing" && (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  )}
                  {status}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Document grid */}
        {isLoading && documents.length === 0 ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : documents.length === 0 ? (
          <div
            className="text-center py-20 animate-slide-up"
            style={{ animationDelay: "100ms" }}
          >
            <p className="text-sm text-muted-foreground">
              No documents yet. Upload your first document to get started.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {documents.map((doc, i) => (
              <DocCard
                key={doc.doc_id}
                doc={doc}
                onDelete={remove}
                index={i}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
```

**Step 4: Verify compilation**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -20`
Expected: No errors.

**Step 5: Commit**

```bash
git add frontend/src/pages/Documents.tsx frontend/src/components/documents/DocCard.tsx frontend/src/components/documents/UploadZone.tsx
git commit -m "feat(frontend): document grid layout, type-colored cards, redesigned upload zone"
```

---

### Task 6: Build Verification & Final Commit

**Files:** None (verification only)

**Step 1: Full TypeScript check**

Run: `cd frontend && npx tsc --noEmit`
Expected: Clean — zero errors.

**Step 2: Production build**

Run: `cd frontend && npm run build`
Expected: Build succeeds with output in `dist/`.

**Step 3: Visual verification**

Run: `cd frontend && npm run dev`
Then open `http://localhost:3000` and verify:
- [ ] Top nav bar with frosted glass effect, logo, pill tabs, dark mode toggle
- [ ] Dark mode toggles the entire theme when clicking the moon/sun icon
- [ ] Chat empty state shows floating sparkles icon, heading, suggested queries
- [ ] Chat input is a floating glass bar at the bottom with textarea, LLM dropdown, send button
- [ ] LLM dropdown opens upward with model options
- [ ] Messages animate in with slide-up effect
- [ ] User messages are right-aligned violet pills
- [ ] Assistant messages have avatar, markdown rendering
- [ ] Source cards have confidence bars (green/amber/red)
- [ ] Agent trace shows as horizontal pipeline pills
- [ ] Documents page shows grid of cards with type-specific gradient headers
- [ ] Document cards lift on hover with shadow
- [ ] Delete button appears on card hover
- [ ] Upload zone has dashed border, scales slightly on drag
- [ ] Background has subtle decorative blur orbs
- [ ] Custom thin scrollbar

**Step 4: Squash commit (if not already committed per-task)**

```bash
git add -A frontend/
git commit -m "feat(frontend): Glass Observatory redesign — frosted panels, violet-blue theme, floating chat input, markdown rendering, pipeline stepper, document grid"
```
