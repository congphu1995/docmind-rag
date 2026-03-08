import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { MessageSquare, FileText, Layers, Moon, Sun, BrainCircuit, LogOut } from "lucide-react";
import Chat from "./pages/Chat";
import Documents from "./pages/Documents";
import Chunks from "./pages/Chunks";
import Login from "./pages/Login";
import { useAuthStore } from "@/stores/authStore";

type Page = "chat" | "documents" | "chunks";

export default function App() {
  const [page, setPage] = useState<Page>("chat");
  const [dark, setDark] = useState(false);
  const { isAuthenticated, user, logout, loadFromStorage } = useAuthStore();

  useEffect(() => {
    loadFromStorage();
  }, [loadFromStorage]);

  const toggleDark = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
  };

  if (!isAuthenticated) {
    return <Login />;
  }

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
              <BrainCircuit className="w-5 h-5 text-primary-foreground" />
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
                { key: "chunks", label: "Chunks", icon: Layers },
              ] as const
            ).map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setPage(key)}
                aria-current={page === key ? "page" : undefined}
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

          {/* Right-side controls */}
          <div className="flex items-center gap-1">
            {user && (
              <span className="text-xs text-muted-foreground mr-2 hidden sm:inline">
                {user.username}
              </span>
            )}
            <button
              onClick={toggleDark}
              aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
              className="w-9 h-9 rounded-xl flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
            >
              {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
            <button
              onClick={logout}
              aria-label="Log out"
              className="w-9 h-9 rounded-xl flex items-center justify-center text-muted-foreground hover:text-destructive hover:bg-muted/50 transition-colors"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className="pt-16 h-screen">
        {page === "chat" && <Chat />}
        {page === "documents" && <Documents />}
        {page === "chunks" && <Chunks />}
      </main>
    </div>
  );
}
