import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { MessageSquare, FileText } from "lucide-react";
import Chat from "./pages/Chat";
import Documents from "./pages/Documents";

type Page = "chat" | "documents";

export default function App() {
  const [page, setPage] = useState<Page>("chat");

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <nav className="w-56 bg-card border-r flex flex-col">
        <div className="p-4">
          <h1 className="text-lg font-semibold">DocMind</h1>
          <p className="text-xs text-muted-foreground">Document Intelligence</p>
        </div>
        <Separator />
        <div className="flex-1 p-2 space-y-1">
          <Button
            variant={page === "chat" ? "secondary" : "ghost"}
            className="w-full justify-start gap-2"
            onClick={() => setPage("chat")}
          >
            <MessageSquare className="h-4 w-4" />
            Chat
          </Button>
          <Button
            variant={page === "documents" ? "secondary" : "ghost"}
            className="w-full justify-start gap-2"
            onClick={() => setPage("documents")}
          >
            <FileText className="h-4 w-4" />
            Documents
          </Button>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-hidden">
        {page === "chat" && <Chat />}
        {page === "documents" && <Documents />}
      </main>
    </div>
  );
}
