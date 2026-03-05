import { create } from "zustand";
import { type ChatMeta, streamChat } from "@/api/chat";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  meta?: ChatMeta;
  isStreaming?: boolean;
}

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  llm: "openai" | "claude";
  selectedDocIds: string[];
  abortController: AbortController | null;

  setLLM: (llm: "openai" | "claude") => void;
  setSelectedDocIds: (ids: string[]) => void;
  sendMessage: (question: string) => void;
  stopGeneration: () => void;
  clearMessages: () => void;
}

let messageIdCounter = 0;
const nextId = () => `msg-${++messageIdCounter}`;

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isLoading: false,
  llm: "openai",
  selectedDocIds: [],
  abortController: null,

  setLLM: (llm) => set({ llm }),
  setSelectedDocIds: (ids) => set({ selectedDocIds: ids }),

  sendMessage: (question: string) => {
    const userMsg: Message = { id: nextId(), role: "user", content: question };
    const assistantId = nextId();
    const assistantMsg: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      isStreaming: true,
    };

    set((s) => ({
      messages: [...s.messages, userMsg, assistantMsg],
      isLoading: true,
    }));

    const controller = streamChat(question, {
      llm: get().llm,
      doc_ids: get().selectedDocIds,
      onMeta: (meta) => {
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === assistantId ? { ...m, meta } : m
          ),
        }));
      },
      onToken: (token) => {
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === assistantId ? { ...m, content: m.content + token } : m
          ),
        }));
      },
      onDone: () => {
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === assistantId ? { ...m, isStreaming: false } : m
          ),
          isLoading: false,
          abortController: null,
        }));
      },
      onError: (error) => {
        set((s) => ({
          messages: s.messages.map((m) =>
            m.id === assistantId
              ? { ...m, content: `Error: ${error}`, isStreaming: false }
              : m
          ),
          isLoading: false,
          abortController: null,
        }));
      },
    });

    set({ abortController: controller });
  },

  stopGeneration: () => {
    get().abortController?.abort();
    set((s) => ({
      messages: s.messages.map((m) =>
        m.isStreaming ? { ...m, isStreaming: false } : m
      ),
      isLoading: false,
      abortController: null,
    }));
  },

  clearMessages: () => set({ messages: [], isLoading: false }),
}));
