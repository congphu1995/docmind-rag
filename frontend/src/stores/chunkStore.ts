import { create } from "zustand";
import { type ChunkNode, fetchDocumentChunks } from "@/api/chunks";

interface ChunkState {
  chunks: ChunkNode[];
  selectedChunk: ChunkNode | null;
  isLoading: boolean;
  filters: { type?: string; page?: number; search?: string };

  loadChunks: (docId: string) => Promise<void>;
  selectChunk: (chunk: ChunkNode | null) => void;
  setFilters: (filters: Partial<ChunkState["filters"]>) => void;
}

export const useChunkStore = create<ChunkState>((set, get) => ({
  chunks: [],
  selectedChunk: null,
  isLoading: false,
  filters: {},

  loadChunks: async (docId) => {
    set({ isLoading: true });
    try {
      const data = await fetchDocumentChunks(docId, get().filters);
      set({ chunks: data.chunks, isLoading: false });
    } catch {
      set({ chunks: [], isLoading: false });
    }
  },

  selectChunk: (chunk) => set({ selectedChunk: chunk }),

  setFilters: (filters) =>
    set((s) => ({ filters: { ...s.filters, ...filters } })),
}));
