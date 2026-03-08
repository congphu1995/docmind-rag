import { useEffect } from "react";
import { useChunkStore } from "@/stores/chunkStore";

export function useChunks(docId: string | null) {
  const { chunks, isLoading, loadChunks, filters } = useChunkStore();

  useEffect(() => {
    if (docId) loadChunks(docId);
  }, [docId, filters, loadChunks]);

  return { chunks, isLoading };
}
