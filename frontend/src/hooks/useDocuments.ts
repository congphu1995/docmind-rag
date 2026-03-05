import { useEffect } from "react";
import { useDocumentStore } from "@/stores/documentStore";

/**
 * Convenience hook wrapping the document store.
 * Auto-fetches on mount.
 */
export function useDocuments() {
  const {
    documents,
    isLoading,
    uploadProgress,
    fetchDocuments,
    upload,
    remove,
  } = useDocumentStore();

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  return {
    documents,
    isLoading,
    uploadProgress,
    refresh: fetchDocuments,
    upload,
    remove,
  };
}
