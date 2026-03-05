import { create } from "zustand";
import {
  type DocumentInfo,
  deleteDocument,
  listDocuments,
  uploadDocument,
  getTaskStatus,
} from "@/api/documents";

interface DocumentState {
  documents: DocumentInfo[];
  isLoading: boolean;
  uploadProgress: Map<string, string>;

  fetchDocuments: () => Promise<void>;
  upload: (file: File) => Promise<void>;
  remove: (docId: string) => Promise<void>;
}

export const useDocumentStore = create<DocumentState>((set, get) => ({
  documents: [],
  isLoading: false,
  uploadProgress: new Map(),

  fetchDocuments: async () => {
    set({ isLoading: true });
    try {
      const { documents } = await listDocuments();
      set({ documents, isLoading: false });
    } catch {
      set({ isLoading: false });
    }
  },

  upload: async (file: File) => {
    const taskName = file.name;
    set((s) => ({
      uploadProgress: new Map(s.uploadProgress).set(taskName, "uploading"),
    }));

    try {
      const result = await uploadDocument(file);

      set((s) => ({
        uploadProgress: new Map(s.uploadProgress).set(taskName, "processing"),
      }));

      // Poll for completion
      const pollInterval = setInterval(async () => {
        try {
          const status = await getTaskStatus(result.task_id);
          if (status.status === "ready" || status.status === "success") {
            clearInterval(pollInterval);
            set((s) => {
              const progress = new Map(s.uploadProgress);
              progress.delete(taskName);
              return { uploadProgress: progress };
            });
            get().fetchDocuments();
          } else if (status.status === "failed") {
            clearInterval(pollInterval);
            set((s) => {
              const progress = new Map(s.uploadProgress);
              progress.set(taskName, "failed");
              return { uploadProgress: progress };
            });
          }
        } catch {
          clearInterval(pollInterval);
        }
      }, 2000);
    } catch {
      set((s) => ({
        uploadProgress: new Map(s.uploadProgress).set(taskName, "failed"),
      }));
    }
  },

  remove: async (docId: string) => {
    await deleteDocument(docId);
    set((s) => ({
      documents: s.documents.filter((d) => d.doc_id !== docId),
    }));
  },
}));
