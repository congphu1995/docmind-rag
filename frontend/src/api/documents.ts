import api from "./client";

export interface DocumentInfo {
  doc_id: string;
  doc_name: string;
  status: string;
}

export interface UploadResult {
  status: string;
  task_id: string;
  message: string;
}

export async function uploadDocument(
  file: File,
  language = "en"
): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("language", language);
  const { data } = await api.post("/documents/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function listDocuments(): Promise<{
  documents: DocumentInfo[];
  total: number;
}> {
  const { data } = await api.get("/documents/");
  return data;
}

export async function deleteDocument(docId: string): Promise<void> {
  await api.delete(`/documents/${docId}`);
}

export async function getTaskStatus(
  taskId: string
): Promise<{ status: string; result?: Record<string, unknown> }> {
  const { data } = await api.get(`/documents/task/${taskId}`);
  return data;
}
