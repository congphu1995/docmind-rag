import api from "./client";

export interface ChunkChild {
  chunk_id: string;
  parent_id: string;
  content: string;
  content_raw: string;
  type: string;
  page: number;
  section: string;
}

export interface ChunkNode {
  chunk_id: string;
  content_raw: string;
  content_markdown: string | null;
  content_html: string | null;
  type: string;
  page: number;
  section: string;
  language: string;
  word_count: number;
  children: ChunkChild[];
}

export interface ChunksResponse {
  doc_id: string;
  chunks: ChunkNode[];
  total: number;
}

export async function fetchDocumentChunks(
  docId: string,
  filters?: { type?: string; page?: number; search?: string },
): Promise<ChunksResponse> {
  const params = new URLSearchParams();
  if (filters?.type) params.set("type_filter", filters.type);
  if (filters?.page !== undefined) params.set("page_filter", String(filters.page));
  if (filters?.search) params.set("search", filters.search);

  const { data } = await api.get(`/documents/${docId}/chunks?${params}`);
  return data;
}
