export interface ChatMeta {
  sources: Array<{
    source_num: number;
    doc_name: string;
    page: number;
    section: string;
    content_preview: string;
    score: number;
  }>;
  llm_used: string;
  hyde_used: boolean;
  query_type: string;
  agent_trace: string[];
}

export function streamChat(
  question: string,
  options: {
    llm?: string;
    doc_ids?: string[];
    onMeta: (meta: ChatMeta) => void;
    onToken: (token: string) => void;
    onDone: () => void;
    onError: (error: string) => void;
  }
): AbortController {
  const controller = new AbortController();

  const body = JSON.stringify({
    question,
    llm: options.llm || "openai",
    doc_ids: options.doc_ids || [],
    stream: true,
  });

  fetch("/api/v1/chat/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        options.onError(`HTTP ${response.status}`);
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        options.onError("No response body");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);

          if (data === "[DONE]") {
            options.onDone();
            return;
          }

          if (data.startsWith("__META__") && data.endsWith("__META__")) {
            const json = data.slice(8, -8);
            try {
              options.onMeta(JSON.parse(json));
            } catch {
              /* ignore parse errors */
            }
            continue;
          }

          options.onToken(data);
        }
      }

      options.onDone();
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        options.onError(err.message);
      }
    });

  return controller;
}
