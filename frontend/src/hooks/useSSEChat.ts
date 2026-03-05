import { useCallback } from "react";
import { useChatStore } from "@/stores/chatStore";

/**
 * Convenience hook wrapping the chat store.
 * Components use this instead of importing the store directly.
 */
export function useSSEChat() {
  const {
    messages,
    isLoading,
    llm,
    selectedDocIds,
    setLLM,
    setSelectedDocIds,
    sendMessage,
    stopGeneration,
    clearMessages,
  } = useChatStore();

  const send = useCallback(
    (question: string) => {
      if (!question.trim() || isLoading) return;
      sendMessage(question.trim());
    },
    [isLoading, sendMessage]
  );

  return {
    messages,
    isLoading,
    llm,
    selectedDocIds,
    setLLM,
    setSelectedDocIds,
    send,
    stop: stopGeneration,
    clear: clearMessages,
  };
}
