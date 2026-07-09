import { useState, useRef, useCallback } from 'react';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { useAuthStore } from '../stores/authStore';

export interface ChatMessageData {
  id: string; // temporary id for ui
  role: 'user' | 'assistant';
  content: string;
}

export const useChatStream = () => {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  
  const token = useAuthStore((state: any) => state.token);
  const logout = useAuthStore((state: any) => state.logout);

  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsGenerating(false);
    setIsThinking(false);
  }, []);

  const sendMessage = useCallback(
    async (question: string, conversationId?: string) => {
      if (!question.trim()) return;

      const userMsgId = Date.now().toString();
      const botMsgId = (Date.now() + 1).toString();

      // Add user message to UI immediately
      setMessages((prev) => [
        ...prev,
        { id: userMsgId, role: 'user', content: question },
      ]);

      setIsGenerating(true);
      setIsThinking(true);

      const ctrl = new AbortController();
      abortControllerRef.current = ctrl;

      try {
        await fetchEventSource('http://localhost:8000/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            question,
            conversation_id: conversationId,
            top_k: 5,
          }),
          signal: ctrl.signal,
          async onopen(response) {
            if (response.ok && response.headers.get('content-type')?.includes('text/event-stream')) {
              setIsThinking(false);
              // Add empty assistant message placeholder
              setMessages((prev) => [
                ...prev,
                { id: botMsgId, role: 'assistant', content: '' },
              ]);
              return; // everything's good
            } else if (response.status >= 400 && response.status < 500 && response.status !== 429) {
              if (response.status === 401) {
                logout();
                throw new Error("Unauthorized");
              }
              if (response.status === 403) {
                throw new Error("Forbidden: Guest limit exceeded");
              }
              throw new Error(`Client Error ${response.status}`);
            } else {
              throw new Error(`Server Error ${response.status}`);
            }
          },
          onmessage(msg) {
            if (msg.event === 'error') {
              throw new Error(msg.data);
            }
            if (msg.data) {
              // Parse stream chunk
              try {
                // Usually SSE data is JSON, assuming the backend streams raw text or JSON
                // If it streams simple text tokens inside msg.data:
                // Let's assume the backend will send standard SSE chunks.
                // We'll update backend later to stream either raw text token or {"token": "..."}
                let tokenStr = msg.data;
                
                // Let's assume it sends JSON `{ "token": "..." }` to handle line breaks better
                if (tokenStr.startsWith('{') && tokenStr.endsWith('}')) {
                  const parsed = JSON.parse(tokenStr);
                  if (parsed.token) {
                     tokenStr = parsed.token;
                  }
                }
                
                setMessages((prev) => {
                  const newMessages = [...prev];
                  const lastMsg = newMessages[newMessages.length - 1];
                  if (lastMsg && lastMsg.role === 'assistant') {
                    lastMsg.content += tokenStr;
                  }
                  return newMessages;
                });
              } catch (e) {
                // fallback to append raw string
                setMessages((prev) => {
                  const newMessages = [...prev];
                  const lastMsg = newMessages[newMessages.length - 1];
                  if (lastMsg && lastMsg.role === 'assistant') {
                    lastMsg.content += msg.data;
                  }
                  return newMessages;
                });
              }
            }
          },
          onclose() {
            setIsGenerating(false);
            setIsThinking(false);
          },
          onerror(err) {
            console.error('SSE Error:', err);
            setIsGenerating(false);
            setIsThinking(false);
            
            // Handle display error in UI if needed
            if (err.message?.includes('Forbidden')) {
              alert("Guest question limit exceeded. Please register for a free account.");
            }
            
            throw err; // rethrow to stop fetchEventSource from retrying
          },
        });
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          console.error('Fetch error:', err);
          setIsGenerating(false);
          setIsThinking(false);
        }
      }
    },
    [token, logout]
  );

  return {
    messages,
    setMessages, // allow manual reset
    isGenerating,
    isThinking,
    sendMessage,
    stopGeneration,
  };
};
