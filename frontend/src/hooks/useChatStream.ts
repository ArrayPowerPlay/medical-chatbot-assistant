import { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { useAuthStore } from '../stores/authStore';
import { useConversationStore } from '../stores/conversationStore';

export interface ChatMessageData {
  id: string; // temporary id for ui
  role: 'user' | 'assistant';
  content: string;
  feedback_type?: 'like' | 'dislike' | 'none';
  feedback_comment?: string;
  sources?: any[];
}

export const useChatStream = () => {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const navigate = useNavigate();

  const token = useAuthStore((state: any) => state.token);
  const logout = useAuthStore((state: any) => state.logout);
  const fetchConversations = useConversationStore((state: any) => state.fetchConversations);

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

      // Abort any ongoing request to prevent concurrent spam
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

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
          openWhenHidden: true,
          async onopen(response) {
            if (response.ok && response.headers.get('content-type')?.includes('text/event-stream')) {
              // Add empty assistant message placeholder if not exists
              setMessages((prev) => {
                const lastMsg = prev[prev.length - 1];
                if (lastMsg?.id === botMsgId) return prev;
                return [
                  ...prev,
                  { id: botMsgId, role: 'assistant', content: '' },
                ];
              });
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
              setIsThinking(false);
              throw new Error(msg.data);
            }
            if (msg.event === 'metadata') {
              try {
                const parsed = JSON.parse(msg.data);
                if (parsed.conversation_id && !conversationId) {
                  // Navigate to the new conversation URL without reloading the page
                  navigate(`/c/${parsed.conversation_id}`, { replace: true });
                  // Immediately refresh the sidebar so the user sees the new conversation
                  fetchConversations();
                }
                if (parsed.sources && parsed.sources.length > 0) {
                  setMessages((prev) => {
                    const newMessages = [...prev];
                    const lastIndex = newMessages.length - 1;
                    const lastMsg = newMessages[lastIndex];
                    if (lastMsg && lastMsg.role === 'assistant') {
                      newMessages[lastIndex] = { ...lastMsg, sources: parsed.sources };
                    }
                    return newMessages;
                  });
                }
              } catch (e) { }
              return;
            }
            if (msg.event === 'message_id') {
              try {
                const parsed = JSON.parse(msg.data);
                if (parsed.message_id) {
                  setMessages((prev) => {
                    const newMessages = [...prev];
                    const lastIndex = newMessages.length - 1;
                    const lastMsg = newMessages[lastIndex];
                    if (lastMsg && lastMsg.role === 'assistant') {
                      newMessages[lastIndex] = { ...lastMsg, id: parsed.message_id.toString() };
                    }
                    return newMessages;
                  });
                  
                  const storeState = useAuthStore.getState() as any;
                  if (storeState.user?.role === 'guest') {
                    storeState.updateUser({ question_count: (storeState.user.question_count || 0) + 1 });
                  }
                }
              } catch (e) { }
              return;
            }
            if (msg.event === 'done') {
              return; // Ignore the done event to prevent rendering {"status": "success"}
            }
            if (msg.data) {
              // Parse stream chunk
              try {
                let tokenStr = msg.data;

                try {
                  const parsed = JSON.parse(tokenStr);
                  if (typeof parsed === 'string') {
                    tokenStr = parsed;
                  } else if (parsed && typeof parsed === 'object' && parsed.token) {
                    tokenStr = parsed.token;
                  }
                } catch (e) {
                  // Fallback: use raw tokenStr if it's not valid JSON
                }

                // Only turn off thinking when we actually receive the first text token
                setIsThinking(false);

                setMessages((prev) => {
                  const newMessages = [...prev];
                  const lastIndex = newMessages.length - 1;
                  const lastMsg = newMessages[lastIndex];
                  if (lastMsg && lastMsg.role === 'assistant') {
                    newMessages[lastIndex] = { ...lastMsg, content: lastMsg.content + tokenStr };
                  }
                  return newMessages;
                });
              } catch (e) {
                // fallback to append raw string
                setMessages((prev) => {
                  const newMessages = [...prev];
                  const lastIndex = newMessages.length - 1;
                  const lastMsg = newMessages[lastIndex];
                  if (lastMsg && lastMsg.role === 'assistant') {
                    newMessages[lastIndex] = { ...lastMsg, content: lastMsg.content + msg.data };
                  }
                  return newMessages;
                });
              }
            }
          },
          onclose() {
            setIsGenerating(false);
            setIsThinking(false);
            if (!conversationId) {
              fetchConversations();
            }
            // Explicitly abort to prevent fetch-event-source from automatically retrying on close
            if (abortControllerRef.current) {
              abortControllerRef.current.abort();
              abortControllerRef.current = null;
            }
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

          const errorMsg = err.message?.includes('Forbidden')
            ? "Guest question limit exceeded. Please register for a free account to continue."
            : "I'm sorry, I encountered an internal system error while processing your request. Please try again later.";

          setMessages((prev) => {
            const newMessages = [...prev];
            const lastIndex = newMessages.length - 1;
            const lastMsg = newMessages[lastIndex];

            if (lastMsg && lastMsg.role === 'assistant' && lastMsg.content === '') {
              newMessages[lastIndex] = { ...lastMsg, content: errorMsg };
              return newMessages;
            } else if (lastMsg && lastMsg.role === 'assistant') {
              if (!lastMsg.content.includes(errorMsg)) {
                newMessages[lastIndex] = { ...lastMsg, content: lastMsg.content + `\n\n*[System Error: ${errorMsg}]*` };
              }
              return newMessages;
            } else {
              return [
                ...newMessages,
                {
                  id: (Date.now() + 2).toString(),
                  role: 'assistant',
                  content: errorMsg,
                },
              ];
            }
          });
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
