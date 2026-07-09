import React, { useEffect, useRef, useState } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { useChatStream } from '../../hooks/useChatStream';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { conversationApi } from '../../api/conversationApi';
import { ChevronUp, ChevronDown, X } from 'lucide-react';

const ChatLayout: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { messages, setMessages, isGenerating, isThinking, sendMessage, stopGeneration } = useChatStream();
  const bottomRef = useRef<HTMLDivElement>(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const prevIdRef = useRef<string | undefined>(undefined);

  // Search Highlight Logic
  const matchIdsParam = searchParams.get('match_ids');
  const searchQ = searchParams.get('search_q') || '';
  const matchIds = matchIdsParam ? matchIdsParam.split(',') : [];
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);

  // Reset match index when conversation changes
  useEffect(() => {
    setCurrentMatchIndex(0);
  }, [id, matchIdsParam]);

  // Load history when conversation ID changes
  useEffect(() => {
    // If the ID hasn't changed, don't do anything (prevents refetching when isGenerating changes)
    if (id === prevIdRef.current) {
      return;
    }

    // If no ID, we are on the home page, so clear messages
    if (!id) {
      if (isGenerating) stopGeneration();
      setMessages([]);
      prevIdRef.current = id;
      return;
    }

    // Skip fetching history if we are currently generating a response for a newly created chat
    if (isGenerating && prevIdRef.current === undefined) {
      prevIdRef.current = id;
      return;
    }
    
    // If we switch to a different conversation while generating, abort the current generation
    if (isGenerating && prevIdRef.current !== undefined) {
      stopGeneration();
    }

    prevIdRef.current = id;
    setMessages([]); // Clear previous conversation's messages
    setIsLoadingHistory(true);
    conversationApi.getMessages(id)
      .then((data) => {
        // Reverse messages because API returns newest first (cursor based)
        const history = data.messages.map(m => ({
          id: m.id.toString(),
          role: m.role,
          content: m.content,
          feedback_type: m.feedback_type,
          feedback_comment: m.feedback_comment,
          sources: m.sources
        }));
        setMessages(history);
      })
      .catch((err) => {
        console.error("Failed to load history", err);
        if (err.response?.status === 404) {
          navigate('/c', { replace: true });
        }
      })
      .finally(() => {
        setIsLoadingHistory(false);
      });
  }, [id, setMessages, navigate, isGenerating]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    // Only scroll to bottom if there are NO active search matches. 
    // If there are matches, ChatMessage handles scrolling to the matched message.
    if (bottomRef.current && matchIds.length === 0) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isThinking, matchIds.length]);

  return (
    <div className="flex flex-col h-full w-full bg-white dark:bg-slate-900 relative">
      
      {/* Find-in-Page Widget */}
      {matchIds.length > 0 && (
        <div className="absolute top-4 right-4 bg-white dark:bg-slate-800 shadow-xl rounded-lg border border-slate-200 dark:border-slate-700 flex items-center p-2 z-20 transition-all transform animate-in fade-in slide-in-from-top-4">
          <div className="text-sm font-medium px-2 max-w-[150px] truncate text-slate-800 dark:text-slate-200" title={searchQ}>
            "{searchQ}"
          </div>
          <div className="text-xs text-slate-500 border-l border-slate-300 dark:border-slate-600 pl-2 mr-2 min-w-[3rem] text-center">
            {currentMatchIndex + 1} / {matchIds.length}
          </div>
          <button 
            onClick={() => setCurrentMatchIndex(prev => Math.max(0, prev - 1))}
            disabled={currentMatchIndex === 0}
            className="p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-700 hover:text-slate-800 dark:hover:text-slate-200 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            title="Previous match"
          >
            <ChevronUp className="w-4 h-4" />
          </button>
          <button 
            onClick={() => setCurrentMatchIndex(prev => Math.min(matchIds.length - 1, prev + 1))}
            disabled={currentMatchIndex === matchIds.length - 1}
            className="p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-700 hover:text-slate-800 dark:hover:text-slate-200 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            title="Next match"
          >
            <ChevronDown className="w-4 h-4" />
          </button>
          <div className="w-px h-4 bg-slate-300 dark:bg-slate-600 mx-1"></div>
          <button 
            onClick={() => {
              searchParams.delete('match_ids');
              searchParams.delete('search_q');
              setSearchParams(searchParams);
            }}
            className="p-1.5 text-slate-400 hover:bg-red-50 dark:hover:bg-red-900/30 hover:text-red-500 rounded transition-colors"
            title="Clear search"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto scroll-smooth">
        {isLoadingHistory ? (
          <div className="h-full flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        ) : messages.length === 0 ? (
          // Empty State
          <div className="h-full flex flex-col items-center justify-center text-center p-4">
            <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center mb-6">
              <span className="text-3xl">⚕️</span>
            </div>
            <h2 className="text-2xl font-semibold text-slate-800 dark:text-slate-200 mb-2">
              Welcome to Med Assistant
            </h2>
            <p className="text-slate-500 dark:text-slate-400 max-w-md">
              Ask questions about diseases, treatments, symptoms, and more. 
              The system searches medical literature and knowledge graphs to provide evidence-based answers.
            </p>
          </div>
        ) : (
          // Message List
          <div className="flex flex-col pb-6">
            {messages.map((msg, idx) => {
              // If it's the last assistant message and we are "thinking", pass the prop
              const isLast = idx === messages.length - 1;
              const currentMsgIsThinking = isLast && msg.role === 'assistant' && isThinking;
              
              const isMatch = matchIds.includes(msg.id);
              const isActiveMatch = isMatch && msg.id === matchIds[currentMatchIndex];

              return (
                <ChatMessage 
                  key={msg.id} 
                  message={msg} 
                  isThinking={currentMsgIsThinking} 
                  isHighlighted={isMatch}
                  isActiveMatch={isActiveMatch}
                />
              );
            })}
            <div ref={bottomRef} className="h-4" />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="shrink-0 bg-gradient-to-t from-white via-white to-transparent dark:from-slate-900 dark:via-slate-900 dark:to-transparent pt-6">
        <ChatInput 
          onSendMessage={(text) => sendMessage(text, id)} 
          onStop={stopGeneration} 
          isGenerating={isGenerating} 
        />
      </div>
    </div>
  );
};

export default ChatLayout;
