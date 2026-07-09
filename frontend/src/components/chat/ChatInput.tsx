import React, { useState, useRef, useEffect } from 'react';
import { Send, Square } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  onStop: () => void;
  isGenerating: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, onStop, isGenerating }) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { user } = useAuthStore();

  const handleSend = () => {
    if (!input.trim() || isGenerating) return;
    onSendMessage(input.trim());
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'; // reset height
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    // Auto-resize
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  };

  // Focus input on load, except on mobile where it might popup keyboard annoyingly
  useEffect(() => {
    if (window.innerWidth > 768) {
      textareaRef.current?.focus();
    }
  }, []);

  const isGuestExceeded = user?.role === 'guest' && (user?.question_count || 0) >= 10;

  return (
    <div className="w-full max-w-4xl mx-auto p-4 pt-0">
      <div className="relative flex flex-col bg-white dark:bg-slate-800 rounded-2xl border border-slate-300 dark:border-slate-700 shadow-sm overflow-hidden focus-within:ring-1 focus-within:ring-blue-500 transition-shadow">
        
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={isGuestExceeded ? "You have reached the guest limit." : "Ask a medical question..."}
          disabled={isGenerating || isGuestExceeded}
          className="w-full max-h-[200px] py-4 pl-4 pr-12 bg-transparent text-slate-800 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 resize-none outline-none leading-relaxed"
          rows={1}
        />

        <div className="absolute right-2 bottom-3 flex items-center justify-center">
          {isGenerating ? (
            <button
              onClick={onStop}
              className="p-2 text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10 rounded-lg transition-colors flex items-center justify-center"
              title="Stop generating"
            >
              <Square className="w-5 h-5 fill-current" />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim() || isGuestExceeded}
              className={`p-2 rounded-lg transition-colors flex items-center justify-center ${
                input.trim() && !isGuestExceeded
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'bg-slate-100 text-slate-400 dark:bg-slate-700 dark:text-slate-500 cursor-not-allowed'
              }`}
              title="Send message"
            >
              <Send className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>
      
      {/* Disclaimer / Info */}
      <div className="mt-2 text-center text-xs text-slate-500 dark:text-slate-400 px-4">
        MedKG-RAG is an AI assistant for medical knowledge graph search. It can make mistakes. Consider verifying important information.
      </div>
    </div>
  );
};

export default ChatInput;
