import React, { useEffect, useRef } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { useChatStream } from '../../hooks/useChatStream';

const ChatLayout: React.FC = () => {
  const { messages, isGenerating, isThinking, sendMessage, stopGeneration } = useChatStream();
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isThinking]);

  return (
    <div className="flex flex-col h-full w-full bg-white dark:bg-slate-900">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto scroll-smooth">
        {messages.length === 0 ? (
          // Empty State
          <div className="h-full flex flex-col items-center justify-center text-center p-4">
            <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center mb-6">
              <span className="text-3xl">⚕️</span>
            </div>
            <h2 className="text-2xl font-semibold text-slate-800 dark:text-slate-200 mb-2">
              Welcome to MedKG-RAG
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
              
              return (
                <ChatMessage 
                  key={msg.id} 
                  message={msg} 
                  isThinking={currentMsgIsThinking} 
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
          onSendMessage={(text) => sendMessage(text)} 
          onStop={stopGeneration} 
          isGenerating={isGenerating} 
        />
      </div>
    </div>
  );
};

export default ChatLayout;
