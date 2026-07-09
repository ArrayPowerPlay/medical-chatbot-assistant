import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { User, Bot } from 'lucide-react';
import type { ChatMessageData } from '../../hooks/useChatStream';

interface ChatMessageProps {
  message: ChatMessageData;
  isThinking?: boolean;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isThinking }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-4 p-4 md:p-6 lg:px-8 max-w-4xl mx-auto w-full group ${isUser ? 'bg-white dark:bg-slate-900' : 'bg-slate-50 dark:bg-slate-900/50'}`}>
      {/* Avatar */}
      <div className="shrink-0 pt-1">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white ${isUser ? 'bg-blue-600' : 'bg-teal-600'}`}>
          {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
        </div>
      </div>

      {/* Message Content */}
      <div className="flex-1 min-w-0">
        <div className="font-semibold text-slate-800 dark:text-slate-200 mb-1">
          {isUser ? 'You' : 'MedKG-RAG'}
        </div>
        
        {/* Thinking Indicator */}
        {!isUser && isThinking && (
          <div className="flex items-center gap-2 text-slate-500 mt-2">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-teal-500"></span>
            </span>
            <span className="text-sm font-medium animate-pulse">Thinking...</span>
          </div>
        )}

        {/* Markdown Content */}
        {!isThinking && (
          <div className="prose prose-slate dark:prose-invert max-w-none break-words
            prose-p:leading-relaxed prose-pre:bg-slate-100 dark:prose-pre:bg-slate-800 
            prose-pre:text-slate-800 dark:prose-pre:text-slate-200"
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
