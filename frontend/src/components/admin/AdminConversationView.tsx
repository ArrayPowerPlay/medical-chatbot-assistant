import React, { useEffect, useState, useRef } from 'react';
import { adminApi } from '../../api/adminApi';
import type { Message } from '../../api/conversationApi';
import { X, Loader2, BookOpen, ChevronUp, ChevronDown, ThumbsUp, ThumbsDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

interface AdminConversationViewProps {
  conversationId: string;
  conversationTitle: string;
  username?: string;
  email?: string;
  highlightMessageId?: number | string;
  onClose: () => void;
}

const AdminConversationView: React.FC<AdminConversationViewProps> = ({ 
  conversationId, 
  conversationTitle, 
  username,
  email,
  highlightMessageId,
  onClose 
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  // Track open sources states per message ID
  const [openSources, setOpenSources] = useState<Record<string, boolean>>({});
  
  const highlightRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchMessages = async () => {
      try {
        const data = await adminApi.getConversationMessages(conversationId, 100);
        setMessages(data.messages);
      } catch (err: any) {
        setError(err.message || 'Failed to load conversation');
      } finally {
        setLoading(false);
      }
    };
    fetchMessages();
  }, [conversationId]);

  // Auto-scroll to highlighted message
  useEffect(() => {
    if (!loading && highlightMessageId && highlightRef.current) {
      setTimeout(() => {
        highlightRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 100);
    }
  }, [loading, highlightMessageId]);

  const toggleSources = (msgId: string) => {
    setOpenSources(prev => ({ ...prev, [msgId]: !prev[msgId] }));
  };

  const displayUser = username || email || 'User';

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/50">
      <div className="w-full max-w-3xl bg-slate-50 dark:bg-slate-900 h-full flex flex-col shadow-2xl animate-in slide-in-from-right duration-300 border-l border-slate-200 dark:border-slate-800">
        
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 shrink-0">
          <div className="flex-1 min-w-0 pr-4">
            <h3 className="font-semibold text-lg text-slate-800 dark:text-white truncate">{conversationTitle}</h3>
          </div>
          <button 
            onClick={onClose}
            className="p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full transition-colors shrink-0"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
          {loading && (
            <div className="flex items-center justify-center h-full text-slate-500">
              <Loader2 className="w-6 h-6 animate-spin mr-2" /> Loading messages...
            </div>
          )}
          {error && (
            <div className="flex items-center justify-center h-full text-red-500">
              {error}
            </div>
          )}
          {!loading && !error && messages.length === 0 && (
            <div className="text-center text-slate-500 mt-10">No messages found.</div>
          )}
          
          {!loading && !error && messages.map((msg) => {
            const isHighlighted = String(msg.id) === String(highlightMessageId);
            const isUser = msg.role === 'user';
            
            return (
              <div 
                key={msg.id}
                ref={isHighlighted ? highlightRef : null}
                className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}
              >
                <div 
                  className={`max-w-[90%] rounded-2xl p-4 transition-all duration-500 ${
                    isUser 
                      ? 'bg-blue-600 text-white rounded-br-none' 
                      : 'bg-white dark:bg-slate-800 border text-slate-800 dark:text-slate-200 rounded-bl-none shadow-sm'
                  } ${
                    isHighlighted 
                      ? 'ring-2 ring-amber-500 ring-offset-2 ring-offset-slate-50 dark:ring-offset-slate-900 border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20' 
                      : isUser ? 'border-transparent' : 'border-slate-200 dark:border-slate-700'
                  }`}
                >
                  <div className="text-xs font-bold mb-2 opacity-80 uppercase tracking-wider">
                    {isUser ? displayUser : 'MedKG-RAG'}
                  </div>
                  
                  <div className={`prose dark:prose-invert max-w-none text-sm break-words
                    ${isUser ? 'prose-p:text-white prose-strong:text-white' : ''}`}
                  >
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeRaw]}
                    >
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                  
                  {/* Sources Display for Assistant */}
                  {!isUser && msg.sources && msg.sources.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-700/50">
                      <button
                        onClick={() => toggleSources(msg.id)}
                        className="flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                      >
                        <BookOpen className="w-3.5 h-3.5" />
                        <span>View relevant sources ({msg.sources.length})</span>
                        {openSources[String(msg.id)] ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                      </button>
                      
                      {openSources[String(msg.id)] && (
                        <div className="mt-3 space-y-2 max-h-[400px] overflow-y-auto pr-2">
                          {msg.sources.map((source: any, i: number) => (
                            <div key={i} className="text-xs bg-slate-50 dark:bg-slate-950 p-3 rounded-lg border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300">
                              <div className="flex items-center gap-2 mb-1">
                                <span className="font-semibold text-blue-600 dark:text-blue-400">
                                  {source.source_type === 'kg' ? 'Knowledge Graph' : 'Document'}
                                </span>
                                {source.pmid && (
                                  <span className="text-[10px] px-1.5 py-0.5 bg-slate-200 dark:bg-slate-800 rounded-full">
                                    PMID: {source.pmid}
                                  </span>
                                )}
                              </div>
                              <div className="mt-1 line-clamp-3 hover:line-clamp-none transition-all">
                                {source.content}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Feedback Display */}
                  {!isUser && msg.feedback_type && msg.feedback_type !== 'none' && (
                    <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-700/50">
                      <div className="flex flex-col gap-2">
                        {msg.feedback_type === 'like' ? (
                          <div className="flex items-center gap-1.5 text-green-600 dark:text-green-500 bg-green-50 dark:bg-green-900/20 px-2 py-1 rounded text-xs font-medium w-fit">
                            <ThumbsUp className="w-3.5 h-3.5" />
                            Liked
                          </div>
                        ) : (
                          <div className="flex items-center gap-1.5 text-red-600 dark:text-red-500 bg-red-50 dark:bg-red-900/20 px-2 py-1 rounded text-xs font-medium w-fit">
                            <ThumbsDown className="w-3.5 h-3.5" />
                            Disliked
                          </div>
                        )}
                        
                        {msg.feedback_comment && (
                          <div className="text-sm italic text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-900/50 p-3 rounded-lg border border-slate-100 dark:border-slate-800">
                            "{msg.feedback_comment}"
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default AdminConversationView;
