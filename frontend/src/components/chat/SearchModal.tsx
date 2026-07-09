import React, { useState, useEffect, useRef } from 'react';
import { Search, X, MessageSquare, ChevronRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { conversationApi } from '../../api/conversationApi';
import type { Conversation } from '../../api/conversationApi';

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const SearchModal: React.FC<SearchModalProps> = ({ isOpen, onClose }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setResults([]);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  useEffect(() => {
    const fetchResults = async () => {
      if (!query.trim()) {
        setResults([]);
        return;
      }
      setIsLoading(true);
      try {
        const data = await conversationApi.searchConversations(query);
        setResults(data.conversations);
      } catch (error) {
        console.error('Search failed:', error);
      } finally {
        setIsLoading(false);
      }
    };

    const debounceId = setTimeout(fetchResults, 300);
    return () => clearTimeout(debounceId);
  }, [query]);

  if (!isOpen) return null;

  const handleSelect = (conv: Conversation) => {
    onClose();
    if (conv.matched_message_ids && conv.matched_message_ids.length > 0) {
      // Pass matched IDs in query params or state
      const matchParam = conv.matched_message_ids.join(',');
      navigate(`/c/${conv.id}?match_ids=${matchParam}&search_q=${encodeURIComponent(query)}`);
    } else {
      navigate(`/c/${conv.id}`);
    }
  };

  const renderHighlighted = (text: string, highlight: string) => {
    if (!highlight.trim()) return text;
    const parts = text.split(new RegExp(`(${highlight})`, 'gi'));
    return (
      <>
        {parts.map((part, i) => 
          part.toLowerCase() === highlight.toLowerCase() ? 
            <mark key={i} className="bg-amber-200 dark:bg-amber-900/60 text-amber-900 dark:text-amber-100 font-medium px-0.5 rounded bg-transparent">{part}</mark> : part
        )}
      </>
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 bg-slate-900/50 backdrop-blur-sm px-4">
      <div 
        className="fixed inset-0 z-0"
        onClick={onClose}
      />
      
      <div className="bg-white dark:bg-slate-900 rounded-xl shadow-2xl w-full max-w-2xl z-10 flex flex-col border border-slate-200 dark:border-slate-800 overflow-hidden" style={{ maxHeight: '80vh' }}>
        
        {/* Search Input Area */}
        <div className="flex items-center px-4 py-4 border-b border-slate-200 dark:border-slate-800">
          <Search className="w-5 h-5 text-slate-400 mr-3 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Search messages and conversations..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 bg-transparent border-none outline-none text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 text-lg"
          />
          <button 
            onClick={onClose}
            className="p-1 rounded-md text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 ml-2"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Results Area */}
        <div className="overflow-y-auto flex-1 bg-slate-50 dark:bg-slate-900/50 min-h-[100px]">
          {isLoading && (
            <div className="p-4 text-center text-slate-500 dark:text-slate-400">Searching...</div>
          )}
          
          {!isLoading && query && results.length === 0 && (
            <div className="p-8 text-center text-slate-500 dark:text-slate-400">
              No results found for "{query}"
            </div>
          )}

          {!isLoading && results.length > 0 && (
            <div className="py-2">
              {results.map((conv) => (
                <div 
                  key={conv.id}
                  onClick={() => handleSelect(conv)}
                  className="px-4 py-3 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer flex gap-4 group transition-colors border-b border-slate-100 dark:border-slate-800/50 last:border-0"
                >
                  <div className="pt-1 text-slate-400 group-hover:text-blue-500 shrink-0">
                    <MessageSquare className="w-5 h-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100 truncate pr-4">
                        {renderHighlighted(conv.title, query)}
                      </h4>
                      <span className="text-xs text-slate-500 whitespace-nowrap">
                        {new Date(conv.updated_at).toLocaleDateString()}
                      </span>
                    </div>
                    {conv.snippet && (
                      <p className="text-sm text-slate-600 dark:text-slate-400 line-clamp-2">
                        {renderHighlighted(conv.snippet, query)}
                      </p>
                    )}
                  </div>
                  <div className="flex items-center text-slate-300 dark:text-slate-600 group-hover:text-blue-500 opacity-0 group-hover:opacity-100 transition-opacity">
                    <ChevronRight className="w-5 h-5" />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* Footer */}
        <div className="px-4 py-2 border-t border-slate-200 dark:border-slate-800 text-xs text-slate-500 dark:text-slate-500 bg-white dark:bg-slate-900 flex justify-between">
          <span>Search through all your conversation history</span>
          <span>Press Esc to close</span>
        </div>
      </div>
    </div>
  );
};

export default SearchModal;
