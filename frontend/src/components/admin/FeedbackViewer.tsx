import React, { useEffect, useState } from 'react';
import { adminApi } from '../../api/adminApi';
import type { FeedbackMessage } from '../../api/adminApi';
import { ThumbsUp, ThumbsDown, MessageSquare, ChevronRight, Loader2 } from 'lucide-react';
import AdminConversationView from './AdminConversationView';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const PAGE_SIZE = 20;

const FeedbackViewer: React.FC = () => {
  const [tab, setTab] = useState<'bad' | 'good'>('bad');
  const [feedbacks, setFeedbacks] = useState<FeedbackMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [offset, setOffset] = useState(0);
  
  // Modal state
  const [selectedConversation, setSelectedConversation] = useState<{id: string, title: string, username?: string, email?: string, highlightId?: number} | null>(null);

  useEffect(() => {
    setOffset(0);
    fetchFeedback(0, tab);
  }, [tab]);

  const fetchFeedback = async (currentOffset: number, currentTab: 'bad' | 'good') => {
    setLoading(true);
    try {
      if (currentTab === 'bad') {
        const data = await adminApi.getBadFeedback(PAGE_SIZE, currentOffset);
        if (currentOffset === 0) setFeedbacks(data);
        else setFeedbacks(prev => [...prev, ...data]);
      } else {
        const data = await adminApi.getGoodFeedback(PAGE_SIZE, currentOffset);
        if (currentOffset === 0) setFeedbacks(data);
        else setFeedbacks(prev => [...prev, ...data]);
      }
    } catch (err: any) {
      console.error(err);
      alert('Failed to load feedback: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadMore = () => {
    const nextOffset = offset + PAGE_SIZE;
    setOffset(nextOffset);
    fetchFeedback(nextOffset, tab);
  };

  return (
    <div className="space-y-6">
      
      {/* Conversation Viewer Modal */}
      {selectedConversation && (
        <AdminConversationView 
          conversationId={selectedConversation.id}
          conversationTitle={selectedConversation.title}
          username={selectedConversation.username}
          email={selectedConversation.email}
          highlightMessageId={selectedConversation.highlightId}
          onClose={() => setSelectedConversation(null)}
        />
      )}

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-2xl font-bold text-slate-800 dark:text-white">Feedback & Quality</h2>
        
        {/* Tabs */}
        <div className="flex p-1 bg-slate-200 dark:bg-slate-800 rounded-lg max-w-sm">
          <button
            onClick={() => setTab('bad')}
            className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              tab === 'bad' 
                ? 'bg-white dark:bg-slate-700 text-red-600 shadow-sm' 
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <ThumbsDown className="w-4 h-4" /> Negative
          </button>
          <button
            onClick={() => setTab('good')}
            className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              tab === 'good' 
                ? 'bg-white dark:bg-slate-700 text-emerald-600 shadow-sm' 
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <ThumbsUp className="w-4 h-4" /> Positive
          </button>
        </div>
      </div>

      <div className="space-y-4">
        {feedbacks.length === 0 && !loading && (
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-12 text-center text-slate-500">
            No feedback found in this category.
          </div>
        )}
        
        {feedbacks.map((fb) => (
          <div 
            key={fb.id}
            className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden hover:border-blue-300 dark:hover:border-blue-700 transition-colors cursor-pointer group"
            onClick={() => setSelectedConversation({ 
              id: fb.conversation_id, 
              title: fb.conversation_title,
              username: fb.username,
              email: fb.email,
              highlightId: fb.id
            })}
          >
            <div className="p-5">
              <div className="flex items-start justify-between gap-4 mb-3">
                <div className="flex items-center gap-2 text-sm text-slate-500">
                  <MessageSquare className="w-4 h-4" />
                  <span className="font-medium text-slate-700 dark:text-slate-300">{fb.conversation_title}</span>
                  <span className="text-slate-400">&bull; {new Date(fb.created_at).toLocaleString()}</span>
                </div>
                {tab === 'bad' ? (
                  <span className="px-2 py-1 bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 rounded text-xs font-medium uppercase shrink-0">Disliked</span>
                ) : (
                  <span className="px-2 py-1 bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 rounded text-xs font-medium uppercase shrink-0">Liked</span>
                )}
              </div>
              
              <div className="bg-slate-50 dark:bg-slate-900/50 p-4 rounded-lg border border-slate-100 dark:border-slate-800 mb-3 max-h-32 overflow-hidden relative">
                <div className="prose dark:prose-invert max-w-none text-sm text-slate-700 dark:text-slate-300">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {fb.content}
                  </ReactMarkdown>
                </div>
                <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-slate-50 dark:from-slate-900/90 to-transparent"></div>
              </div>

              {fb.feedback_comment ? (
                <div className="mt-3 text-sm flex items-start gap-2 text-slate-600 dark:text-slate-400 bg-amber-50 dark:bg-amber-900/10 p-3 rounded-lg border border-amber-100 dark:border-amber-900/30">
                  <span className="font-semibold text-amber-700 dark:text-amber-500">User Comment:</span>
                  <span className="italic">"{fb.feedback_comment}"</span>
                </div>
              ) : (
                <div className="mt-3 text-sm text-slate-400 italic">No comment provided.</div>
              )}
            </div>
            <div className="bg-slate-50 dark:bg-slate-900/50 px-5 py-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-end text-sm text-blue-600 dark:text-blue-400 font-medium">
              View full conversation <ChevronRight className="w-4 h-4 ml-1 opacity-0 group-hover:opacity-100 transition-opacity transform group-hover:translate-x-1" />
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-center p-8 text-slate-500">
            <Loader2 className="w-6 h-6 animate-spin mr-2" /> Loading feedback...
          </div>
        )}

        {!loading && feedbacks.length >= PAGE_SIZE && feedbacks.length % PAGE_SIZE === 0 && (
          <div className="flex justify-center mt-6">
            <button
              onClick={loadMore}
              className="px-6 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-700 rounded-lg text-sm font-medium transition-colors"
            >
              Load More
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default FeedbackViewer;
