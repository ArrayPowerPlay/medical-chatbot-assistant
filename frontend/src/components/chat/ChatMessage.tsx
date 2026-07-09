import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { User, Bot, ThumbsUp, ThumbsDown, Check, X } from 'lucide-react';
import type { ChatMessageData } from '../../hooks/useChatStream';
import { conversationApi } from '../../api/conversationApi';
import { useParams } from 'react-router-dom';

interface ChatMessageProps {
  message: ChatMessageData;
  isThinking?: boolean;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isThinking }) => {
  const isUser = message.role === 'user';
  const { id: conversationId } = useParams<{ id: string }>();

  const [feedbackType, setFeedbackType] = useState<'like' | 'dislike' | 'none'>(message.feedback_type || 'none');
  const [showCommentBox, setShowCommentBox] = useState(false);
  const [feedbackComment, setFeedbackComment] = useState(message.feedback_comment || '');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleFeedbackClick = (type: 'like' | 'dislike') => {
    if (feedbackType === type) return; // already set
    setFeedbackType(type);
    setShowCommentBox(true);
  };

  const submitFeedback = async () => {
    if (!conversationId) return;
    setIsSubmitting(true);
    try {
      await conversationApi.submitFeedback(conversationId, message.id, feedbackType, feedbackComment);
      setShowCommentBox(false);
    } catch (error) {
      console.error("Failed to submit feedback", error);
    } finally {
      setIsSubmitting(false);
    }
  };

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

        {/* Feedback UI (Only for Assistant & fully generated) */}
        {!isUser && !isThinking && conversationId && (
          <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-800 flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <button
                onClick={() => handleFeedbackClick('like')}
                disabled={feedbackType === 'like'}
                className={`p-1.5 rounded transition-colors ${
                  feedbackType === 'like' 
                    ? 'text-green-600 bg-green-100 dark:bg-green-900/40' 
                    : 'text-slate-400 hover:text-green-600 hover:bg-slate-200 dark:hover:bg-slate-800'
                }`}
                title="Good response"
              >
                <ThumbsUp className="w-4 h-4" />
              </button>
              
              <button
                onClick={() => handleFeedbackClick('dislike')}
                disabled={feedbackType === 'dislike'}
                className={`p-1.5 rounded transition-colors ${
                  feedbackType === 'dislike' 
                    ? 'text-red-600 bg-red-100 dark:bg-red-900/40' 
                    : 'text-slate-400 hover:text-red-600 hover:bg-slate-200 dark:hover:bg-slate-800'
                }`}
                title="Bad response"
              >
                <ThumbsDown className="w-4 h-4" />
              </button>
              
              {(feedbackType === 'like' || feedbackType === 'dislike') && !showCommentBox && (
                <span className="text-xs text-slate-500 dark:text-slate-400 ml-2 italic">Feedback submitted</span>
              )}
            </div>

            {/* Comment Box */}
            {showCommentBox && (
              <div className="flex flex-col gap-2 bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-700 p-3 rounded-lg w-full max-w-sm">
                <textarea 
                  value={feedbackComment}
                  onChange={(e) => setFeedbackComment(e.target.value)}
                  placeholder="Tell us more about your feedback... (optional)"
                  className="w-full bg-transparent text-sm text-slate-800 dark:text-slate-200 outline-none resize-none h-16"
                />
                <div className="flex justify-end gap-2">
                  <button 
                    onClick={() => {
                      setShowCommentBox(false);
                      // Revert to original if cancelled
                      setFeedbackType(message.feedback_type || 'none');
                    }}
                    className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 rounded transition-colors"
                    title="Cancel"
                  >
                    <X className="w-4 h-4" />
                  </button>
                  <button 
                    onClick={submitFeedback}
                    disabled={isSubmitting}
                    className="p-1 text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded transition-colors"
                    title="Submit"
                  >
                    <Check className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
