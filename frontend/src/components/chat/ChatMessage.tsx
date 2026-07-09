import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { User, Bot, ThumbsUp, ThumbsDown, Check, X, BookOpen, ChevronUp, ChevronDown } from 'lucide-react';
import type { ChatMessageData } from '../../hooks/useChatStream';
import { conversationApi } from '../../api/conversationApi';
import { useParams } from 'react-router-dom';

interface ChatMessageProps {
  message: ChatMessageData;
  isThinking?: boolean;
  isHighlighted?: boolean;
  isActiveMatch?: boolean;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isThinking, isHighlighted, isActiveMatch }) => {
  const isUser = message.role === 'user';
  const { id: conversationId } = useParams<{ id: string }>();
  const messageRef = useRef<HTMLDivElement>(null);

  const [feedbackType, setFeedbackType] = useState<'like' | 'dislike' | 'none'>(message.feedback_type || 'none');
  const [showCommentBox, setShowCommentBox] = useState(false);
  const [feedbackComment, setFeedbackComment] = useState(message.feedback_comment || '');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const [isFeedbackSubmitted, setIsFeedbackSubmitted] = useState(!!message.feedback_type && message.feedback_type !== 'none');

  useEffect(() => {
    setFeedbackType(message.feedback_type || 'none');
    setFeedbackComment(message.feedback_comment || '');
    setIsFeedbackSubmitted(!!message.feedback_type && message.feedback_type !== 'none');
  }, [message.feedback_type, message.feedback_comment]);

  useEffect(() => {
    if (isActiveMatch && messageRef.current) {
      messageRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [isActiveMatch]);

  const handleFeedbackClick = (type: 'like' | 'dislike') => {
    if (isFeedbackSubmitted) return;
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
      setIsFeedbackSubmitted(true);
    } catch (error) {
      console.error("Failed to submit feedback", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Base styling for user vs assistant
  const baseBg = isUser ? 'bg-white dark:bg-slate-900' : 'bg-slate-50 dark:bg-slate-900/50';
  // Highlight styling
  const highlightBg = isActiveMatch 
    ? 'bg-amber-100/80 dark:bg-amber-900/40 ring-2 ring-amber-400 dark:ring-amber-600' 
    : isHighlighted 
      ? 'bg-amber-50/50 dark:bg-amber-900/20' 
      : baseBg;

  return (
    <div 
      ref={messageRef}
      className={`flex gap-4 p-4 md:p-6 lg:px-8 max-w-4xl mx-auto w-full group transition-colors duration-500 ${highlightBg}`}
    >
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
          <div className="flex items-center gap-3 text-slate-500 dark:text-slate-400 mt-2 py-1">
            <div className="w-5 h-5 border-2 border-blue-500 border-t-green-500 rounded-full animate-spin"></div>
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

        {/* Sources Section */}
        {!isThinking && message.sources && message.sources.length > 0 && (
          <div className="mt-4">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 text-sm font-medium text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
            >
              <BookOpen className="w-4 h-4" />
              <span>View relevant sources ({message.sources.length})</span>
              {showSources ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            
            {showSources && (
              <div className="mt-3 flex flex-col gap-2">
                {message.sources.map((src: any, idx: number) => (
                  <div key={idx} className="bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 p-3 rounded-lg text-sm text-slate-700 dark:text-slate-300">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold text-blue-600 dark:text-blue-400">
                        {src.source_type === 'kg' ? 'Knowledge Graph' : 'Document'}
                      </span>
                      {src.pmid && (
                        <span className="text-xs px-2 py-0.5 bg-slate-100 dark:bg-slate-800 rounded-full">
                          PMID: {src.pmid}
                        </span>
                      )}
                      {src.score && (
                        <span className="text-xs px-2 py-0.5 bg-slate-100 dark:bg-slate-800 rounded-full">
                          Score: {src.score.toFixed(3)}
                        </span>
                      )}
                    </div>
                    <div className="line-clamp-3 hover:line-clamp-none transition-all">
                      {src.content}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Feedback UI (Only for Assistant & fully generated) */}
        {!isUser && !isThinking && conversationId && (
          <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-800 flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <button
                onClick={() => handleFeedbackClick('like')}
                disabled={isFeedbackSubmitted || feedbackType === 'like'}
                className={`p-1.5 rounded transition-colors ${
                  feedbackType === 'like' 
                    ? 'text-green-600 bg-green-100 dark:bg-green-900/40' 
                    : 'text-slate-400 hover:text-green-600 hover:bg-slate-200 dark:hover:bg-slate-800'
                } ${isFeedbackSubmitted ? 'opacity-75 cursor-not-allowed' : ''}`}
                title="Good response"
              >
                <ThumbsUp className="w-4 h-4" />
              </button>
              
              <button
                onClick={() => handleFeedbackClick('dislike')}
                disabled={isFeedbackSubmitted || feedbackType === 'dislike'}
                className={`p-1.5 rounded transition-colors ${
                  feedbackType === 'dislike' 
                    ? 'text-red-600 bg-red-100 dark:bg-red-900/40' 
                    : 'text-slate-400 hover:text-red-600 hover:bg-slate-200 dark:hover:bg-slate-800'
                } ${isFeedbackSubmitted ? 'opacity-75 cursor-not-allowed' : ''}`}
                title="Bad response"
              >
                <ThumbsDown className="w-4 h-4" />
              </button>
              
              {isFeedbackSubmitted && (
                <span className="text-xs text-green-600 dark:text-green-400 ml-2 font-medium flex items-center gap-1">
                  <Check className="w-3.5 h-3.5" />
                  Thank you for your feedback!
                </span>
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
