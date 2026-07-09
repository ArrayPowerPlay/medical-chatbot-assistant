import axiosClient from './axiosClient';

export interface Conversation {
  id: string;
  title: string;
  is_pinned: boolean;
  user_id: number;
  created_at: string;
  updated_at: string;
  matched_message_ids?: number[];
  snippet?: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  feedback_type?: 'like' | 'dislike' | 'none';
  feedback_comment?: string;
  created_at: string;
}

export const conversationApi = {
  getConversations: async () => {
    const response = await axiosClient.get<{ conversations: Conversation[] }>('/api/conversations');
    return response.data;
  },

  getMessages: async (conversationId: string, limit: number = 50, beforeId?: string) => {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (beforeId) params.append('before_id', beforeId);
    
    const response = await axiosClient.get<{ messages: Message[], has_more: boolean }>(`/api/conversations/${conversationId}/messages?${params.toString()}`);
    return response.data;
  },

  renameConversation: async (id: string, title: string) => {
    const response = await axiosClient.put<Conversation>(`/api/conversations/${id}`, { title });
    return response.data;
  },

  pinConversation: async (id: string, isPinned: boolean) => {
    const response = await axiosClient.put<Conversation>(`/api/conversations/${id}/pin`, { is_pinned: isPinned });
    return response.data;
  },

  deleteConversation: async (id: string) => {
    const response = await axiosClient.delete<{ message: string }>(`/api/conversations/${id}`);
    return response.data;
  },

  searchConversations: async (query: string) => {
    const params = new URLSearchParams({ q: query });
    const response = await axiosClient.get<{ conversations: Conversation[] }>(`/api/conversations/search?${params.toString()}`);
    return response.data;
  },

  submitFeedback: async (conversationId: string, messageId: string, type: 'like' | 'dislike' | 'none', comment: string = '') => {
    const response = await axiosClient.post(`/api/conversations/${conversationId}/messages/${messageId}/feedback`, {
      feedback_type: type,
      feedback_comment: comment
    });
    return response.data;
  }
};
