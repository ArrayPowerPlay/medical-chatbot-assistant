import axiosClient from './axiosClient';
import type { Conversation, Message } from './conversationApi';

export interface AdminStats {
  total_users: number;
  total_guests: number;
  new_users_week: number;
  total_questions: number;
  questions_24h: number;
  total_likes: number;
  total_dislikes: number;
}

export interface User {
  id: number;
  username: string;
  email: string | null;
  role: string;
  question_count: number;
  created_at: string;
}

export interface UserPageResponse {
  total: number;
  users: User[];
}

export interface FeedbackMessage {
  id: number;
  conversation_id: string;
  content: string;
  feedback_comment: string | null;
  created_at: string;
  conversation_title: string;
  username?: string;
  email?: string;
}

export const adminApi = {
  getStats: async () => {
    const response = await axiosClient.get<AdminStats>('/api/admin/stats');
    return response.data;
  },

  getUsers: async (limit: number = 20, offset: number = 0, search?: string, role?: string) => {
    const params = new URLSearchParams({ limit: limit.toString(), offset: offset.toString() });
    if (search) params.append('search', search);
    if (role && role !== 'all') params.append('role', role);
    const response = await axiosClient.get<UserPageResponse>(`/api/admin/users?${params.toString()}`);
    return response.data;
  },

  deleteUser: async (userId: number) => {
    await axiosClient.delete(`/api/admin/users/${userId}`);
  },

  resetUserPassword: async (userId: number, newPassword: string) => {
    const response = await axiosClient.put<{ message: string }>(`/api/admin/users/${userId}/password`, {
      new_password: newPassword,
    });
    return response.data;
  },

  getBadFeedback: async (limit: number = 20, offset: number = 0) => {
    const params = new URLSearchParams({ limit: limit.toString(), offset: offset.toString() });
    const response = await axiosClient.get<FeedbackMessage[]>(`/api/admin/feedback/bad?${params.toString()}`);
    return response.data;
  },

  getGoodFeedback: async (limit: number = 20, offset: number = 0) => {
    const params = new URLSearchParams({ limit: limit.toString(), offset: offset.toString() });
    const response = await axiosClient.get<FeedbackMessage[]>(`/api/admin/feedback/good?${params.toString()}`);
    return response.data;
  },

  getUserConversations: async (userId: number) => {
    const response = await axiosClient.get<Conversation[]>(`/api/admin/users/${userId}/conversations`);
    return response.data;
  },

  getConversationMessages: async (conversationId: string, limit: number = 50, beforeId?: string) => {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (beforeId) params.append('before_id', beforeId);
    
    const response = await axiosClient.get<{ messages: Message[], has_more: boolean }>(`/api/admin/conversations/${conversationId}/messages?${params.toString()}`);
    return response.data;
  }
};
