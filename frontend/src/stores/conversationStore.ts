import { create } from 'zustand';
import { conversationApi } from '../api/conversationApi';
import type { Conversation } from '../api/conversationApi';

interface ConversationState {
  conversations: Conversation[];
  isLoading: boolean;
  error: string | null;
  fetchConversations: () => Promise<void>;
  searchConversations: (query: string) => Promise<void>;
  deleteConversation: (id: string) => Promise<void>;
  renameConversation: (id: string, newTitle: string) => Promise<void>;
  pinConversation: (id: string, isPinned: boolean) => Promise<void>;
  addConversation: (conv: Conversation) => void;
}

export const useConversationStore = create<ConversationState>((set, get) => ({
  conversations: [],
  isLoading: false,
  error: null,

  fetchConversations: async () => {
    set({ isLoading: true, error: null });
    try {
      const data = await conversationApi.getConversations();
      // Ensure pinned ones are on top
      const sorted = data.conversations.sort((a, b) => {
        if (a.is_pinned === b.is_pinned) {
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        }
        return a.is_pinned ? -1 : 1;
      });
      set({ conversations: sorted, isLoading: false });
    } catch (err: any) {
      set({ error: err.message || 'Failed to load conversations', isLoading: false });
    }
  },

  searchConversations: async (query: string) => {
    if (!query.trim()) {
      return get().fetchConversations();
    }
    set({ isLoading: true, error: null });
    try {
      const data = await conversationApi.searchConversations(query);
      set({ conversations: data.conversations, isLoading: false });
    } catch (err: any) {
      set({ error: err.message || 'Search failed', isLoading: false });
    }
  },

  deleteConversation: async (id: string) => {
    try {
      await conversationApi.deleteConversation(id);
      set((state) => ({
        conversations: state.conversations.filter(c => c.id !== id)
      }));
    } catch (err: any) {
      console.error('Failed to delete', err);
      throw err;
    }
  },

  renameConversation: async (id: string, newTitle: string) => {
    try {
      // Optimistic update
      set((state) => ({
        conversations: state.conversations.map(c => c.id === id ? { ...c, title: newTitle } : c)
      }));
      await conversationApi.renameConversation(id, newTitle);
    } catch (err: any) {
      console.error('Failed to rename', err);
      // Revert could be implemented here by refetching, but let's keep it simple
      get().fetchConversations();
      throw err;
    }
  },

  pinConversation: async (id: string, isPinned: boolean) => {
    try {
      // Optimistic update + resort
      set((state) => {
        const updated = state.conversations.map(c => c.id === id ? { ...c, is_pinned: isPinned } : c);
        const sorted = updated.sort((a, b) => {
          if (a.is_pinned === b.is_pinned) {
            return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
          }
          return a.is_pinned ? -1 : 1;
        });
        return { conversations: sorted };
      });
      await conversationApi.pinConversation(id, isPinned);
    } catch (err: any) {
      console.error('Failed to pin', err);
      get().fetchConversations();
      throw err;
    }
  },

  addConversation: (conv: Conversation) => {
    set((state) => {
      const exists = state.conversations.some(c => c.id === conv.id);
      if (exists) return state;
      const updated = [conv, ...state.conversations];
      const sorted = updated.sort((a, b) => {
        if (a.is_pinned === b.is_pinned) {
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        }
        return a.is_pinned ? -1 : 1;
      });
      return { conversations: sorted };
    });
  }
}));
