import React, { useEffect, useState } from 'react';
import { MessageSquarePlus, X, LogOut, PanelLeftClose, Pin, Trash2, Edit2, Check, Search, AlertTriangle } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import { useConversationStore } from '../../stores/conversationStore';
import { useNavigate, useParams } from 'react-router-dom';

interface SidebarProps {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, setIsOpen }) => {
  const { user, logout } = useAuthStore();
  const { conversations, fetchConversations, searchConversations, deleteConversation, renameConversation, pinConversation } = useConversationStore();
  const navigate = useNavigate();
  const { id: activeId } = useParams<{ id: string }>();

  const [searchQuery, setSearchQuery] = useState('');
  
  // States for rename
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');

  // States for delete popup
  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

  // Debounced search
  useEffect(() => {
    const handler = setTimeout(() => {
      searchConversations(searchQuery);
    }, 300);
    return () => clearTimeout(handler);
  }, [searchQuery, searchConversations]);

  const handleRenameSubmit = async (e: React.FormEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    if (!editTitle.trim()) return;
    try {
      await renameConversation(id, editTitle.trim());
    } finally {
      setEditingId(null);
    }
  };

  const handleConfirmDelete = async () => {
    if (deletingId) {
      await deleteConversation(deletingId);
      if (activeId === deletingId) {
        navigate('/', { replace: true });
      }
      setDeletingId(null);
    }
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-slate-900/50 z-20 md:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Delete Confirmation Modal */}
      {deletingId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="bg-white dark:bg-slate-900 rounded-xl max-w-sm w-full p-6 shadow-xl border border-slate-200 dark:border-slate-800">
            <div className="flex items-center gap-3 text-red-600 dark:text-red-500 mb-4">
              <AlertTriangle className="w-6 h-6" />
              <h3 className="text-lg font-semibold">Delete Chat?</h3>
            </div>
            <p className="text-slate-600 dark:text-slate-400 mb-6">
              Are you sure you want to delete this conversation? This action cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button 
                onClick={() => setDeletingId(null)}
                className="px-4 py-2 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button 
                onClick={handleConfirmDelete}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Sidebar Container */}
      <aside 
        className={`fixed inset-y-0 left-0 z-30 w-72 bg-slate-50 dark:bg-slate-950 border-r border-slate-200 dark:border-slate-800 transform transition-transform duration-300 ease-in-out flex flex-col ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0 md:relative'
        }`}
      >
        {/* Header - New Chat */}
        <div className="p-4 flex items-center justify-between">
          <button 
            className="flex-1 flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors font-medium"
            onClick={() => {
              navigate('/');
              if (window.innerWidth < 768) setIsOpen(false);
            }}
          >
            <MessageSquarePlus className="w-5 h-5" />
            <span>New Chat</span>
          </button>

          <button 
            className="md:hidden ml-2 p-2 text-slate-500 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg transition-colors"
            onClick={() => setIsOpen(false)}
          >
            <X className="w-5 h-5" />
          </button>
          
          <button 
            className="hidden md:flex ml-2 p-2 text-slate-500 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg transition-colors"
            onClick={() => setIsOpen(false)}
            title="Close sidebar"
          >
            <PanelLeftClose className="w-5 h-5" />
          </button>
        </div>

        {/* Search Bar */}
        <div className="px-4 pb-2">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input 
              type="text" 
              placeholder="Search history..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg pl-9 pr-3 py-1.5 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto p-4 pt-2">
          <div className="flex flex-col gap-1">
            {conversations.length === 0 ? (
              <div className="px-3 py-2 text-sm text-slate-400 dark:text-slate-500 italic text-center mt-4">
                {searchQuery ? "No results found" : "No history yet"}
              </div>
            ) : (
              conversations.map(conv => {
                const isActive = activeId === conv.id;
                const isEditing = editingId === conv.id;
                
                return (
                  <div 
                    key={conv.id}
                    onClick={() => {
                      if (!isEditing) {
                        navigate(`/c/${conv.id}`);
                        if (window.innerWidth < 768) setIsOpen(false);
                      }
                    }}
                    className={`group relative flex items-center justify-between px-3 py-2.5 rounded-lg cursor-pointer transition-colors ${
                      isActive 
                        ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-900 dark:text-blue-100' 
                        : 'text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-800/80'
                    }`}
                  >
                    {isEditing ? (
                      <form onSubmit={(e) => handleRenameSubmit(e, conv.id)} className="flex-1 flex items-center mr-2">
                        <input
                          autoFocus
                          type="text"
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          onClick={(e) => e.stopPropagation()}
                          className="w-full bg-white dark:bg-slate-950 border border-blue-500 rounded px-2 py-0.5 text-sm outline-none"
                        />
                        <button type="submit" className="ml-1 p-1 text-green-600 hover:bg-green-100 dark:hover:bg-green-900/50 rounded">
                          <Check className="w-3.5 h-3.5" />
                        </button>
                      </form>
                    ) : (
                      <>
                        <div className="flex-1 min-w-0 pr-6 flex items-center gap-2">
                          {conv.is_pinned && <Pin className="w-3.5 h-3.5 shrink-0 text-blue-500 fill-current" />}
                          <span className="truncate text-sm font-medium">{conv.title}</span>
                        </div>
                        
                        {/* Action Icons (Visible on Hover or Active) */}
                        <div className={`absolute right-2 flex items-center gap-1 bg-gradient-to-l from-slate-200 via-slate-200 to-transparent dark:from-slate-800 dark:via-slate-800 pl-4 ${isActive ? 'from-blue-100 via-blue-100 dark:from-blue-900/40 dark:via-blue-900/40' : 'opacity-0 group-hover:opacity-100'}`}>
                          
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              pinConversation(conv.id, !conv.is_pinned);
                            }}
                            className={`p-1 rounded hover:bg-black/5 dark:hover:bg-white/10 ${conv.is_pinned ? 'text-blue-500' : 'text-slate-400'}`}
                            title={conv.is_pinned ? "Unpin" : "Pin"}
                          >
                            <Pin className="w-3.5 h-3.5" />
                          </button>
                          
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              setEditTitle(conv.title);
                              setEditingId(conv.id);
                            }}
                            className="p-1 text-slate-400 hover:text-blue-500 hover:bg-black/5 dark:hover:bg-white/10 rounded"
                            title="Rename"
                          >
                            <Edit2 className="w-3.5 h-3.5" />
                          </button>

                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              setDeletingId(conv.id);
                            }}
                            className="p-1 text-slate-400 hover:text-red-500 hover:bg-black/5 dark:hover:bg-white/10 rounded"
                            title="Delete"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>

                        </div>
                      </>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Footer - User Profile & Settings */}
        <div className="p-4 border-t border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors cursor-pointer group">
            <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center text-blue-600 dark:text-blue-300 font-bold shrink-0">
              {user?.email?.[0].toUpperCase() || 'G'}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
                {user?.email || 'Guest User'}
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400 capitalize">
                {user?.role || 'Guest'}
              </div>
            </div>
            <button 
              onClick={(e) => {
                e.stopPropagation();
                logout();
              }}
              className="text-slate-400 hover:text-red-500 transition-colors opacity-0 group-hover:opacity-100"
              title="Logout"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
