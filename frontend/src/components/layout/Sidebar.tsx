import React from 'react';
import { MessageSquarePlus, X, LogOut, PanelLeftClose } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';

interface SidebarProps {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, setIsOpen }) => {
  const { user, logout } = useAuthStore();

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-slate-900/50 z-20 md:hidden"
          onClick={() => setIsOpen(false)}
        />
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
              // TODO: Reset current chat / Navigate to /
              if (window.innerWidth < 768) setIsOpen(false);
            }}
          >
            <MessageSquarePlus className="w-5 h-5" />
            <span>New Chat</span>
          </button>

          {/* Close button for mobile only */}
          <button 
            className="md:hidden ml-2 p-2 text-slate-500 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg transition-colors"
            onClick={() => setIsOpen(false)}
          >
            <X className="w-5 h-5" />
          </button>
          
          {/* Close button for desktop to collapse (optional feature, but we add the UI here) */}
          <button 
            className="hidden md:flex ml-2 p-2 text-slate-500 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg transition-colors"
            onClick={() => setIsOpen(false)}
            title="Close sidebar"
          >
            <PanelLeftClose className="w-5 h-5" />
          </button>
        </div>

        {/* Conversation List Placeholder (Phase 5) */}
        <div className="flex-1 overflow-y-auto p-4 pt-0">
          <div className="text-sm font-semibold text-slate-500 dark:text-slate-400 mb-3 px-2">Recent</div>
          <div className="flex flex-col gap-1">
            {/* We will map history here in Phase 5 */}
            <div className="px-3 py-2 text-sm text-slate-400 dark:text-slate-500 italic">
              History will appear here
            </div>
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
