import React from 'react';
import { Outlet, Link, useNavigate } from 'react-router-dom';
import { LogOut, Users, Activity, MessageSquare, Moon, Sun } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import { useThemeStore } from '../../stores/themeStore';

const AdminLayout: React.FC = () => {
  const { logout } = useAuthStore();
  const { theme, toggleTheme } = useThemeStore();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="flex h-screen bg-slate-100 dark:bg-slate-900 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-white dark:bg-slate-800 border-r border-slate-200 dark:border-slate-700 flex flex-col">
        <div className="p-4 border-b border-slate-200 dark:border-slate-700">
          <h1 className="text-xl font-bold text-slate-800 dark:text-white flex items-center">
            <Activity className="mr-2 text-blue-600 dark:text-blue-500" />
            MedKG Admin
          </h1>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          <Link to="/admin" className="flex items-center px-4 py-2 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-md">
            <Activity className="mr-3 w-5 h-5" />
            Dashboard Stats
          </Link>
          <Link to="/admin/users" className="flex items-center px-4 py-2 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-md">
            <Users className="mr-3 w-5 h-5" />
            Manage Users
          </Link>
          <Link to="/admin/conversations" className="flex items-center px-4 py-2 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-md">
            <MessageSquare className="mr-3 w-5 h-5" />
            Conversations
          </Link>
        </nav>
        <div className="p-4 border-t border-slate-200 dark:border-slate-700 flex flex-col space-y-2">
          <button
            onClick={toggleTheme}
            className="flex items-center px-4 py-2 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-md"
          >
            {theme === 'light' ? <Moon className="mr-3 w-5 h-5" /> : <Sun className="mr-3 w-5 h-5" />}
            {theme === 'light' ? 'Dark Mode' : 'Light Mode'}
          </button>
          <button
            onClick={handleLogout}
            className="flex items-center px-4 py-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-md"
          >
            <LogOut className="mr-3 w-5 h-5" />
            Logout
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <header className="bg-white dark:bg-slate-800 shadow-sm p-4">
          <div className="flex justify-between items-center">
            <h2 className="text-lg font-medium text-slate-800 dark:text-white">Admin Dashboard</h2>
            <Link to="/" className="text-sm text-blue-600 dark:text-blue-400 hover:underline">
              Back to Chat
            </Link>
          </div>
        </header>
        <div className="p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default AdminLayout;
