import React from 'react';
import { Outlet, Link, useNavigate, useLocation } from 'react-router-dom';
import { LogOut, Users, Activity, MessageSquare, Moon, Sun } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import { useThemeStore } from '../../stores/themeStore';
import logoUrl from '../../assets/logo.png';

const AdminLayout: React.FC = () => {
  const { user, logout } = useAuthStore();
  const { theme, toggleTheme } = useThemeStore();
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    if (window.confirm('Are you sure you want to logout?')) {
      logout();
      navigate('/login');
    }
  };

  const navItems = [
    { path: '/admin', icon: Activity, label: 'Dashboard Stats', exact: true },
    { path: '/admin/users', icon: Users, label: 'Manage Users', exact: false },
    { path: '/admin/feedback', icon: MessageSquare, label: 'Feedback & Quality', exact: false }
  ];

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-900 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-white dark:bg-slate-800 border-r border-slate-200 dark:border-slate-800 flex flex-col shrink-0 shadow-sm z-10">
        <div className="p-4 h-16 border-b border-slate-200 dark:border-slate-800 flex items-center gap-3 cursor-pointer shrink-0" onClick={() => navigate('/')}>
          <img src={logoUrl} alt="Med Assistant Logo" className="w-8 h-8 object-contain rounded" onError={(e) => e.currentTarget.style.display = 'none'} />
          <h1 className="text-xl font-bold text-slate-800 dark:text-white flex items-center tracking-tight">
            MedKG Admin
          </h1>
        </div>
        
        <nav className="flex-1 p-4 space-y-1.5 overflow-y-auto">
          {navItems.map((item) => {
            const isActive = item.exact 
              ? location.pathname === item.path 
              : location.pathname.startsWith(item.path);
              
            const Icon = item.icon;
            
            return (
              <Link 
                key={item.path}
                to={item.path} 
                className={`flex items-center px-4 py-2.5 rounded-lg transition-all duration-200 font-medium border-l-4 ${
                  isActive 
                    ? 'border-blue-600 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 shadow-sm' 
                    : 'border-transparent text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700/50 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                <Icon className={`mr-3 w-5 h-5 transition-colors ${isActive ? 'text-blue-600 dark:text-blue-400' : 'text-slate-400 dark:text-slate-500 group-hover:text-slate-600 dark:group-hover:text-slate-400'}`} />
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* Footer - User Profile & Settings */}
        <div className="p-4 border-t border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors cursor-pointer group">
            <div className="w-8 h-8 rounded-full bg-blue-600 dark:bg-blue-500 flex items-center justify-center text-white font-bold shrink-0">
              {(user?.email?.[0] || user?.username?.[0] || 'A').toUpperCase()}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
                {user?.email || user?.username || 'Admin User'}
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400 capitalize">
                {user?.role || 'Admin'}
              </div>
            </div>
            {/* Always visible logout button */}
            <button 
              onClick={(e) => {
                e.stopPropagation();
                handleLogout();
              }}
              className="text-slate-400 hover:text-red-500 transition-colors p-1"
              title="Logout"
            >
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden flex flex-col relative z-0">
        {/* Header */}
        <header className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-800 h-16 px-6 flex justify-between items-center shrink-0 shadow-sm">
          <h2 className="text-xl font-semibold text-slate-800 dark:text-white flex items-center">
            {navItems.find(i => (i.exact ? location.pathname === i.path : location.pathname.startsWith(i.path)))?.label || 'Admin Dashboard'}
          </h2>
          
          <div className="flex items-center gap-4">
            <button
              className="px-3 py-1.5 rounded-xl transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2 bg-slate-100 text-slate-600 hover:bg-blue-100 hover:text-blue-600 dark:bg-slate-800 dark:text-amber-400 dark:hover:bg-slate-700 shadow-sm border border-slate-200 dark:border-slate-700 text-sm font-medium"
              onClick={toggleTheme}
              title={theme === 'light' ? "Switch to Light Mode" : "Switch to Dark Mode"}
            >
              {theme === 'light' ? (
                <><Moon className="w-4 h-4" /><span className="hidden sm:inline">Dark Mode</span></>
              ) : (
                <><Sun className="w-4 h-4 drop-shadow-sm" /><span className="hidden sm:inline">Light Mode</span></>
              )}
            </button>
          </div>
        </header>
        
        {/* Page Content */}
        <div className="p-6 flex-1 overflow-y-auto bg-slate-50/50 dark:bg-slate-900/50">
          <div className="max-w-7xl mx-auto">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  );
};

export default AdminLayout;
