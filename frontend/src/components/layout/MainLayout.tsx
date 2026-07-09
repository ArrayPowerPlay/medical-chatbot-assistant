import React, { useState, useEffect } from 'react';
import { Outlet, useNavigate } from 'react-router-dom';
import Sidebar from './Sidebar';
import { Menu, Moon, Sun, PanelLeftOpen } from 'lucide-react';
import { useThemeStore } from '../../stores/themeStore';
import logoUrl from '../../assets/logo.png';

const MainLayout: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(() => window.innerWidth >= 768);
  const { theme, toggleTheme } = useThemeStore();
  const isDarkMode = theme === 'dark';
  const navigate = useNavigate();

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  return (
    <div className="flex h-screen bg-white dark:bg-slate-900 overflow-hidden text-slate-800 dark:text-slate-50 selection:bg-blue-200 selection:text-blue-900 dark:selection:bg-blue-800 dark:selection:text-blue-50">
      
      {/* Sidebar */}
      <Sidebar isOpen={isSidebarOpen} setIsOpen={setIsSidebarOpen} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        
        {/* Top Header (Visible mainly on mobile or when sidebar is closed) */}
        <header className="h-14 flex items-center justify-between px-4 border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm z-10 shrink-0">
          <div className="flex items-center gap-2">
            {!isSidebarOpen && (
              <button
                className="p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
                onClick={() => setIsSidebarOpen(true)}
                title="Open sidebar"
              >
                <Menu className="w-5 h-5 md:hidden" />
                <PanelLeftOpen className="w-5 h-5 hidden md:block" />
              </button>
            )}
            <div className={`flex items-center gap-2 cursor-pointer ml-1 ${isSidebarOpen ? 'md:hidden' : ''}`} onClick={() => navigate('/')}>
              <img src={logoUrl} alt="Med Assistant Logo" className="w-7 h-7 object-contain rounded" onError={(e) => e.currentTarget.style.display = 'none'} />
              <h1 className="font-bold text-lg tracking-tight text-blue-700 dark:text-blue-400 hidden sm:block">Med Assistant</h1>
            </div>
          </div>

          {/* Theme Toggle */}
          <button
            className="px-3 py-1.5 rounded-xl transition-all duration-300 hover:scale-105 flex items-center justify-center gap-2 bg-slate-100 text-slate-600 hover:bg-blue-100 hover:text-blue-600 dark:bg-slate-800 dark:text-amber-400 dark:hover:bg-slate-700 shadow-sm border border-slate-200 dark:border-slate-700 text-sm font-medium"
            onClick={toggleTheme}
            title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {isDarkMode ? (
              <><Sun className="w-4 h-4 drop-shadow-sm" /><span className="hidden sm:inline">Light Mode</span></>
            ) : (
              <><Moon className="w-4 h-4" /><span className="hidden sm:inline">Dark Mode</span></>
            )}
          </button>
        </header>

        {/* Chat / Content Area */}
        <main className="flex-1 overflow-hidden relative flex flex-col">
          <Outlet />
        </main>

      </div>
    </div>
  );
};

export default MainLayout;
