import React, { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import { Menu, Moon, Sun, PanelLeftOpen } from 'lucide-react';
import { useThemeStore } from '../../stores/themeStore';

const MainLayout: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const { theme, toggleTheme } = useThemeStore();
  const isDarkMode = theme === 'dark';

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
                {/* Use Menu icon on mobile, PanelLeftOpen on desktop */}
                <Menu className="w-5 h-5 md:hidden" />
                <PanelLeftOpen className="w-5 h-5 hidden md:block" />
              </button>
            )}
            <h1 className="font-semibold text-lg md:hidden">MedKG-RAG</h1>
          </div>

          {/* Theme Toggle */}
          <button
            className="p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
            onClick={toggleTheme}
            title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
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
