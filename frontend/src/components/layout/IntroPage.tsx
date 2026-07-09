import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { BrainCircuit, Search, HeartPulse, Sun, Moon, LogOut } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import { useThemeStore } from '../../stores/themeStore';
import logoUrl from '../../assets/logo.png'; // Assuming logo is placed here

const IntroPage: React.FC = () => {
  const navigate = useNavigate();
  const { user, token, logout } = useAuthStore();
  const isAuthenticated = !!token;
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
    <div className="flex-1 overflow-y-auto bg-slate-50 dark:bg-slate-950 flex flex-col min-h-screen text-slate-800 dark:text-slate-50">
      
      {/* Header */}
      <header className="h-16 px-6 flex items-center justify-between border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => navigate('/')}>
          <img src={logoUrl} alt="Med Assistant Logo" className="w-8 h-8 object-contain rounded" onError={(e) => e.currentTarget.style.display = 'none'} />
          <h1 className="font-bold text-xl tracking-tight text-blue-700 dark:text-blue-400">Med Assistant</h1>
        </div>
        
        <div className="flex items-center gap-4">
          <button
            className="p-2 rounded-xl transition-all duration-300 hover:scale-105 flex items-center justify-center bg-slate-100 text-slate-600 hover:bg-blue-100 hover:text-blue-600 dark:bg-slate-800 dark:text-amber-400 dark:hover:bg-slate-700 shadow-sm border border-slate-200 dark:border-slate-700"
            onClick={toggleTheme}
            title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {isDarkMode ? <Sun className="w-5 h-5 drop-shadow-sm" /> : <Moon className="w-5 h-5" />}
          </button>
          
          {isAuthenticated ? (
            <div className="flex items-center gap-3">
              {user?.email && (
                <span className="hidden md:inline-block text-sm text-slate-600 dark:text-slate-400">
                  Hi, {user.email}
                </span>
              )}
              <button 
                onClick={() => navigate('/c')}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors text-sm"
              >
                Go to Chat
              </button>
              <button 
                onClick={logout}
                className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 border border-red-200 dark:border-red-900/50 rounded-lg transition-colors"
                title="Log Out"
              >
                <LogOut className="w-4 h-4" />
                <span>Log Out</span>
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <button 
                onClick={() => navigate('/login')}
                className="px-4 py-2 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-lg font-medium transition-colors text-sm"
              >
                Sign in
              </button>
              <button 
                onClick={() => navigate('/register')}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors text-sm"
              >
                Sign up
              </button>
            </div>
          )}
        </div>
      </header>

      <div className="flex-1 p-6 md:p-12 relative flex flex-col justify-center max-w-5xl mx-auto w-full">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-6xl font-extrabold text-blue-700 dark:text-blue-400 mb-6 leading-tight">
            Med Assistant: Fast Medical Insights with KG-RAG
          </h1>
          <p className="text-lg md:text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            Your intelligent healthcare companion. Search diseases, symptoms, and treatments powered by an advanced medical Knowledge Graph and real-time Retrieval-Augmented Generation.
          </p>
          
          <button 
            onClick={() => navigate(isAuthenticated ? '/c' : '/login')}
            className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-full font-semibold text-lg transition-all transform hover:scale-105 shadow-lg hover:shadow-blue-500/30"
          >
            Start Chatting
          </button>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8">
          {/* Feature 1 */}
          <div className="bg-white dark:bg-slate-900 p-8 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-800 hover:shadow-md transition-shadow">
            <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/50 rounded-xl flex items-center justify-center text-blue-600 dark:text-blue-400 mb-6">
              <Search className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-3">Accurate Retrieval</h3>
            <p className="text-slate-600 dark:text-slate-400">
              Powered by advanced Vector and BM25 hybrid search to find the exact medical evidence you need.
            </p>
          </div>

          {/* Feature 2 */}
          <div className="bg-white dark:bg-slate-900 p-8 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-800 hover:shadow-md transition-shadow">
            <div className="w-12 h-12 bg-cyan-100 dark:bg-cyan-900/50 rounded-xl flex items-center justify-center text-cyan-600 dark:text-cyan-400 mb-6">
              <BrainCircuit className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-3">Knowledge Graph</h3>
            <p className="text-slate-600 dark:text-slate-400">
              Integrated entity relationships mapping diseases, symptoms, and drugs for comprehensive answers.
            </p>
          </div>

          {/* Feature 3 */}
          <div className="bg-white dark:bg-slate-900 p-8 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-800 hover:shadow-md transition-shadow">
            <div className="w-12 h-12 bg-emerald-100 dark:bg-emerald-900/50 rounded-xl flex items-center justify-center text-emerald-600 dark:text-emerald-400 mb-6">
              <HeartPulse className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-3">Seamless UI</h3>
            <p className="text-slate-600 dark:text-slate-400">
              Real-time interactive Q&A with smart context retention to guide you through medical literature.
            </p>
          </div>
        </div>
      </div>
      
      {/* Footer info */}
      <div className="mt-16 text-center text-slate-500 text-sm">
        <p>&copy; {new Date().getFullYear()} Med Assistant. For research and informational purposes only.</p>
      </div>
    </div>
  );
};

export default IntroPage;
