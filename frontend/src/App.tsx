import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useThemeStore } from './stores/themeStore';

// Layouts
import MainLayout from './components/layout/MainLayout';
import AdminLayout from './components/layout/AdminLayout';
import ProtectedRoute from './components/layout/ProtectedRoute';

// Auth
import Login from './components/auth/Login';
import Register from './components/auth/Register';

import ChatLayout from './components/chat/ChatLayout';

import IntroPage from './components/layout/IntroPage';

// Admin Components
import DashboardStats from './components/admin/DashboardStats';
import UserManagement from './components/admin/UserManagement';
import FeedbackViewer from './components/admin/FeedbackViewer';

function App() {
  const { theme } = useThemeStore();

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  return (
    <BrowserRouter>
      <Routes>
        {/* Public Routes */}
        <Route path="/" element={<IntroPage />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Protected Routes (User & Guest) */}
        <Route element={<ProtectedRoute allowedRoles={['user', 'guest']} />}>
          <Route element={<MainLayout />}>
            <Route path="/chat" element={<Navigate to="/c" replace />} />
            <Route path="/c/:id?" element={<ChatLayout />} />
          </Route>
        </Route>

        {/* Protected Admin Routes */}
        <Route element={<ProtectedRoute allowedRoles={['admin']} />}>
          <Route path="/admin" element={<AdminLayout />}>
            <Route index element={<DashboardStats />} />
            <Route path="users" element={<UserManagement />} />
            <Route path="feedback" element={<FeedbackViewer />} />
          </Route>
        </Route>

        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
