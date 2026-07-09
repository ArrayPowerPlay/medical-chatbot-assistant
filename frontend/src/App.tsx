import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

// Layouts
import MainLayout from './components/layout/MainLayout';
import AdminLayout from './components/layout/AdminLayout';
import ProtectedRoute from './components/layout/ProtectedRoute';

// Auth
import Login from './components/auth/Login';
import Register from './components/auth/Register';

import ChatLayout from './components/chat/ChatLayout';

// Placeholders for future phases
const AdminDashboard = () => <div>Admin Dashboard Content</div>;

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public Routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Protected Routes (User & Guest) */}
        <Route element={<ProtectedRoute allowedRoles={['user', 'guest', 'admin']} />}>
          <Route path="/" element={<MainLayout />}>
            <Route index element={<ChatLayout />} />
            {/* Future route: <Route path="/c/:id" element={<ChatPlaceholder />} /> */}
          </Route>
        </Route>

        {/* Protected Admin Routes */}
        <Route element={<ProtectedRoute allowedRoles={['admin']} />}>
          <Route path="/admin" element={<AdminLayout />}>
            <Route index element={<AdminDashboard />} />
            <Route path="users" element={<div>Manage Users UI</div>} />
            <Route path="conversations" element={<div>Conversations UI</div>} />
          </Route>
        </Route>

        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
