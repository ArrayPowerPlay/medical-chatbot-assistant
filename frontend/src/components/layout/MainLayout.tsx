import React from 'react';
import { Outlet } from 'react-router-dom';

const MainLayout: React.FC = () => {
  return (
    <div className="flex h-screen bg-white dark:bg-slate-900 overflow-hidden text-slate-800 dark:text-slate-50">
      {/* Sidebar will be built in Phase 4/5, for now we just render Outlet */}
      <main className="flex-1 overflow-hidden relative">
        <Outlet />
      </main>
    </div>
  );
};

export default MainLayout;
