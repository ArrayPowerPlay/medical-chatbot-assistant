import React, { useEffect, useState } from 'react';
import { adminApi } from '../../api/adminApi';
import type { User } from '../../api/adminApi';
import type { Conversation } from '../../api/conversationApi';
import { Trash2, KeyRound, ChevronLeft, MessageSquare, AlertTriangle, ChevronRight, Loader2 } from 'lucide-react';
import AdminConversationView from './AdminConversationView';

const PAGE_SIZE = 20;

const UserManagement: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [totalUsers, setTotalUsers] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  
  // Filter and Search states
  const [searchInput, setSearchInput] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [roleFilter, setRoleFilter] = useState('all');
  
  // View states
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null);
  const [userConversations, setUserConversations] = useState<Conversation[]>([]);
  const [loadingConversations, setLoadingConversations] = useState(false);
  
  // Conversation View states
  const [selectedConversation, setSelectedConversation] = useState<{id: string, title: string} | null>(null);

  // Modals
  const [resetPasswordUserId, setResetPasswordUserId] = useState<number | null>(null);
  const [newPassword, setNewPassword] = useState('');
  const [deleteUserId, setDeleteUserId] = useState<number | null>(null);

  useEffect(() => {
    fetchUsers(offset, searchQuery, roleFilter);
  }, [offset, searchQuery, roleFilter]);

  const fetchUsers = async (currentOffset: number, search: string, role: string) => {
    setLoading(true);
    try {
      const data = await adminApi.getUsers(PAGE_SIZE, currentOffset, search, role);
      setUsers(data.users);
      setTotalUsers(data.total);
    } catch (err: any) {
      console.error(err);
      alert('Failed to load users: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleNextPage = () => {
    if (offset + PAGE_SIZE < totalUsers) {
      setOffset(offset + PAGE_SIZE);
    }
  };

  const handlePrevPage = () => {
    if (offset - PAGE_SIZE >= 0) {
      setOffset(offset - PAGE_SIZE);
    }
  };

  const handleUserClick = async (userId: number) => {
    setSelectedUserId(userId);
    setLoadingConversations(true);
    try {
      const convs = await adminApi.getUserConversations(userId);
      setUserConversations(convs);
    } catch (err: any) {
      alert('Failed to load conversations for user: ' + err.message);
    } finally {
      setLoadingConversations(false);
    }
  };

  const handleDeleteUser = async () => {
    if (deleteUserId !== null) {
      try {
        await adminApi.deleteUser(deleteUserId);
        fetchUsers(offset, searchQuery, roleFilter);
        if (selectedUserId === deleteUserId) {
          setSelectedUserId(null);
        }
      } catch (err: any) {
        alert('Failed to delete user: ' + err.message);
      } finally {
        setDeleteUserId(null);
      }
    }
  };

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (resetPasswordUserId !== null && newPassword.length >= 6) {
      try {
        await adminApi.resetUserPassword(resetPasswordUserId, newPassword);
        alert('Password updated successfully');
      } catch (err: any) {
        alert('Failed to reset password: ' + err.message);
      } finally {
        setResetPasswordUserId(null);
        setNewPassword('');
      }
    } else {
      alert("Password must be at least 6 characters.");
    }
  };

  // Render Sub-view: User Conversations
  if (selectedUserId !== null) {
    const user = users.find(u => u.id === selectedUserId);
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setSelectedUserId(null)}
            className="p-2 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 rounded-full transition-colors"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <div>
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white">Conversations</h2>
            <p className="text-sm text-slate-500">User: {user?.username || user?.email || `ID ${selectedUserId}`}</p>
          </div>
        </div>

        {loadingConversations ? (
          <div className="flex items-center justify-center p-12 text-slate-500">
             <Loader2 className="w-6 h-6 animate-spin mr-2" /> Loading conversations...
          </div>
        ) : (
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden">
            {userConversations.length === 0 ? (
              <div className="p-8 text-center text-slate-500">This user has no conversations.</div>
            ) : (
              <ul className="divide-y divide-slate-200 dark:divide-slate-700">
                {userConversations.map(conv => (
                  <li 
                    key={conv.id} 
                    className="p-4 hover:bg-slate-50 dark:hover:bg-slate-700/50 cursor-pointer flex items-center justify-between group transition-colors"
                    onClick={() => setSelectedConversation({ id: conv.id, title: conv.title })}
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-lg">
                        <MessageSquare className="w-5 h-5" />
                      </div>
                      <div>
                        <h4 className="font-medium text-slate-800 dark:text-white">{conv.title}</h4>
                        <p className="text-xs text-slate-500">{new Date(conv.updated_at).toLocaleString()}</p>
                      </div>
                    </div>
                    <ChevronRight className="w-5 h-5 text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Conversation Viewer Modal */}
        {selectedConversation && (
          <AdminConversationView 
            conversationId={selectedConversation.id}
            conversationTitle={selectedConversation.title}
            username={user?.username}
            email={user?.email || undefined}
            onClose={() => setSelectedConversation(null)}
          />
        )}
      </div>
    );
  }

  // Main View: Users Table
  return (
    <div className="space-y-6">
      
      {/* Delete User Modal */}
      {deleteUserId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="bg-white dark:bg-slate-900 rounded-xl max-w-sm w-full p-6 shadow-xl border border-slate-200 dark:border-slate-800">
            <div className="flex items-center gap-3 text-red-600 mb-4">
              <AlertTriangle className="w-6 h-6" />
              <h3 className="text-lg font-semibold">Delete User?</h3>
            </div>
            <p className="text-slate-600 dark:text-slate-400 mb-6">
              Are you sure you want to permanently delete user ID {deleteUserId}? All their conversations will be lost.
            </p>
            <div className="flex justify-end gap-3">
              <button 
                onClick={() => setDeleteUserId(null)}
                className="px-4 py-2 text-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button 
                onClick={handleDeleteUser}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Reset Password Modal */}
      {resetPasswordUserId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="bg-white dark:bg-slate-900 rounded-xl max-w-sm w-full p-6 shadow-xl border border-slate-200 dark:border-slate-800">
            <h3 className="text-lg font-semibold text-slate-800 dark:text-white mb-4">Reset Password</h3>
            <form onSubmit={handleResetPassword}>
              <div className="mb-6">
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">New Password</label>
                <input 
                  type="password" 
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full px-3 py-2 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Min 6 characters"
                  required
                  minLength={6}
                />
              </div>
              <div className="flex justify-end gap-3">
                <button 
                  type="button"
                  onClick={() => {
                    setResetPasswordUserId(null);
                    setNewPassword('');
                  }}
                  className="px-4 py-2 text-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button 
                  type="submit"
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  Update
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <h2 className="text-2xl font-bold text-slate-800 dark:text-white">User Management</h2>
        
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <input 
              type="text"
              value={searchInput}
              onChange={e => setSearchInput(e.target.value)}
              onKeyDown={e => { if(e.key === 'Enter') { setOffset(0); setSearchQuery(searchInput.trim()); } }}
              placeholder="Search username/email..."
              className="w-48 px-3 py-2 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button 
              onClick={() => { setOffset(0); setSearchQuery(searchInput.trim()); }}
              className="px-4 py-2 bg-slate-200 dark:bg-slate-700 text-slate-800 dark:text-slate-200 rounded-lg text-sm font-medium hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
            >
              Search
            </button>
            {searchQuery && (
              <button 
                onClick={() => { setSearchInput(''); setSearchQuery(''); setOffset(0); }}
                className="px-2 py-2 text-slate-500 hover:text-red-500 text-sm"
              >
                Clear
              </button>
            )}
          </div>
          
          <div className="h-6 w-px bg-slate-300 dark:bg-slate-700 hidden lg:block"></div>
          
          <select 
            value={roleFilter}
            onChange={(e) => { setRoleFilter(e.target.value); setOffset(0); }}
            className="px-3 py-2 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Roles</option>
            <option value="admin">Admin</option>
            <option value="user">User</option>
            <option value="guest">Guest</option>
          </select>
        </div>
      </div>
      
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 dark:bg-slate-900/50 text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-slate-700">
              <tr>
                <th className="px-6 py-4 font-medium">ID</th>
                <th className="px-6 py-4 font-medium">Username / Email</th>
                <th className="px-6 py-4 font-medium">Role</th>
                <th className="px-6 py-4 font-medium">Questions</th>
                <th className="px-6 py-4 font-medium">Joined</th>
                <th className="px-6 py-4 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
              {loading ? (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-center text-slate-500">
                    <div className="flex justify-center items-center"><Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading users...</div>
                  </td>
                </tr>
              ) : users.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-center text-slate-500">No users found</td>
                </tr>
              ) : (
                users.map((user) => (
                  <tr key={user.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                    <td className="px-6 py-4">{user.id}</td>
                    <td className="px-6 py-4 font-medium cursor-pointer text-blue-600 dark:text-blue-400 hover:underline" onClick={() => handleUserClick(user.id)}>
                      {user.username || user.email || 'N/A'}
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium uppercase ${
                        user.role === 'admin' ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400' :
                        user.role === 'guest' ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400' :
                        'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                      }`}>
                        {user.role}
                      </span>
                    </td>
                    <td className="px-6 py-4">{user.question_count}</td>
                    <td className="px-6 py-4 text-slate-500">{new Date(user.created_at).toLocaleDateString()}</td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button 
                          onClick={() => setResetPasswordUserId(user.id)}
                          className="p-2 text-slate-500 hover:text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg transition-colors"
                          title="Reset Password"
                        >
                          <KeyRound className="w-4 h-4" />
                        </button>
                        <button 
                          onClick={() => setDeleteUserId(user.id)}
                          className="p-2 text-slate-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
                          title="Delete User"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        
        {/* Pagination Controls */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/50">
          <p className="text-sm text-slate-500">
            Showing <span className="font-medium">{totalUsers === 0 ? 0 : offset + 1}</span> to <span className="font-medium">{Math.min(offset + PAGE_SIZE, totalUsers)}</span> of <span className="font-medium">{totalUsers}</span> results
          </p>
          <div className="flex gap-2">
            <button
              onClick={handlePrevPage}
              disabled={offset === 0}
              className="px-3 py-1.5 text-sm bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-md hover:bg-slate-50 dark:hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-slate-700 dark:text-slate-300"
            >
              Previous
            </button>
            <button
              onClick={handleNextPage}
              disabled={offset + PAGE_SIZE >= totalUsers}
              className="px-3 py-1.5 text-sm bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-md hover:bg-slate-50 dark:hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-slate-700 dark:text-slate-300"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserManagement;
