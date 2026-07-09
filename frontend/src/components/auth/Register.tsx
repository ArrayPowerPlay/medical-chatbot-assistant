import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Lock, UserPlus, User } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import { GoogleLogin } from '@react-oauth/google';
import axiosClient from '../../api/axiosClient';
import logoUrl from '../../assets/logo.png';

const Register: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { setAuth } = useAuthStore();
  const navigate = useNavigate();

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters long');
      return;
    }

    if (!/\d/.test(password)) {
      setError('Password must contain at least one number');
      return;
    }

    if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
      setError('Password must contain at least one special character');
      return;
    }

    setLoading(true);

    try {
      // Create user
      await axiosClient.post('/api/auth/register', { 
        username, 
        password, 
        confirm_password: confirmPassword 
      });
      
      // Auto login after register
      const response = await axiosClient.post('/api/auth/login', { username, password });
      setAuth(response.data.access_token, response.data.user);
      
      navigate('/c');
    } catch (err: any) {
      let errorMessage = 'Failed to register. Please try again.';
      if (err.response?.data?.detail) {
        const detail = err.response.data.detail;
        errorMessage = Array.isArray(detail) ? detail.map((d: any) => d.msg).join('; ') : detail;
      }
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 transition-colors duration-200">
      <header className="h-16 px-6 flex items-center border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => navigate('/')}>
          <img src={logoUrl} alt="Med Assistant Logo" className="w-8 h-8 object-contain rounded" onError={(e) => e.currentTarget.style.display = 'none'} />
          <h1 className="font-bold text-xl tracking-tight text-blue-700 dark:text-blue-400">Med Assistant</h1>
        </div>
      </header>

      <div className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-md bg-white dark:bg-slate-900 rounded-2xl shadow-xl border border-slate-200 dark:border-slate-800 p-8">
          <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-500 mb-4">
            <UserPlus size={32} />
          </div>
          <h2 className="text-3xl font-bold text-slate-800 dark:text-slate-50">Create an account</h2>
          <p className="text-slate-500 dark:text-slate-400 mt-2">Join us to start asking medical questions</p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/30 rounded-lg text-red-600 dark:text-red-400 text-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleRegister} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Username</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                <User size={20} />
              </div>
              <input
                type="text"
                required
                className="block w-full pl-10 pr-3 py-3 border border-slate-300 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-50 focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-blue-500 transition-shadow outline-none"
                placeholder="your username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Password</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                <Lock size={20} />
              </div>
              <input
                type="password"
                required
                minLength={6}
                className="block w-full pl-10 pr-3 py-3 border border-slate-300 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-50 focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-blue-500 transition-shadow outline-none"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Confirm Password</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                <Lock size={20} />
              </div>
              <input
                type="password"
                required
                minLength={6}
                className="block w-full pl-10 pr-3 py-3 border border-slate-300 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-50 focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-blue-500 transition-shadow outline-none"
                placeholder="••••••••"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-70 disabled:cursor-not-allowed transition-colors mt-6"
          >
            {loading ? 'Creating account...' : 'Create account'}
          </button>
        </form>

        <div className="mt-8">
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-slate-300 dark:border-slate-700"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white dark:bg-slate-900 text-slate-500">Or continue with</span>
            </div>
          </div>

          <div className="mt-6 space-y-4">
            <div className="flex justify-center w-full">
              <GoogleLogin
                onSuccess={async (credentialResponse) => {
                  setLoading(true);
                  try {
                    const response = await axiosClient.post('/api/auth/google', { token: credentialResponse.credential });
                    setAuth(response.data.access_token, response.data.user);
                    if (response.data.user.role === 'admin') {
                      navigate('/admin');
                    } else {
                      navigate('/c');
                    }
                  } catch (err: any) {
                    setError('Google signup failed.');
                  } finally {
                    setLoading(false);
                  }
                }}
                onError={() => {
                  setError('Google signup failed.');
                }}
                useOneTap
              />
            </div>
          </div>
        </div>


        <p className="mt-8 text-center text-sm text-slate-600 dark:text-slate-400">
          Already have an account?{' '}
          <Link to="/login" className="font-medium text-blue-600 dark:text-blue-400 hover:text-blue-500">
            Sign in
          </Link>
        </p>
      </div>
      </div>
    </div>
  );
};

export default Register;
