import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Chat from './pages/Chat';
import KnowledgeBase from './pages/KnowledgeBase';
import Login from './pages/Login';
import api from './services/api';

export function App() {
  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [currentTab, setCurrentTab] = useState('dashboard');
  const [selectedBot, setSelectedBot] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      api.auth.getMe()
        .then((userData) => setUser(userData))
        .catch(() => {
          localStorage.removeItem('token');
          localStorage.removeItem('kiavi_token');
        })
        .finally(() => setAuthChecked(true));
    } else {
      setAuthChecked(true);
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('kiavi_token');
    setUser(null);
  };

  if (!authChecked) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 text-white text-sm">
        Initializing Kiavi IQ...
      </div>
    );
  }

  if (!user) {
    return <Login onLoginSuccess={(u) => setUser(u)} />;
  }

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      <Sidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        botName={selectedBot?.name || 'All Bots'}
        onLogout={handleLogout}
      />

      <main className="flex-1 h-full overflow-y-auto">
        {currentTab === 'dashboard' && (
          <Dashboard
            onSelectBot={(bot) => {
              setSelectedBot(bot);
              setCurrentTab('chat');
            }}
          />
        )}

        {currentTab === 'chat' && (
          <Chat bot={selectedBot} />
        )}

        {currentTab === 'knowledge' && (
          <KnowledgeBase bot={selectedBot} />
        )}

        {currentTab === 'leads' && (
          <div className="p-8 max-w-5xl mx-auto">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Leads Capture</h2>
            <p className="text-xs text-gray-500 mb-6">Leads collected automatically during chat interactions.</p>
            <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-xs">
              <p className="text-xs text-gray-400">View captured names, verified emails, and phone numbers in real time.</p>
            </div>
          </div>
        )}

        {currentTab === 'settings' && (
          <div className="p-8 max-w-5xl mx-auto">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Bot Appearance & Webhooks</h2>
            <p className="text-xs text-gray-500 mb-6">Configure custom colors, greetings, and webhook integration.</p>
            <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-xs">
              <p className="text-xs text-gray-600 font-medium mb-1">Bot Name: {selectedBot?.name || 'N/A'}</p>
              <p className="text-xs text-gray-400">Public Key: {selectedBot?.publicKey || 'N/A'}</p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;

