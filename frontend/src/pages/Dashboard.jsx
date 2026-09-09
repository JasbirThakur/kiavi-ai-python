import React, { useState, useEffect } from 'react';
import api from '../services/api';

export function Dashboard({ onSelectBot }) {
  const [bots, setBots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [newBotName, setNewBotName] = useState('');
  const [newBotDomain, setNewBotDomain] = useState('');
  const [creating, setCreating] = useState(false);

  const fetchBots = async () => {
    try {
      setLoading(true);
      const data = await api.bots.list();
      setBots(data);
      if (data.length > 0 && onSelectBot) {
        onSelectBot(data[0]);
      }
    } catch (err) {
      setError(err.message || 'Failed to load bots');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBots();
  }, []);

  const handleCreateBot = async (e) => {
    e.preventDefault();
    if (!newBotName.trim() || creating) return;
    setCreating(true);

    try {
      const created = await api.bots.create(newBotName.trim(), newBotDomain.trim() || 'appdeft.ai');
      setNewBotName('');
      setNewBotDomain('');
      fetchBots();
      if (onSelectBot) onSelectBot(created);
    } catch (err) {
      alert(err.message || 'Failed to create bot');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="dashboard-page p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight">Your AI Agents</h1>
          <p className="text-xs text-gray-500">Manage grounded conversational bots and knowledge deployment.</p>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 text-red-700 text-xs rounded-xl">
          {error}
        </div>
      )}

      {/* New Bot Creator Card */}
      <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-xs">
        <h3 className="text-sm font-semibold text-gray-800 mb-3">Create New AI Assistant</h3>
        <form onSubmit={handleCreateBot} className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            required
            value={newBotName}
            onChange={(e) => setNewBotName(e.target.value)}
            placeholder="Agent Name (e.g. Sales Expert)"
            className="flex-1 px-4 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500"
          />
          <input
            type="text"
            value={newBotDomain}
            onChange={(e) => setNewBotDomain(e.target.value)}
            placeholder="Primary Website Domain (e.g. alorica.com)"
            className="flex-1 px-4 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500"
          />
          <button
            type="submit"
            disabled={creating || !newBotName.trim()}
            className="px-5 py-2 bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-sm rounded-xl transition-colors disabled:opacity-50"
          >
            {creating ? 'Creating...' : '+ Create Agent'}
          </button>
        </form>
      </div>

      {/* Bot Cards Grid */}
      {loading ? (
        <div className="text-center py-12 text-sm text-gray-400">Loading agents...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {bots.map((bot) => (
            <div
              key={bot.id}
              onClick={() => onSelectBot?.(bot)}
              className="bot-card bg-white p-5 rounded-2xl border border-gray-100 shadow-xs hover:shadow-md hover:border-emerald-200 cursor-pointer transition-all space-y-4"
            >
              <div className="flex items-center space-x-3">
                {bot.logoUrl ? (
                  <img src={bot.logoUrl} alt={bot.name} className="w-12 h-12 rounded-xl object-cover border" />
                ) : (
                  <div className="w-12 h-12 rounded-xl bg-emerald-100 text-emerald-700 flex items-center justify-center font-bold text-lg">
                    {bot.name.charAt(0).toUpperCase()}
                  </div>
                )}
                <div>
                  <h3 className="font-semibold text-gray-800 text-base">{bot.name}</h3>
                  <p className="text-xs text-gray-400">{bot.domain || 'All Knowledge'}</p>
                </div>
              </div>

              <div className="border-t border-gray-50 pt-3 flex justify-between items-center text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                  Grounded & Ready
                </span>
                <span className="text-emerald-600 font-medium">Open Agent →</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Dashboard;

