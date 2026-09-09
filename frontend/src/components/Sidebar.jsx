import React from 'react';

export function Sidebar({ currentTab, onSelectTab, botName = "Kiavi Assistant", onLogout }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: '📊' },
    { id: 'chat', label: 'Chat Preview', icon: '💬' },
    { id: 'knowledge', label: 'Knowledge Base', icon: '📚' },
    { id: 'leads', label: 'Leads Capture', icon: '🎯' },
    { id: 'settings', label: 'Bot Settings', icon: '⚙️' }
  ];

  return (
    <aside className="sidebar w-64 bg-slate-900 text-white flex flex-col justify-between p-4 border-r border-slate-800">
      <div>
        <div className="flex items-center space-x-2.5 px-2 py-3 mb-6">
          <div className="w-8 h-8 rounded-lg bg-emerald-500 flex items-center justify-center font-bold text-white shadow-sm">
            K
          </div>
          <div>
            <h1 className="font-bold text-base tracking-tight text-white">Kiavi IQ</h1>
            <span className="text-2xs text-emerald-400 font-medium">Grounded Agent</span>
          </div>
        </div>

        <nav className="space-y-1">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                currentTab === item.id
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <span className="text-base">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
      </div>

      <div className="border-t border-slate-800 pt-4">
        <div className="px-2 mb-3">
          <p className="text-xs text-slate-400">Selected Bot</p>
          <p className="text-sm font-semibold text-white truncate">{botName}</p>
        </div>
        {onLogout && (
          <button
            onClick={onLogout}
            className="w-full flex items-center space-x-2 px-3 py-2 text-xs font-medium text-red-400 hover:bg-red-950/40 rounded-lg transition-colors"
          >
            <span>🚪</span>
            <span>Sign Out</span>
          </button>
        )}
      </div>
    </aside>
  );
}

export default Sidebar;

