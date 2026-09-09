import React from 'react';

export function BotHeader({ bot, onHumanHandoff, isHandedOff = false }) {
  return (
    <div className="bot-header flex items-center justify-between p-4 border-b bg-white shadow-sm">
      <div className="flex items-center space-x-3">
        {bot?.logoUrl ? (
          <img src={bot.logoUrl} alt={bot.name} className="w-10 h-10 rounded-full object-cover border" />
        ) : (
          <div className="w-10 h-10 rounded-full bg-emerald-500 text-white flex items-center justify-center font-bold">
            {bot?.name ? bot.name.charAt(0).toUpperCase() : 'K'}
          </div>
        )}
        <div>
          <h2 className="font-semibold text-gray-800 text-base">{bot?.name || 'Kiavi IQ Assistant'}</h2>
          <div className="flex items-center text-xs text-gray-500">
            <span className={`w-2 h-2 rounded-full mr-1.5 ${isHandedOff ? 'bg-amber-500' : 'bg-emerald-500 animate-pulse'}`}></span>
            {isHandedOff ? 'Live Agent Active' : 'AI Grounded & Active'}
          </div>
        </div>
      </div>

      <button
        onClick={onHumanHandoff}
        className="px-3 py-1.5 text-xs font-medium text-emerald-700 bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 rounded-lg transition-colors flex items-center gap-1.5"
      >
        <span>👤</span>
        <span>Talk to Human</span>
      </button>
    </div>
  );
}

export default BotHeader;

