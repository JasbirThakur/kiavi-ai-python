import React, { useState } from 'react';

export function MessageBubble({ message, onOptionClick }) {
  const isUser = message.role === 'user';
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Parse any follow up options block if present
  let cleanContent = message.content || '';
  let followUpOptions = [];

  if (!isUser && cleanContent.includes('<<<FOLLOW_UP>>>')) {
    const parts = cleanContent.split('<<<FOLLOW_UP>>>');
    cleanContent = parts[0].trim();
    const followUpRaw = parts[1]?.split('<<<END_FOLLOW_UP>>>')[0]?.trim();
    if (followUpRaw) {
      try {
        const parsed = JSON.parse(followUpRaw);
        followUpOptions = parsed.options || [];
      } catch (_) {}
    }
  }

  const primaryOptions = followUpOptions.slice(0, 3);
  const remainingOptions = followUpOptions.slice(3);
  const filteredRemaining = remainingOptions.filter(opt =>
    opt.toLowerCase().includes(searchQuery.trim().toLowerCase())
  );

  return (
    <div className={`message-row flex flex-col my-2 ${isUser ? 'items-end' : 'items-start'}`}>
      <div
        className={`message-bubble max-w-[85%] md:max-w-[75%] px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? 'bg-emerald-600 text-white rounded-br-sm'
            : 'bg-white text-gray-800 border border-gray-100 shadow-sm rounded-bl-sm'
        }`}
      >
        {cleanContent || (
          <div className="flex items-center space-x-1.5 py-1">
            <span className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce"></span>
            <span className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce [animation-delay:0.2s]"></span>
            <span className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce [animation-delay:0.4s]"></span>
          </div>
        )}
      </div>

      {followUpOptions.length > 0 && (
        <div className="follow-up-section mt-2 max-w-[85%] space-y-2">
          {/* Primary Pills */}
          <div className="flex flex-wrap gap-1.5">
            {(followUpOptions.length <= 4 ? followUpOptions : primaryOptions).map((opt, idx) => (
              <button
                key={idx}
                onClick={() => onOptionClick?.(opt)}
                className="text-xs px-3 py-1 bg-white hover:bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full transition-colors shadow-2xs cursor-pointer"
              >
                {opt}
              </button>
            ))}

            {followUpOptions.length > 4 && (
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="text-xs px-3 py-1 bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-300 font-medium rounded-full transition-colors shadow-2xs cursor-pointer flex items-center gap-1"
              >
                <span>{isExpanded ? 'Collapse ▴' : `More (+${remainingOptions.length}) Topics ▾`}</span>
              </button>
            )}
          </div>

          {/* Expandable Search & Remaining Pills */}
          {followUpOptions.length > 4 && isExpanded && (
            <div className="p-3 bg-gray-50 border border-gray-200 rounded-xl space-y-2 animate-fadeIn">
              <div className="relative">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder={`Search ${followUpOptions.length} topics...`}
                  className="w-full text-xs px-3 py-1.5 bg-white border border-gray-300 rounded-lg outline-none focus:border-emerald-500"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery('')}
                    className="absolute right-2.5 top-1.5 text-gray-400 hover:text-gray-600 text-xs"
                  >
                    ✕
                  </button>
                )}
              </div>

              <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto pr-1">
                {filteredRemaining.length > 0 ? (
                  filteredRemaining.map((opt, idx) => (
                    <button
                      key={idx}
                      onClick={() => onOptionClick?.(opt)}
                      className="text-xs px-2.5 py-1 bg-white hover:bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-lg transition-colors cursor-pointer"
                    >
                      {opt}
                    </button>
                  ))
                ) : (
                  <div className="text-[11px] text-gray-400 italic py-1">No matching topics found.</div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default MessageBubble;

