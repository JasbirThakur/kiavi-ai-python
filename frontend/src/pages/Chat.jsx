import React, { useState } from 'react';
import BotHeader from '../components/BotHeader';
import ChatWindow from '../components/ChatWindow';
import useChat from '../hooks/useChat';

export function Chat({ bot }) {
  const { messages, isStreaming, error, sendMessage, requestHumanAgent } = useChat(bot?.id);
  const [handoffStatus, setHandoffStatus] = useState(null);

  const handleHumanHandoff = async () => {
    const userEmail = prompt("Please provide your email address for the human agent notification:", "jasbirsingh17050@gmail.com");
    if (!userEmail) return;

    const res = await requestHumanAgent(userEmail);
    if (res.success) {
      setHandoffStatus('Requested. A human agent has been alerted via email.');
    } else {
      alert(res.message || 'Unable to request handoff.');
    }
  };

  return (
    <div className="chat-page flex flex-col h-full bg-slate-50">
      <BotHeader
        bot={bot}
        onHumanHandoff={handleHumanHandoff}
        isHandedOff={Boolean(handoffStatus)}
      />

      {handoffStatus && (
        <div className="bg-amber-50 border-b border-amber-200 px-4 py-2 text-xs text-amber-800 flex justify-between items-center">
          <span>🔔 {handoffStatus}</span>
          <button onClick={() => setHandoffStatus(null)} className="font-bold ml-2 text-amber-900">×</button>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2 text-xs text-red-700">
          ⚠️ {error}
        </div>
      )}

      <div className="flex-1 overflow-hidden">
        <ChatWindow
          messages={messages}
          isStreaming={isStreaming}
          onSendMessage={sendMessage}
          placeholder={`Message ${bot?.name || 'Assistant'}...`}
        />
      </div>
    </div>
  );
}

export default Chat;

