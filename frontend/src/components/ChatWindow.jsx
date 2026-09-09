import React, { useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import ChatInput from './ChatInput';

export function ChatWindow({ messages, isStreaming, onSendMessage, onOptionClick, placeholder }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isStreaming]);

  return (
    <div className="chat-window flex flex-col h-full bg-slate-50">
      <div className="messages-area flex-1 overflow-y-auto p-4 space-y-2">
        {messages.length === 0 ? (
          <div className="empty-state flex flex-col items-center justify-center h-full text-center text-gray-400">
            <span className="text-4xl mb-2">💬</span>
            <p className="text-sm">Start a conversation. All answers are grounded in verified knowledge.</p>
          </div>
        ) : (
          messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              onOptionClick={onOptionClick || onSendMessage}
            />
          ))
        )}
        <div ref={bottomRef} />
      </div>

      <ChatInput
        onSendMessage={onSendMessage}
        disabled={isStreaming}
        placeholder={placeholder}
      />
    </div>
  );
}

export default ChatWindow;

