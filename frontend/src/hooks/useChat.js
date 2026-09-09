import { useState, useCallback, useRef } from 'react';
import api from '../services/api';

export function useChat(botId, initialConversationId = null) {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [conversationId, setConversationId] = useState(initialConversationId);
  const [error, setError] = useState(null);
  const currentTokenStream = useRef('');

  const sendMessage = useCallback((text, userName = null) => {
    if (!text || !text.trim() || isStreaming) return;

    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text.trim(),
      timestamp: new Date().toISOString()
    };

    const assistantPlaceholder = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString()
    };

    setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
    setIsStreaming(true);
    setError(null);
    currentTokenStream.current = '';

    api.chat.stream(botId, text, {
      conversationId,
      userName,
      onToken: (token) => {
        currentTokenStream.current += token;
        setMessages((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
            updated[lastIdx] = {
              ...updated[lastIdx],
              content: currentTokenStream.current
            };
          }
          return updated;
        });
      },
      onDone: (data) => {
        setIsStreaming(false);
        if (data?.conversation_id && !conversationId) {
          setConversationId(data.conversation_id);
        }
      },
      onError: (err) => {
        setIsStreaming(false);
        setError(err.message || 'Error receiving AI response.');
      }
    });
  }, [botId, conversationId, isStreaming]);

  const requestHumanAgent = useCallback(async (userEmail = '', userName = '') => {
    if (!conversationId) return { success: false, message: 'No active conversation.' };
    try {
      const res = await api.chat.requestAgent(botId, conversationId, userEmail, userName);
      return res;
    } catch (err) {
      return { success: false, message: err.message };
    }
  }, [botId, conversationId]);

  return {
    messages,
    isStreaming,
    error,
    conversationId,
    sendMessage,
    requestHumanAgent,
    setMessages
  };
}

export default useChat;

