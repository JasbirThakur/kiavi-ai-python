/**
 * Kiavi IQ Centralized API Service
 * Handles all backend communication, JWT authentication, and SSE streaming.
 */

const API_BASE = import.meta.env?.VITE_API_BASE_URL || '/api';

function getAuthHeaders() {
  const token = localStorage.getItem('token') || localStorage.getItem('kiavi_token');
  const headers = { 'Accept': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

async function handleResponse(res) {
  if (!res.ok) {
    let errorMsg = 'Request failed';
    try {
      const data = await res.json();
      errorMsg = data.message || data.detail || errorMsg;
    } catch (_) {
      errorMsg = await res.text() || errorMsg;
    }
    throw new Error(errorMsg);
  }
  return res.json();
}

export const api = {
  // Authentication
  auth: {
    async login(email, password) {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      return handleResponse(res);
    },

    async signup(email, password, name = '', companyName = 'AppDeft AI') {
      const res = await fetch(`${API_BASE}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, name, company_name: companyName })
      });
      return handleResponse(res);
    },

    async getMe() {
      const res = await fetch(`${API_BASE}/auth/me`, {
        headers: getAuthHeaders()
      });
      return handleResponse(res);
    }
  },

  // Bot Management
  bots: {
    async list() {
      const res = await fetch(`${API_BASE}/bots/`, { headers: getAuthHeaders() });
      return handleResponse(res);
    },

    async get(botId) {
      const res = await fetch(`${API_BASE}/bots/${botId}`, { headers: getAuthHeaders() });
      return handleResponse(res);
    },

    async create(name, domain = 'appdeft.ai') {
      const res = await fetch(`${API_BASE}/bots/`, {
        method: 'POST',
        headers: { ...getAuthHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, domain })
      });
      return handleResponse(res);
    },

    async update(botId, data) {
      const res = await fetch(`${API_BASE}/bots/${botId}/appearance`, {
        method: 'POST',
        headers: { ...getAuthHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      return handleResponse(res);
    },

    async uploadLogo(botId, file) {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE}/bots/${botId}/logo`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: formData
      });
      return handleResponse(res);
    }
  },

  // Knowledge Management
  knowledge: {
    async getSources(botId) {
      const res = await fetch(`${API_BASE}/knowledge/sources/${botId}`, { headers: getAuthHeaders() });
      return handleResponse(res);
    },

    async scrapeWebsite(url, botId = null, isUniversal = false) {
      const formData = new FormData();
      formData.append('url', url);
      if (botId) formData.append('bot_id', botId);
      formData.append('is_universal', String(isUniversal));

      const res = await fetch(`${API_BASE}/knowledge/website`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: formData
      });
      return handleResponse(res);
    },

    async uploadDocument(file, botId = null, isUniversal = false) {
      const formData = new FormData();
      formData.append('file', file);
      if (botId) formData.append('bot_id', botId);
      formData.append('is_universal', String(isUniversal));

      const res = await fetch(`${API_BASE}/knowledge/upload`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: formData
      });
      return handleResponse(res);
    },

    async deleteSource(sourceId) {
      const res = await fetch(`${API_BASE}/knowledge/source/${sourceId}`, {
        method: 'DELETE',
        headers: getAuthHeaders()
      });
      return handleResponse(res);
    }
  },

  // Chat & Streaming
  chat: {
    stream(botId, question, { conversationId = null, sessionId = null, userName = null, onToken, onDone, onError }) {
      const formData = new FormData();
      formData.append('bot_id', botId);
      formData.append('question', question);
      if (conversationId) formData.append('conversation_id', conversationId);
      if (sessionId) formData.append('session_id', sessionId);
      if (userName) formData.append('user_name', userName);

      fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: formData
      })
      .then(async (res) => {
        if (!res.ok) throw new Error(`Chat stream error (${res.status})`);
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop(); // Keep partial segment in buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'token') onToken?.(data.token);
                if (data.type === 'done') onDone?.(data);
              } catch (_) {}
            }
          }
        }
      })
      .catch((err) => onError?.(err));
    },

    async requestAgent(botId, conversationId, userEmail = '', userName = '') {
      const formData = new FormData();
      formData.append('bot_id', botId);
      formData.append('conversation_id', conversationId);
      formData.append('user_email', userEmail);
      formData.append('user_name', userName);

      const res = await fetch(`${API_BASE}/public/chat/request-agent`, {
        method: 'POST',
        body: formData
      });
      return handleResponse(res);
    }
  },

  // Leads
  leads: {
    async list(botId) {
      const res = await fetch(`${API_BASE}/leads/${botId}`, { headers: getAuthHeaders() });
      return handleResponse(res);
    }
  },

  // Insights
  insights: {
    async get(botId) {
      const res = await fetch(`${API_BASE}/insights/${botId}`, { headers: getAuthHeaders() });
      return handleResponse(res);
    }
  },

  // Health
  async health() {
    const res = await fetch(`${API_BASE}/health`);
    return handleResponse(res);
  }
};

export default api;

