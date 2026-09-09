import React, { useState, useEffect } from 'react';
import api from '../services/api';
import KnowledgeUpload from '../components/KnowledgeUpload';

export function KnowledgeBase({ bot }) {
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchSources = async () => {
    if (!bot?.id) return;
    try {
      setLoading(true);
      const data = await api.knowledge.getSources(bot.id);
      setSources(data);
    } catch (err) {
      console.error('Failed to load sources:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSources();
  }, [bot?.id]);

  const handleDelete = async (sourceId) => {
    if (!confirm('Are you sure you want to delete this knowledge source?')) return;
    try {
      await api.knowledge.deleteSource(sourceId);
      fetchSources();
    } catch (err) {
      alert(err.message || 'Delete failed');
    }
  };

  return (
    <div className="knowledge-page p-6 max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Knowledge Base: {bot?.name}</h1>
        <p className="text-xs text-gray-500">Indexed documents, crawled domains, and vector embeddings.</p>
      </div>

      <KnowledgeUpload botId={bot?.id} onSourceAdded={fetchSources} />

      <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-xs space-y-4">
        <h3 className="text-sm font-semibold text-gray-800">Indexed Sources ({sources.length})</h3>

        {loading ? (
          <p className="text-xs text-gray-400">Loading sources...</p>
        ) : sources.length === 0 ? (
          <p className="text-xs text-gray-400">No knowledge sources configured yet. Add a website URL or document above.</p>
        ) : (
          <div className="divide-y divide-gray-100">
            {sources.map((src) => (
              <div key={src.id} className="py-3 flex justify-between items-center">
                <div>
                  <h4 className="text-sm font-medium text-gray-800">{src.title}</h4>
                  <p className="text-xs text-gray-400">
                    Type: <span className="uppercase">{src.kind}</span> • Tokens: {src.tokenCount?.toLocaleString() || 0}
                    {src.isUniversal && <span className="ml-2 px-2 py-0.5 bg-purple-50 text-purple-700 rounded-md text-2xs">Universal</span>}
                  </p>
                </div>
                <button
                  onClick={() => handleDelete(src.id)}
                  className="text-xs text-red-500 hover:text-red-700 px-3 py-1 rounded-lg hover:bg-red-50 transition-colors"
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default KnowledgeBase;

