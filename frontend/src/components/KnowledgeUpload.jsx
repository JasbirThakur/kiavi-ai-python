import React, { useState } from 'react';
import api from '../services/api';

export function KnowledgeUpload({ botId, onSourceAdded }) {
  const [url, setUrl] = useState('');
  const [isScraping, setIsScraping] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [message, setMessage] = useState(null);

  const handleScrape = async (e) => {
    e.preventDefault();
    if (!url.trim() || isScraping) return;
    setIsScraping(true);
    setMessage(null);

    try {
      const res = await api.knowledge.scrapeWebsite(url.trim(), botId);
      setMessage({ type: 'success', text: `Website scraped successfully: ${res.token_count || 0} tokens indexed.` });
      setUrl('');
      onSourceAdded?.();
    } catch (err) {
      setMessage({ type: 'error', text: err.message || 'Scraping failed.' });
    } finally {
      setIsScraping(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file || isUploading) return;
    setIsUploading(true);
    setMessage(null);

    try {
      const res = await api.knowledge.uploadDocument(file, botId);
      setMessage({ type: 'success', text: `Document ingested: ${res.chunks_stored || 0} chunks created.` });
      onSourceAdded?.();
    } catch (err) {
      setMessage({ type: 'error', text: err.message || 'Upload failed.' });
    } finally {
      setIsUploading(false);
      e.target.value = '';
    }
  };

  return (
    <div className="knowledge-upload-panel bg-white p-6 rounded-2xl border border-gray-100 shadow-xs space-y-6">
      <div>
        <h3 className="text-base font-semibold text-gray-800">Add Knowledge Sources</h3>
        <p className="text-xs text-gray-500">Ingest websites, PDF manuals, CSV tables, or ZIP codebases.</p>
      </div>

      {message && (
        <div className={`p-3 rounded-xl text-xs font-medium ${
          message.type === 'success' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-red-50 text-red-700 border border-red-200'
        }`}>
          {message.text}
        </div>
      )}

      {/* Website Scraper */}
      <form onSubmit={handleScrape} className="space-y-2">
        <label className="text-xs font-semibold text-gray-700">Scrape Website or Documentation URL</label>
        <div className="flex gap-2">
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/docs"
            disabled={isScraping}
            className="flex-1 px-3.5 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500"
          />
          <button
            type="submit"
            disabled={!url.trim() || isScraping}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-medium rounded-xl transition-colors disabled:opacity-50"
          >
            {isScraping ? 'Scraping...' : 'Crawl & Ingest'}
          </button>
        </div>
      </form>

      <div className="relative flex py-1 items-center">
        <div className="flex-grow border-t border-gray-100"></div>
        <span className="flex-shrink mx-3 text-2xs text-gray-400 uppercase font-medium">Or Upload Files</span>
        <div className="flex-grow border-t border-gray-100"></div>
      </div>

      {/* File Ingestion */}
      <div>
        <label className="block text-xs font-semibold text-gray-700 mb-1">Supported: PDF, DOCX, CSV, TXT, ZIP, VSIX</label>
        <label className="border-2 border-dashed border-gray-200 hover:border-emerald-400 rounded-2xl p-6 flex flex-col items-center justify-center cursor-pointer transition-colors bg-gray-50/50">
          <span className="text-2xl mb-1">📄</span>
          <span className="text-xs font-medium text-gray-700">
            {isUploading ? 'Ingesting document...' : 'Click to select file or drag here'}
          </span>
          <span className="text-2xs text-gray-400 mt-0.5">Maximum file size: 500MB</span>
          <input
            type="file"
            onChange={handleFileUpload}
            disabled={isUploading}
            accept=".pdf,.docx,.csv,.txt,.zip,.vsix"
            className="hidden"
          />
        </label>
      </div>
    </div>
  );
}

export default KnowledgeUpload;

