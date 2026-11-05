import { useEffect, useRef, useState } from 'react';
import { isTauri } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import './App.css';

type Result = { path: string; score: number; snippet: string; chunk_id: number | null };
type Status = { indexed_files: number; chunks: number; state: string };
type Theme = 'system' | 'light' | 'dark';
const API = import.meta.env.DEV ? '/api' : 'http://127.0.0.1:8000';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}
const filename = (path: string) => path.split(/[\\/]/).pop() || path;
const directory = (path: string) => path.split(/[\\/]/).slice(0, -1).join(' / ') || 'Indexed folder';
const typeLabel = (path: string) => (filename(path).split('.').pop() || 'FILE').toUpperCase().slice(0, 4);

function Icon({ name }: { name: string }) {
  const paths: Record<string, React.ReactNode> = {
    search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 4 4"/></>,
    sidebar: <><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/></>,
    folder: <path d="M3 6.5h6l2 2h10v9.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>,
    plus: <path d="M12 5v14M5 12h14"/>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="M12 3v2m0 14v2M3 12h2m14 0h2M5.6 5.6 7 7m10 10 1.4 1.4m0-12.8L17 7M7 17l-1.4 1.4"/><circle cx="12" cy="12" r="8"/></>,
    close: <path d="M6 6l12 12M18 6 6 18"/>,
  };
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function App() {
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem('refinder-theme') as Theme) || 'system');
  const [collapsed, setCollapsed] = useState(() => window.innerWidth < 800);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Result[]>([]);
  const [selected, setSelected] = useState<Result | null>(null);
  const [fileText, setFileText] = useState('');
  const [status, setStatus] = useState<Status | null>(null);
  const [folder, setFolder] = useState('');
  const [busy, setBusy] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState('');
  const [previewOpen, setPreviewOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('refinder-theme', theme); }, [theme]);
  async function refreshStatus() {
    try { setStatus(await request<Status>('/status')); } catch { setStatus(null); }
  }
  useEffect(() => { void refreshStatus(); }, []);
  useEffect(() => {
    function shortcut(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); inputRef.current?.focus(); }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'y' && selected) { e.preventDefault(); setPreviewOpen(v => !v); }
      if (e.key === 'Escape') { setPreviewOpen(false); setSettingsOpen(false); }
    }
    window.addEventListener('keydown', shortcut); return () => window.removeEventListener('keydown', shortcut);
  }, [selected]);
  useEffect(() => {
    if (!selected) { setFileText(''); return; }
    const controller = new AbortController(); setFileText('');
    request<{ text: string }>(`/file?path=${encodeURIComponent(selected.path)}`, { signal: controller.signal })
      .then(data => setFileText(data.text)).catch(e => { if (e.name !== 'AbortError') setFileText('Could not load extracted text.'); });
    return () => controller.abort();
  }, [selected]);

  async function search(e: React.FormEvent) {
    e.preventDefault(); if (!query.trim() || busy) return;
    setBusy(true); setError(''); setSearched(true); setPreviewOpen(false);
    try { const data = await request<{ results: Result[] }>(`/search?q=${encodeURIComponent(query.trim())}&k=20`); setResults(data.results); setSelected(data.results[0] || null); }
    catch (e) { setResults([]); setSelected(null); setError(e instanceof Error ? e.message : 'Search failed'); }
    finally { setBusy(false); }
  }
  async function chooseFolder() {
    let path: string | null = null;
    try {
      path = isTauri() ? await open({ directory: true, multiple: false, title: 'Choose a folder to index' }) : window.prompt('Enter the full path to a folder on this computer:');
      if (!path) return;
      if (status?.indexed_files && !window.confirm('Indexing this folder will replace the current index. Continue?')) return;
      setBusy(true); setError('');
      await request('/index', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ folder: path }) });
      setFolder(path); setResults([]); setSelected(null); setSearched(false); await refreshStatus();
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not index folder'); }
    finally { setBusy(false); }
  }
  const statusText = busy ? 'Working…' : !status ? 'Backend unavailable' : status.state === 'indexing' ? 'Indexing…' : status.indexed_files ? `${status.indexed_files.toLocaleString()} files indexed` : 'No folder indexed';
  const selectResult = (result: Result) => { setSelected(result); if (window.innerWidth < 820) setPreviewOpen(true); };
  const preview = <section className="preview panel" aria-label="File preview">
    <div className="preview-bar"><span>Preview</span><button className="icon-button preview-close" onClick={() => setPreviewOpen(false)} aria-label="Close preview"><Icon name="close" /></button></div>
    {selected ? <div className="preview-body"><div className="preview-file"><span className="file-icon">{typeLabel(selected.path)}</span><div><h2>{filename(selected.path)}</h2><p className="path">{directory(selected.path)}</p></div></div><p className="score">Similarity {selected.score.toFixed(3)}</p><div className="preview-divider"/><h3>Extracted text</h3><p className="file-text">{fileText || 'Loading extracted text…'}</p></div> : <div className="empty-preview">Select a result to preview its extracted text.</div>}
  </section>;
  return <div className={`app ${collapsed ? 'sidebar-collapsed' : ''}`}>
    <aside className="sidebar"><div className="sidebar-head"><span className="brand">ReFinder</span><button className="icon-button collapse-button" title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} onClick={() => setCollapsed(!collapsed)}><Icon name="sidebar"/></button></div>
      <div className="sidebar-scroll"><p className="section-label">LIBRARY</p><button className="nav-item active" title="Search" onClick={() => inputRef.current?.focus()}><Icon name="search"/><span>Search</span></button><p className="section-label folder-label">FOLDER</p>{folder && <div className="nav-item folder-item" title={folder}><Icon name="folder"/><span>{filename(folder)}</span></div>}<button className="nav-item" title="Choose folder" onClick={chooseFolder} disabled={busy}><Icon name="plus"/><span>{folder ? 'Change folder' : 'Choose folder'}</span></button></div>
      <div className="sidebar-bottom"><div className="index-status" title={statusText}><span className={`status-dot ${status ? 'online' : ''}`}/><span>{statusText}</span></div><button className="nav-item" title="Settings" onClick={() => setSettingsOpen(true)}><Icon name="settings"/><span>Settings</span></button></div>
    </aside>
    <main className="workspace"><header className="page-heading"><h1>Search</h1><span className="shortcut">⌘ K</span></header><form className="search-box" onSubmit={search}><Icon name="search"/><input ref={inputRef} value={query} onChange={e => setQuery(e.target.value)} placeholder="Search your files by meaning" aria-label="Search your files"/><button type="submit" disabled={busy || !query.trim()}>Search</button></form>
      <div className="scope-row"><span>{folder ? filename(folder) : 'All indexed files'}</span><span>{busy ? 'Searching or indexing…' : statusText}</span></div>
      {error && <div className="error" role="alert">{error === 'Failed to fetch' ? 'Could not connect to the local ReFinder server. Start the backend and try again.' : error}</div>}
      <div className="results-heading"><span>{searched ? `${results.length} ${results.length === 1 ? 'result' : 'results'}` : 'Your results'}</span><span>Sorted by relevance</span></div>
      <div className="content-grid"><section className="results panel" aria-label="Search results">{results.length ? results.map((result, i) => <button key={`${result.path}-${result.chunk_id}-${i}`} className={`result ${selected === result ? 'selected' : ''}`} onClick={() => selectResult(result)}><span className="file-icon">{typeLabel(result.path)}</span><span className="result-content"><strong>{filename(result.path)}</strong><span className="path">{directory(result.path)}</span><span className="snippet">{result.snippet}</span><span className="score">Similarity {result.score.toFixed(3)}</span></span></button>) : <div className="empty-state"><div className="empty-icon"><Icon name="search"/></div><h2>{busy ? 'Searching…' : searched ? 'No results found' : 'Find what you remember'}</h2><p>{busy ? 'This may take a moment while the model loads.' : searched ? 'Try a different description or choose another folder.' : status?.indexed_files ? 'Describe the content you’re looking for.' : 'Choose a folder, then describe the content you’re looking for.'}</p>{!status?.indexed_files && <button className="primary-button" onClick={chooseFolder} disabled={busy}>Choose folder</button>}</div>}</section>{preview}</div>
      <div className="compact-footer"><span>{selected ? filename(selected.path) : 'Select a result to preview'}</span><button onClick={() => setPreviewOpen(true)} disabled={!selected}>Preview <span>⌘ Y</span></button></div>
    </main>
    {previewOpen && <div className="preview-overlay" onClick={() => setPreviewOpen(false)}><div className="preview-sheet" onClick={e => e.stopPropagation()}>{preview}</div></div>}
    {settingsOpen && <div className="modal-backdrop" onClick={() => setSettingsOpen(false)}><section className="settings panel" role="dialog" aria-modal="true" aria-labelledby="settings-title" onClick={e => e.stopPropagation()}><div className="settings-header"><h2 id="settings-title">Settings</h2><button className="icon-button" onClick={() => setSettingsOpen(false)} aria-label="Close settings"><Icon name="close"/></button></div><label htmlFor="theme">Appearance</label><select id="theme" value={theme} onChange={e => setTheme(e.target.value as Theme)}><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select><p>ReFinder uses the local search index on this computer.</p></section></div>}
  </div>;
}
export default App;
