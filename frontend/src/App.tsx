import { useEffect, useRef, useState } from 'react';
import { isTauri } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import './App.css';
import { DocumentView, TextBlocks } from './Preview';

type Result = { path: string; score: number; snippet: string; passage?: string; chunk_id: number | null; modified: number; added: number; size: number; type: string };
type App = { path: string; name: string };
const IS_MAC = /Mac/i.test(navigator.userAgent);
type DatePreset = 'any' | 'today' | '7d' | '30d' | 'year' | 'custom';
type DateRange = { preset: DatePreset; from: string; to: string };
type SizeKey = 'any' | 'small' | 'medium' | 'large' | 'huge';
type SortKey = 'relevance' | 'modified' | 'added' | 'name' | 'size';
type Filters = { types: string[]; modified: DateRange; added: DateRange; size: SizeKey; folder: string };
const anyDate: DateRange = { preset: 'any', from: '', to: '' };
const noFilters: Filters = { types: [], modified: anyDate, added: anyDate, size: 'any', folder: '' };
const DATE_LABELS: Record<DatePreset, string> = { any: 'Any time', today: 'Today', '7d': 'Past 7 days', '30d': 'Past 30 days', year: 'Past year', custom: 'Custom…' };
const SIZES: Record<SizeKey, [string, number | null, number | null]> = { any: ['Any size', null, null], small: ['Under 100 KB', null, 1e5], medium: ['100 KB – 1 MB', 1e5, 1e6], large: ['1 – 10 MB', 1e6, 1e7], huge: ['Over 10 MB', 1e7, null] };
const DAY = 86400;
// Unix-second bounds for a date filter; custom dates are local days, "to" inclusive.
function dateBounds(r: DateRange): [number | null, number | null] {
  const now = Date.now() / 1000;
  if (r.preset === 'today') { const d = new Date(); d.setHours(0, 0, 0, 0); return [d.getTime() / 1000, null]; }
  if (r.preset === '7d') return [now - 7 * DAY, null];
  if (r.preset === '30d') return [now - 30 * DAY, null];
  if (r.preset === 'year') return [now - 365 * DAY, null];
  if (r.preset === 'custom') return [r.from ? new Date(`${r.from}T00:00:00`).getTime() / 1000 : null, r.to ? new Date(`${r.to}T23:59:59`).getTime() / 1000 : null];
  return [null, null];
}
function filterParams(f: Filters): string {
  const p = new URLSearchParams();
  if (f.types.length) p.set('types', f.types.join(','));
  const [ma, mb] = dateBounds(f.modified), [aa, ab] = dateBounds(f.added), [, lo, hi] = SIZES[f.size];
  if (ma !== null) p.set('modified_after', String(ma)); if (mb !== null) p.set('modified_before', String(mb));
  if (aa !== null) p.set('added_after', String(aa)); if (ab !== null) p.set('added_before', String(ab));
  if (lo !== null) p.set('min_size', String(lo)); if (hi !== null) p.set('max_size', String(hi));
  if (f.folder.trim()) p.set('folder', f.folder.trim());
  return p.toString();
}
const formatDate = (t: number) => t ? new Date(t * 1000).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : '—';
const formatSize = (b: number) => b < 1e3 ? `${b} B` : b < 1e6 ? `${(b / 1e3).toFixed(0)} KB` : b < 1e9 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e9).toFixed(2)} GB`;
type Status = { indexed_files: number; chunks: number; state: string; embedded_chunks: number; total_chunks: number; embedded_images?: number; total_images?: number; error: string | null; folder: string | null; folders: string[]; whole_computer: boolean; file_types: Record<string, number> };
type Theme = 'system' | 'light' | 'dark';
type Mode = 'smart' | 'keyword';
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
// Paths are absolute; show the home folder as ~ to keep them short.
const shortPath = (path: string) => path.replace(/^(\/Users|\/home)\/[^/]+/, '~').replace(/^[A-Za-z]:\\Users\\[^\\]+/, '~');
const directory = (path: string) => shortPath(path).split(/[\\/]/).slice(0, -1).join(' / ') || 'Indexed folder';
const typeLabel = (path: string) => (filename(path).split('.').pop() || 'FILE').toUpperCase().slice(0, 4);

function Icon({ name }: { name: string }) {
  const paths: Record<string, React.ReactNode> = {
    search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 4 4"/></>,
    sidebar: <><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/></>,
    folder: <path d="M3 6.5h6l2 2h10v9.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>,
    plus: <path d="M12 5v14M5 12h14"/>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="M12 3v2m0 14v2M3 12h2m14 0h2M5.6 5.6 7 7m10 10 1.4 1.4m0-12.8L17 7M7 17l-1.4 1.4"/><circle cx="12" cy="12" r="8"/></>,
    close: <path d="M6 6l12 12M18 6 6 18"/>,
    reveal: <><path d="M3 6.5h6l2 2h10v9.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M12 11v5M9.5 13.5 12 16l2.5-2.5"/></>,
    expand: <path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7"/>,
    shrink: <path d="M20 10h-6V4M4 14h6v6M14 10l7-7M10 14l-7 7"/>,
    swap: <path d="M7 4 3 8l4 4M3 8h14M17 20l4-4-4-4M21 16H7"/>,
    similar: <><rect x="3" y="3" width="11" height="14" rx="2"/><rect x="10" y="7" width="11" height="14" rx="2"/></>,
    computer: <><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/></>,
    keyword: <><path d="M4 7V5h16v2M12 5v14M9 19h6"/></>,
  };
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function App() {
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem('refinder-theme') as Theme) || 'system');
  const [collapsed, setCollapsed] = useState(() => window.innerWidth < 800);
  const [mode, setMode] = useState<Mode>('smart');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Result[]>([]);
  const [selected, setSelected] = useState<Result | null>(null);
  const [fileText, setFileText] = useState('');
  const [status, setStatus] = useState<Status | null>(null);
  const [folder, setFolder] = useState('');
  const [busy, setBusy] = useState(false);
  const [searched, setSearched] = useState(false);
  const [suggestion, setSuggestion] = useState<string | null>(null);
  const [understood, setUnderstood] = useState<string[]>([]);  // hints read from the query, e.g. ["PDF", "from 2025"]
  const [similarTo, setSimilarTo] = useState<string | null>(null);  // showing "More like this" for this path
  const [filters, setFilters] = useState<Filters>(noFilters);
  const [sort, setSort] = useState<SortKey>('relevance');
  const filtersOn = filterParams(filters) !== '';
  const [error, setError] = useState('');
  const [previewOpen, setPreviewOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);  // preview shown large over the whole window
  const [previewTab, setPreviewTab] = useState<'document' | 'text'>('document');
  const [zoom, setZoom] = useState(1);
  const [openWith, setOpenWith] = useState<{ path: string; default: App | null; apps: App[]; loading: boolean } | null>(null);
  const openWithRef = useRef<HTMLDivElement>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const typeMenuRef = useRef<HTMLDetailsElement>(null);
  // Close the file-type menu on any click outside it.
  useEffect(() => {
    function outside(e: MouseEvent) {
      const d = typeMenuRef.current; if (d?.open && !d.contains(e.target as Node)) d.open = false;
      if (openWithRef.current && !openWithRef.current.contains(e.target as Node)) setOpenWith(null);
    }
    document.addEventListener('mousedown', outside); return () => document.removeEventListener('mousedown', outside);
  }, []);

  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('refinder-theme', theme); }, [theme]);
  async function refreshStatus() {
    try { setStatus(await request<Status>('/status')); } catch { setStatus(null); }
  }
  // The desktop app starts the backend itself; keep checking for up to two minutes while it starts.
  const [starting, setStarting] = useState(true);
  useEffect(() => {
    let tries = 0, stopped = false;
    async function poll() {
      while (!stopped && tries++ < 120) {
        try { setStatus(await request<Status>('/status')); setStarting(false); return; } catch { /* not up yet */ }
        await new Promise(r => setTimeout(r, 1000));
      }
      if (!stopped) setStarting(false);
    }
    void poll();
    return () => { stopped = true; };
  }, []);
  // Files are keyword-searchable as soon as /index returns; meaning-based results fill in while embedding runs.
  const indexing = status?.state === 'extracting' || status?.state === 'embedding' || status?.state === 'updating';
  // "Understanding" = embedding chunks for meaning search. Track the rate between polls for a time-left estimate.
  const embedTotal = (status?.total_chunks || 0) + (status?.total_images || 0);
  const embedDone = (status?.embedded_chunks || 0) + (status?.embedded_images || 0);
  const embedding = status?.state === 'embedding' && embedTotal > 0;
  const embedPct = embedding ? Math.min(100, (100 * embedDone) / embedTotal) : 0;
  const embedStart = useRef<{ t: number; done: number; total: number } | null>(null);
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);
  useEffect(() => {
    if (!embedding || !status) { embedStart.current = null; setEtaSeconds(null); return; }
    const now = Date.now() / 1000, start = embedStart.current;
    if (!start || start.total !== embedTotal || embedDone < start.done) {
      embedStart.current = { t: now, done: embedDone, total: embedTotal }; setEtaSeconds(null); return;
    }
    const rate = (embedDone - start.done) / Math.max(1, now - start.t);
    setEtaSeconds(rate > 0 && now - start.t >= 3 ? (embedTotal - embedDone) / rate : null);
  }, [embedding, status, embedDone, embedTotal]);
  const etaText = etaSeconds === null ? '' : etaSeconds < 60 ? 'less than a minute left' : `about ${Math.round(etaSeconds / 60)} min left`;
  useEffect(() => { if (!indexing) return; const t = setInterval(() => void refreshStatus(), 1000); return () => clearInterval(t); }, [indexing]);
  useEffect(() => {
    function shortcut(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); inputRef.current?.focus(); }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'y' && selected) { e.preventDefault(); setPreviewOpen(v => !v); }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'o' && selected) { e.preventDefault(); void openFile(selected.path); }
      if (e.key === 'Escape') { setOpenWith(null); setPreviewOpen(false); setExpanded(false); setSettingsOpen(false); }
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

  async function runSearch(m: Mode, q = query, f = filters) {
    const fp = filterParams(f);
    if ((!q.trim() && !fp) || busy) return;
    // Clear the previous results right away so they never linger under a new query.
    setBusy(true); setError(''); setSearched(true); setPreviewOpen(false); setSuggestion(null); setUnderstood([]); setSimilarTo(null); setResults([]); setSelected(null);
    try { const data = await request<{ results: Result[]; suggestion?: string | null; understood?: string[] }>(`/search?q=${encodeURIComponent(q.trim())}&k=${q.trim() ? 20 : 100}&mode=${m}${fp ? `&${fp}` : ''}`); setResults(data.results); setSelected(data.results[0] || null); setSuggestion(data.suggestion ?? null); setUnderstood(data.understood ?? []); }
    catch (e) { setResults([]); setSelected(null); setError(e instanceof Error ? e.message : 'Search failed'); }
    finally { setBusy(false); }
  }
  function search(e: React.FormEvent) { e.preventDefault(); void runSearch(mode); }
  // "More like this": files similar to one result (by meaning and distinctive words), within the current filters.
  async function moreLikeThis(path: string) {
    if (busy) return;
    const fp = filterParams(filters);
    setBusy(true); setError(''); setSearched(true); setPreviewOpen(false); setExpanded(false); setSuggestion(null); setUnderstood([]); setResults([]); setSelected(null); setSimilarTo(path);
    try { const data = await request<{ results: Result[] }>(`/similar?path=${encodeURIComponent(path)}&k=20${fp ? `&${fp}` : ''}`); setResults(data.results); setSelected(data.results[0] || null); }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not find similar files'); }
    finally { setBusy(false); }
  }
  function backToSearch() { setSimilarTo(null); if (query.trim()) void runSearch(mode); else { setResults([]); setSelected(null); setSearched(false); } }
  // Changing a filter re-runs the current search (debounced for typing in the folder box).
  function updateFilters(next: Filters) {
    setFilters(next);
    if (searched || filterParams(next)) { clearTimeout(filterTimer.current); filterTimer.current = setTimeout(() => void runSearch(mode, query, next), 250); }
  }
  const filterTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const sortedResults = sort === 'relevance' ? results : [...results].sort((a, b) =>
    sort === 'name' ? filename(a.path).localeCompare(filename(b.path)) : sort === 'size' ? b.size - a.size : b[sort] - a[sort]);
  const typeOptions = Object.entries(status?.file_types || {});
  function dateControl(label: string, key: 'modified' | 'added') {
    const r = filters[key];
    return <span className="filter">
      <select aria-label={label} value={r.preset} className={r.preset !== 'any' ? 'on' : ''} onChange={e => updateFilters({ ...filters, [key]: { ...r, preset: e.target.value as DatePreset } })}>
        {Object.entries(DATE_LABELS).map(([v, l]) => <option key={v} value={v}>{v === 'any' ? label : `${label}: ${l}`}</option>)}
      </select>
      {r.preset === 'custom' && <><input type="date" aria-label={`${label} from`} value={r.from} onChange={e => updateFilters({ ...filters, [key]: { ...r, from: e.target.value } })}/><span>–</span><input type="date" aria-label={`${label} to`} value={r.to} onChange={e => updateFilters({ ...filters, [key]: { ...r, to: e.target.value } })}/></>}
    </span>;
  }
  const filterBar = <div className="filter-bar">
    <details ref={typeMenuRef} className="filter type-filter">
      <summary className={filters.types.length ? 'on' : ''}>{filters.types.length ? `Type: ${filters.types.map(t => t.toUpperCase()).join(', ')}` : 'File type'}</summary>
      <div className="type-menu">{typeOptions.length ? typeOptions.map(([t, n]) => <label key={t}><input type="checkbox" checked={filters.types.includes(t)} onChange={e => updateFilters({ ...filters, types: e.target.checked ? [...filters.types, t] : filters.types.filter(x => x !== t) })}/><span>{t.toUpperCase()}</span><span className="count">{n.toLocaleString()}</span></label>) : <span className="count">No files indexed</span>}</div>
    </details>
    {dateControl('Modified', 'modified')}
    {dateControl('Date added', 'added')}
    <select aria-label="Size" value={filters.size} className={filters.size !== 'any' ? 'on' : ''} onChange={e => updateFilters({ ...filters, size: e.target.value as SizeKey })}>{Object.entries(SIZES).map(([v, [l]]) => <option key={v} value={v}>{v === 'any' ? 'Size' : `Size: ${l}`}</option>)}</select>
    <input className={`folder-filter ${filters.folder.trim() ? 'on' : ''}`} value={filters.folder} onChange={e => updateFilters({ ...filters, folder: e.target.value })} placeholder="In folder…" aria-label="Folder contains"/>
    {filtersOn && <button type="button" className="clear-filters" onClick={() => updateFilters(noFilters)}>Clear filters</button>}
  </div>;
  function acceptSuggestion() { if (suggestion) { setQuery(suggestion); void runSearch(mode, suggestion); } }
  // Switching tabs re-runs the current query so the two rankings can be compared directly.
  function switchMode(m: Mode) {
    if (m !== mode) { setMode(m); if (searched) void runSearch(m); }
    inputRef.current?.focus();
  }
  // The backend reports the indexed folder, so it shows up again after the app restarts.
  const folders = status?.folders?.length ? status.folders : folder ? [folder] : [];
  const currentFolder = folders[folders.length - 1] || '';
  const isHome = (f: string) => shortPath(f) === '~';
  const folderLabel = status?.whole_computer && folders.length === 1 ? 'Whole computer' : folders.length > 1 ? `${folders.length} folders` : currentFolder ? filename(currentFolder) : '';
  async function startIndex(body: { folder?: string; whole_computer?: boolean; mode?: 'add' | 'replace' }, label: string) {
    setBusy(true); setError('');
    // /index returns once files are read; poll meanwhile so the sidebar shows "Reading files…".
    const poll = setTimeout(() => void refreshStatus(), 300);
    try {
      await request('/index', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      setFolder(label); setResults([]); setSelected(null); setSearched(false); setSuggestion(null);
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not index folder'); }
    finally { clearTimeout(poll); setBusy(false); await refreshStatus(); }
  }
  // "add" indexes only the new folder's files next to what is already indexed; "replace" starts over with it.
  async function chooseFolder(mode: 'add' | 'replace') {
    let path: string | null = null;
    const title = mode === 'add' ? 'Add a folder to search' : 'Choose a folder to search instead';
    try {
      path = isTauri() ? await open({ directory: true, multiple: false, title, defaultPath: currentFolder || undefined }) : window.prompt('Enter the full path to a folder on this computer:', currentFolder || '');
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not open folder picker'); return; }
    if (!path) return;
    if (mode === 'replace' && status?.indexed_files && !window.confirm(`Search only this folder? The ${folders.length > 1 ? `${folders.length} folders` : 'folder'} indexed now will be removed from search.`)) return;
    await startIndex({ folder: path, mode }, path);
  }
  // Removing a folder drops only its files from search; the files on disk are untouched.
  async function removeFolder(f: string) {
    const name = isHome(f) ? 'your whole computer' : `“${filename(f)}”`;
    if (!window.confirm(`Stop searching ${name}? Its files stay on your computer; they just won't show up in ReFinder.`)) return;
    setBusy(true); setError('');
    try {
      await request('/folders/remove', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ folder: f }) });
      if (folder === f) setFolder('');
      setResults([]); setSelected(null); setSearched(false);
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not remove the folder'); }
    finally { setBusy(false); await refreshStatus(); }
  }
  async function indexWholeComputer() {
    if (!window.confirm('Add every supported document in your home folder? Apps, system, hidden and code-dependency folders are skipped. Files already indexed are not read again. This can take several minutes.')) return;
    await startIndex({ whole_computer: true, mode: 'add' }, '');
  }
  const statusText = busy && !indexing ? 'Working…' : !status ? (starting ? 'Starting search engine…' : 'Backend unavailable') : status.state === 'extracting' ? 'Reading files…' : status.state === 'updating' ? 'Updating changed files…' : status.state === 'embedding' ? `${status.indexed_files.toLocaleString()} files · understanding ${status.total_chunks ? Math.floor(embedPct) : 0}%` : status.state === 'error' ? (status.error || 'Indexing error') : status.indexed_files ? `${status.indexed_files.toLocaleString()} files indexed` : 'No folder indexed';
  const selectResult = (result: Result) => { setSelected(result); if (window.innerWidth < 820) setPreviewOpen(true); };
  async function openFile(path: string, reveal = false, app?: string) {
    try { await request('/open', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path, reveal, app }) }); }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not open the file'); }
  }
  // "Open With" menu, as in Finder: the default app, the other apps macOS offers for this file, then Other…
  async function toggleOpenWith() {
    if (openWith) { setOpenWith(null); return; }
    if (!selected) return;
    setOpenWith({ path: selected.path, default: null, apps: [], loading: true });
    try {
      const data = await request<{ default: App | null; apps: App[] }>(`/open-with?path=${encodeURIComponent(selected.path)}`);
      setOpenWith(cur => cur && cur.path === selected.path ? { ...cur, ...data, loading: false } : cur);
    } catch { setOpenWith(cur => cur && { ...cur, loading: false }); }
  }
  async function openWithOther(path: string) {
    setOpenWith(null);
    const app = isTauri()
      ? await open({ title: 'Choose an application', defaultPath: '/Applications', filters: [{ name: 'Applications', extensions: ['app'] }] })
      : window.prompt('Full path to an application (.app):', '/Applications/');
    if (typeof app === 'string' && app) await openFile(path, false, app);
  }
  const appItem = (path: string, app: App, isDefault = false) => <button key={app.path} role="menuitem" className="menu-item" title={app.path} onClick={() => { setOpenWith(null); void openFile(path, false, app.path); }}>
    <img src={`${API}/app-icon?app=${encodeURIComponent(app.path)}`} alt="" onError={e => { e.currentTarget.style.visibility = 'hidden'; }}/><span>{app.name}{isDefault && <span className="menu-hint"> (default)</span>}</span>
  </button>;
  const toggleExpanded = () => { const next = !expanded; setExpanded(next); setPreviewOpen(next); };
  const preview = <section className="preview panel" aria-label="File preview">
    <div className="preview-bar">
      <span className="preview-title" title={selected?.path}>{selected ? filename(selected.path) : 'Preview'}</span>
      {selected && <div className="preview-actions">
        <div className="segmented" role="tablist"><button role="tab" aria-selected={previewTab === 'document'} className={previewTab === 'document' ? 'active' : ''} onClick={() => setPreviewTab('document')}>Document</button><button role="tab" aria-selected={previewTab === 'text'} className={previewTab === 'text' ? 'active' : ''} onClick={() => setPreviewTab('text')}>Text</button></div>
        {previewTab === 'document' && ['pdf', 'docx'].includes(selected.type) && <div className="zoom"><button className="icon-button" onClick={() => setZoom(z => Math.max(0.5, +(z - 0.25).toFixed(2)))} aria-label="Zoom out" title="Zoom out">−</button><span>{Math.round(zoom * 100)}%</span><button className="icon-button" onClick={() => setZoom(z => Math.min(3, +(z + 0.25).toFixed(2)))} aria-label="Zoom in" title="Zoom in">+</button></div>}
        <button className="icon-button" onClick={() => void moreLikeThis(selected.path)} title="More like this" aria-label="More like this"><Icon name="similar"/></button>
        <button className="icon-button" onClick={() => void openFile(selected.path, true)} title="Show in Finder" aria-label="Show in Finder"><Icon name="reveal"/></button>
        <button className="icon-button" onClick={toggleExpanded} title={expanded ? 'Exit full view' : 'Full view'} aria-label={expanded ? 'Exit full view' : 'Full view'}><Icon name={expanded ? 'shrink' : 'expand'}/></button>
        <div className="open-split" ref={openWithRef}>
          <button className={`open-button ${IS_MAC ? 'main' : ''}`} onClick={() => void openFile(selected.path)} title="Open in default app (⌘O)">Open</button>
          {IS_MAC && <button className="open-button caret" onClick={() => void toggleOpenWith()} aria-haspopup="menu" aria-expanded={!!openWith} aria-label="Open with" title="Open with">
            <svg viewBox="0 0 10 6" aria-hidden="true"><path d="M1 1l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg></button>}
          {openWith && openWith.path === selected.path && <div className="open-menu" role="menu">
            <div className="menu-label">Open With</div>
            {openWith.loading && <div className="menu-hint menu-pad">Finding apps…</div>}
            {openWith.default && appItem(selected.path, openWith.default, true)}
            {openWith.default && openWith.apps.length > 0 && <div className="menu-sep"/>}
            {openWith.apps.map(a => appItem(selected.path, a))}
            {!openWith.loading && <><div className="menu-sep"/><button role="menuitem" className="menu-item other" onClick={() => void openWithOther(selected.path)}>Other…</button></>}
          </div>}
        </div>
      </div>}
      <button className="icon-button preview-close" onClick={() => { setPreviewOpen(false); setExpanded(false); }} aria-label="Close preview"><Icon name="close" /></button>
    </div>
    {selected ? <div className="preview-body">
      <div className="preview-meta"><span className="path">{directory(selected.path)}</span><span>{(query.trim() || similarTo) && <span className="score">{similarTo ? 'Similarity' : 'Relevance'} {selected.score.toFixed(3)} · </span>}Modified {formatDate(selected.modified)} · Added {formatDate(selected.added)} · {formatSize(selected.size)}</span></div>
      {previewTab === 'document'
        ? <DocumentView key={selected.path} api={API} path={selected.path} query={query} passage={selected.passage || selected.snippet} fallbackText={fileText} zoom={zoom}/>
        : fileText ? <TextBlocks text={fileText} query={query} passage={selected.passage || selected.snippet}/> : <p className="file-text">Loading extracted text…</p>}
    </div> : <div className="empty-preview">Select a result to preview the file.</div>}
  </section>;
  return <div className={`app ${collapsed ? 'sidebar-collapsed' : ''}`}>
    <aside className="sidebar"><div className="sidebar-head"><span className="brand">ReFinder</span><button className="icon-button collapse-button" title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} onClick={() => setCollapsed(!collapsed)}><Icon name="sidebar"/></button></div>
      <div className="sidebar-scroll"><p className="section-label">LIBRARY</p><button className={`nav-item ${mode === 'smart' ? 'active' : ''}`} title="Search" onClick={() => switchMode('smart')}><Icon name="search"/><span>Search</span></button><button className={`nav-item ${mode === 'keyword' ? 'active' : ''}`} title="Keyword search" onClick={() => switchMode('keyword')}><Icon name="keyword"/><span>Keyword search</span></button><p className="section-label folder-label">FOLDER</p>{folders.map(f => <div key={f} className="nav-item folder-item" title={f}><Icon name={isHome(f) ? 'computer' : 'folder'}/><span>{isHome(f) ? 'Whole computer' : filename(f)}</span><button className="folder-remove" onClick={() => void removeFolder(f)} disabled={busy} aria-label={`Stop searching ${filename(f)}`} title="Remove from search"><Icon name="close"/></button></div>)}<button className="nav-item" title="Add a folder to search" onClick={() => chooseFolder(folders.length ? 'add' : 'replace')} disabled={busy}><Icon name="plus"/><span>{folders.length ? 'Add folder' : 'Choose folder'}</span></button>{folders.length > 0 && <button className="nav-item" title="Search a different folder instead" onClick={() => chooseFolder('replace')} disabled={busy}><Icon name="swap"/><span>Replace all</span></button>}<button className="nav-item" title="Index whole computer" onClick={indexWholeComputer} disabled={busy}><Icon name="computer"/><span>Index whole computer</span></button></div>
      <div className="sidebar-bottom">{embedding && <div className="embed-progress" title={`${embedDone.toLocaleString()} of ${embedTotal.toLocaleString()} passages and images understood`}>
        <div className="embed-progress-label"><span>Understanding files</span><span>{Math.floor(embedPct)}%</span></div>
        <div className="progress-track" role="progressbar" aria-label="Understanding files" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.floor(embedPct)}><div className="progress-fill" style={{ width: `${embedPct}%` }}/></div>
        <div className="embed-progress-sub">{etaText || 'Keyword search works now; results get smarter as this finishes.'}</div>
      </div>}<div className="index-status" title={statusText}><span className={`status-dot ${status ? 'online' : ''}`}/><span>{embedding && !collapsed ? `${status!.indexed_files.toLocaleString()} files indexed` : statusText}</span></div><button className="nav-item" title="Settings" onClick={() => setSettingsOpen(true)}><Icon name="settings"/><span>Settings</span></button></div>
    </aside>
    <main className="workspace">{(embedding || status?.state === 'extracting' || status?.state === 'updating') && <div className={`top-progress ${embedding ? '' : 'indeterminate'}`} aria-hidden="true"><div style={embedding ? { width: `${embedPct}%` } : undefined}/></div>}<header className="page-heading"><h1>{mode === 'keyword' ? 'Keyword search' : 'Search'}</h1><span className="shortcut">⌘ K</span></header><form className="search-box" onSubmit={search}><Icon name="search"/><input ref={inputRef} value={query} onChange={e => setQuery(e.target.value)} placeholder={mode === 'keyword' ? 'Find files containing these words' : 'Search your files by meaning'} aria-label="Search your files"/><button type="submit" disabled={busy || (!query.trim() && !filtersOn)}>Search</button></form>
      {filterBar}
      <div className="scope-row"><span title={folders.join('\n')}>{folderLabel || 'All indexed files'}</span><span>{busy ? 'Searching or indexing…' : statusText}</span></div>
      {error && <div className="error" role="alert">{error === 'Failed to fetch' ? 'Could not connect to the local ReFinder server. Start the backend and try again.' : error}</div>}
      {mode === 'keyword' && suggestion && !busy && <div className="suggestion">Did you mean <button type="button" onClick={acceptSuggestion}>{suggestion}</button>?</div>}
      {(understood.length > 0 || similarTo) && <div className="understood">
        {similarTo ? <><span>More like <strong title={similarTo}>{filename(similarTo)}</strong></span><button type="button" className="link-button" onClick={backToSearch}>{query.trim() ? 'Back to search' : 'Clear'}</button></>
          : <><span>Searching for</span>{understood.map(u => <span key={u} className="chip">{u}</span>)}</>}
      </div>}
      <div className="results-heading"><span>{busy && searched ? 'Searching…' : searched ? `${results.length} ${results.length === 1 ? 'result' : 'results'}` : 'Your results'}</span><label className="sort">Sort by <select value={sort} onChange={e => setSort(e.target.value as SortKey)}><option value="relevance">{query.trim() ? (mode === 'keyword' ? 'Keyword match' : 'Relevance') : 'Newest modified'}</option><option value="modified">Date modified</option><option value="added">Date added</option><option value="name">Name</option><option value="size">Size</option></select></label></div>
      <div className="content-grid"><section className="results panel" aria-label="Search results">{sortedResults.length ? sortedResults.map((result, i) => <button key={`${result.path}-${result.chunk_id}-${i}`} className={`result ${selected === result ? 'selected' : ''}`} onClick={() => selectResult(result)} onDoubleClick={() => void openFile(result.path)} title="Double-click to open"><span className="file-icon">{typeLabel(result.path)}</span><span className="result-content"><strong>{filename(result.path)}</strong><span className="path">{directory(result.path)}</span><span className="snippet">{result.snippet}</span><span className="meta">{(query.trim() || similarTo) && <span className="score">{similarTo ? 'Similarity' : 'Relevance'} {result.score.toFixed(3)}</span>}<span>Modified {formatDate(result.modified)}</span><span>Added {formatDate(result.added)}</span><span>{formatSize(result.size)}</span></span></span></button>) : <div className="empty-state"><div className="empty-icon"><Icon name="search"/></div><h2>{busy ? 'Searching…' : searched ? 'No results found' : 'Find what you remember'}</h2><p>{busy ? 'This may take a moment while the model loads.' : searched ? (filtersOn ? 'No files match these filters. Try loosening or clearing them.' : mode === 'keyword' ? 'No file contains these words. Check the spelling or try Search.' : 'Try a different description or choose another folder.') : status?.indexed_files ? 'Describe the content you’re looking for.' : 'Choose a folder, then describe the content you’re looking for.'}</p>{!status?.indexed_files && <button className="primary-button" onClick={() => chooseFolder('replace')} disabled={busy}>Choose folder</button>}</div>}</section>{previewOpen ? <section className="preview panel"><div className="empty-preview">Showing in full view.</div></section> : preview}</div>
      <div className="compact-footer"><span>{selected ? filename(selected.path) : 'Select a result to preview'}</span><span className="footer-actions"><button onClick={() => selected && void openFile(selected.path)} disabled={!selected}>Open <span>⌘ O</span></button><button onClick={() => setPreviewOpen(true)} disabled={!selected}>Preview <span>⌘ Y</span></button></span></div>
    </main>
    {previewOpen && <div className={`preview-overlay ${expanded ? 'wide' : ''}`} onClick={() => { setPreviewOpen(false); setExpanded(false); }}><div className="preview-sheet" onClick={e => e.stopPropagation()}>{preview}</div></div>}
    {settingsOpen && <div className="modal-backdrop" onClick={() => setSettingsOpen(false)}><section className="settings panel" role="dialog" aria-modal="true" aria-labelledby="settings-title" onClick={e => e.stopPropagation()}><div className="settings-header"><h2 id="settings-title">Settings</h2><button className="icon-button" onClick={() => setSettingsOpen(false)} aria-label="Close settings"><Icon name="close"/></button></div><label htmlFor="theme">Appearance</label><select id="theme" value={theme} onChange={e => setTheme(e.target.value as Theme)}><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select><p>ReFinder uses the local search index on this computer.</p></section></div>}
  </div>;
}
export default App;
