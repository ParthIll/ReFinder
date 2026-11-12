import { useEffect, useMemo, useRef, useState } from 'react';
import * as pdfjs from 'pdfjs-dist';
import pdfWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import { isTauri } from '@tauri-apps/api/core';
import DOMPurify from 'dompurify';
import { marked } from 'marked';
import { bestMatch, scrollToMiddle } from './highlight';

pdfjs.GlobalWorkerOptions.workerSrc = pdfWorker;

type Sheet = { name: string; rows: string[][]; truncated: boolean };
type Slide = { number: number; text: string };
export type PreviewData =
  | { kind: 'pdf' }
  | { kind: 'docx' }
  | { kind: 'html'; key: string; slides: Slide[] }
  | { kind: 'table'; sheets: Sheet[] }
  | { kind: 'slides'; slides: Slide[]; thumbnail: boolean }
  | { kind: 'text'; text: string; markdown: boolean };

const MAX_PDF_PAGES = 300;
const STOP = new Set('the and for with that this from what where when which who how are was were has have had you your our their about into find file files show me my a an of in on to is it at by or as be'.split(' '));
const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Text with the query's words highlighted. */
export function Highlighted({ text, query }: { text: string; query: string }) {
  const pattern = useMemo(() => {
    const words = [...new Set(query.toLowerCase().match(/[\p{L}\p{N}]+/gu) || [])].filter(w => w.length >= 3 && !STOP.has(w));
    return words.length ? new RegExp(`(${words.map(escapeRegExp).join('|')})`, 'gi') : null;
  }, [query]);
  if (!pattern) return <>{text}</>;
  return <>{text.split(pattern).map((part, i) => i % 2 ? <mark key={i}>{part}</mark> : part)}</>;
}

/** Plain text split into paragraphs; the ones holding the matched passage are highlighted and scrolled to. */
export function TextBlocks({ text, query, passage, className = '' }: { text: string; query: string; passage: string; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const paragraphs = useMemo(() => text.split(/\n\s*\n/), [text]);
  const hits = useMemo(() => new Set(passage ? bestMatch(passage, paragraphs)?.units || [] : []), [passage, paragraphs]);
  useEffect(() => { scrollToMiddle(ref.current?.querySelector('.passage-hit')); }, [hits]);
  return <div ref={ref} className={`file-text ${className}`}>{paragraphs.map((p, i) => <p key={i} className={hits.has(i) ? 'passage-hit' : ''}><Highlighted text={p} query={query}/></p>)}</div>;
}

/** `rel` resolved against the folder of `file` ("img/a.png", "../b.png"); POSIX or Windows separators. */
function resolvePath(file: string, rel: string): string {
  const sep = file.includes('\\') && !file.includes('/') ? '\\' : '/';
  const parts = file.split(/[\\/]/).slice(0, -1);
  for (const seg of decodeURIComponent(rel).split(/[\\/]/)) {
    if (seg === '..') parts.pop(); else if (seg && seg !== '.') parts.push(seg);
  }
  return parts.join(sep);
}

/** Wraps the query's words in <mark> inside an element's text (skipping code blocks). */
function markQueryWords(root: HTMLElement, query: string) {
  const words = [...new Set(query.toLowerCase().match(/[\p{L}\p{N}]+/gu) || [])].filter(w => w.length >= 3 && !STOP.has(w));
  if (!words.length) return;
  const pattern = new RegExp(`(${words.map(escapeRegExp).join('|')})`, 'gi');
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode: n => n.parentElement?.closest('pre, code, mark') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT });
  const nodes: Text[] = [];
  while (walker.nextNode()) nodes.push(walker.currentNode as Text);
  for (const node of nodes) {
    const parts = node.data.split(pattern);
    if (parts.length < 2) continue;
    const frag = document.createDocumentFragment();
    parts.forEach((part, i) => { if (!part) return; if (i % 2) { const m = document.createElement('mark'); m.textContent = part; frag.append(m); } else frag.append(part); });
    node.replaceWith(frag);
  }
}

/**
 * Markdown rendered as formatted text (GitHub flavour: tables, task lists, code). The HTML is sanitized, so
 * scripts or event handlers in a file never run. Local images load through the backend; remote images are not
 * fetched (search stays offline). Links open in the browser instead of navigating the app.
 */
function MarkdownView({ api, path, text, query, passage }: { api: string; path: string; text: string; query: string; passage: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const html = useMemo(() => DOMPurify.sanitize(marked.parse(text, { gfm: true, async: false }) as string, { USE_PROFILES: { html: true } }), [text]);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    el.innerHTML = html;
    el.querySelectorAll('img').forEach(img => {
      const src = img.getAttribute('src') || '';
      if (/^data:image\//i.test(src)) return;
      if (/^[a-z][a-z0-9+.-]*:/i.test(src) || src.startsWith('//')) {
        const note = document.createElement('span'); note.className = 'md-image-blocked';
        note.textContent = `[image: ${img.alt || src}]`; img.replaceWith(note); return;
      }
      img.src = `${api}/asset?path=${encodeURIComponent(resolvePath(path, src.split(/[?#]/)[0]))}`;
      img.loading = 'lazy';
    });
    markQueryWords(el, query);
    if (!passage) return;
    const blocks = [...el.querySelectorAll<HTMLElement>('p, li, h1, h2, h3, h4, h5, h6, td, th, pre, blockquote')].filter(b => !b.querySelector('p, li, td, pre'));
    const m = bestMatch(passage, blocks.map(b => b.textContent || ''));
    m?.units.forEach(i => { if (blocks[i].textContent?.trim()) blocks[i].classList.add('passage-hit'); });
    if (m) scrollToMiddle(blocks[m.units[0]]);
  }, [html, api, path, query, passage]);
  async function onClick(e: React.MouseEvent) {
    const a = (e.target as HTMLElement).closest('a'); if (!a) return;
    e.preventDefault();
    const href = a.getAttribute('href') || '';
    if (href.startsWith('#')) { scrollToMiddle(ref.current?.querySelector(`[id="${CSS.escape(decodeURIComponent(href.slice(1)))}"]`)); return; }
    if (/^(https?:|mailto:)/i.test(href)) {
      if (isTauri()) { const { openUrl } = await import('@tauri-apps/plugin-opener'); await openUrl(href); } else window.open(href, '_blank', 'noopener');
    }
  }
  return <div ref={ref} className="markdown-body" onClick={e => void onClick(e)}/>;
}

type PageHolder = HTMLDivElement & { draw: () => Promise<HTMLElement[]> };

/**
 * Every page of a PDF (canvas + selectable text layer), each drawn when scrolled near, fit to the panel width times
 * `zoom`. The page and lines holding the matched passage are found from the text and highlighted.
 */
function PdfView({ url, zoom, passage, onError }: { url: string; zoom: number; passage: string; onError: (e: string) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [pages, setPages] = useState(0);
  const [matchPage, setMatchPage] = useState(0);
  useEffect(() => {
    const container = ref.current; if (!container) return;
    let cancelled = false;
    // Fonts, CMaps and decoders ship in public/pdfjs (see scripts/copy-pdfjs-assets.mjs); without them text in
    // PDFs that don't embed their fonts (common in LaTeX and older files) doesn't render.
    const assets = new URL('pdfjs/', document.baseURI).href;
    const task = pdfjs.getDocument({ url, standardFontDataUrl: `${assets}standard_fonts/`, cMapUrl: `${assets}cmaps/`, cMapPacked: true, wasmUrl: `${assets}wasm/`, iccUrl: `${assets}iccs/` });
    const observer = new IntersectionObserver(entries => entries.forEach(e => {
      if (e.isIntersecting) { observer.unobserve(e.target); void (e.target as PageHolder).draw(); }
    }), { root: container.closest('.preview-body'), rootMargin: '600px 0px' });
    container.replaceChildren(); setMatchPage(0);
    task.promise.then(async doc => {
      if (cancelled) return;
      setPages(doc.numPages);
      const width = Math.max(200, container.clientWidth) * zoom;
      const ratio = window.devicePixelRatio || 1;
      const holders: PageHolder[] = [];
      for (let n = 1; n <= Math.min(doc.numPages, MAX_PDF_PAGES) && !cancelled; n++) {
        const page = await doc.getPage(n);
        const viewport = page.getViewport({ scale: width / page.getViewport({ scale: 1 }).width });
        const holder = document.createElement('div') as PageHolder;
        holder.className = 'pdf-page';
        holder.style.width = `${viewport.width}px`; holder.style.height = `${viewport.height}px`;
        holder.style.setProperty('--scale-factor', String(viewport.scale));
        let drawn: Promise<HTMLElement[]> | null = null;
        holder.draw = () => drawn ??= (async () => {
          const canvas = document.createElement('canvas');
          canvas.width = Math.floor(viewport.width * ratio); canvas.height = Math.floor(viewport.height * ratio);
          canvas.style.width = `${viewport.width}px`; canvas.style.height = `${viewport.height}px`;
          const layer = document.createElement('div'); layer.className = 'textLayer';
          holder.append(canvas, layer);
          void page.render({ canvas, viewport, transform: ratio !== 1 ? [ratio, 0, 0, ratio, 0, 0] : undefined }).promise.catch(() => {});
          const text = new pdfjs.TextLayer({ textContentSource: await page.getTextContent(), container: layer, viewport });
          await text.render();
          return text.textDivs;
        })();
        container.appendChild(holder); holders.push(holder);
        observer.observe(holder);
      }
      if (!passage || cancelled) return;
      // Pick the page whose text best matches the passage, draw it, and highlight the matching lines.
      let best = { score: 0, page: -1 };
      for (let i = 0; i < holders.length && !cancelled; i++) {
        const items = (await (await doc.getPage(i + 1)).getTextContent()).items.map(it => ('str' in it ? it.str : ''));
        const m = bestMatch(passage, items);
        if (m && m.score > best.score) best = { score: m.score, page: i };
      }
      if (best.page < 0 || cancelled) return;
      const divs = await holders[best.page].draw();
      const m = bestMatch(passage, divs.map(d => d.textContent || ''));
      if (!m || cancelled) return;
      m.units.forEach(i => divs[i].classList.add('hit'));
      setMatchPage(best.page + 1);
      scrollToMiddle(divs[m.units[0]]);
    }).catch(e => { if (!cancelled) onError(e?.message || 'Could not render this PDF'); });
    return () => { cancelled = true; observer.disconnect(); void task.destroy(); };
  }, [url, zoom, passage, onError]);
  return <div className="pdf-view">
    {matchPage > 0 && <p className="match-note">Best match highlighted on page {matchPage} of {pages}</p>}
    <div ref={ref} className="pdf-pages"/>
    {pages > MAX_PDF_PAGES && <p className="preview-note">Showing the first {MAX_PDF_PAGES} of {pages} pages. Open the file to see the rest.</p>}
  </div>;
}

/** A Word document rendered to HTML (layout, tables, images), scaled to fit the panel; matched paragraphs highlighted. */
function DocxView({ url, zoom, passage, onError }: { url: string; zoom: number; passage: string; onError: (e: string) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const container = ref.current; if (!container) return;
    const controller = new AbortController();
    container.replaceChildren();
    (async () => {
      const [{ renderAsync }, response] = await Promise.all([import('docx-preview'), fetch(url, { signal: controller.signal })]);
      if (!response.ok) throw new Error(`Could not load the document (${response.status})`);
      const blob = await response.blob();
      if (controller.signal.aborted) return;
      await renderAsync(blob, container, undefined, { inWrapper: true, ignoreWidth: false, breakPages: true, className: 'docx' });
      const page = container.querySelector<HTMLElement>('section.docx');
      const fit = page ? (container.parentElement!.clientWidth - 24) / page.offsetWidth : 1;
      container.style.zoom = String(Math.min(1.5, fit * zoom));
      if (!passage) return;
      const blocks = [...container.querySelectorAll<HTMLElement>('p, li, td')].filter(b => !b.querySelector('p, li, td'));
      const m = bestMatch(passage, blocks.map(b => b.textContent || ''));
      // Skip blocks with no text (a paragraph holding only an image) that fall between matching lines.
      m?.units.forEach(i => { if (blocks[i].textContent?.trim()) blocks[i].classList.add('passage-hit'); });
      if (m) scrollToMiddle(blocks[m.units[0]]);
    })().catch(e => { if (e?.name !== 'AbortError') onError(e?.message || 'Could not render this document'); });
    return () => controller.abort();
  }, [url, zoom, passage, onError]);
  return <div className="docx-view"><div ref={ref}/></div>;
}

/**
 * A Quick Look HTML render (every slide of a deck, with layout and images) in a script-free iframe. The slides
 * are scaled to the panel width and the paragraphs matching the passage are highlighted.
 */
function HtmlView({ base, passage, onError }: { base: string; passage: string; onError: (e: string) => void }) {
  const frame = useRef<HTMLIFrameElement>(null);
  const [srcDoc, setSrcDoc] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    setSrcDoc('');
    fetch(`${base}index.html`, { signal: controller.signal })
      .then(r => { if (!r.ok) throw new Error(`Could not load the slides (${r.status})`); return r.text(); })
      .then(html => setSrcDoc(html.replace(/<head>/i, `<head><base href="${base}"><style>
        body{margin:0;padding:8px 0;background:transparent}div.slide{margin:0 auto 12px!important;box-shadow:0 1px 4px #0003!important}
        .passage-hit{background:rgba(255,200,0,.45)!important;border-radius:3px}</style>`)))
      .catch(e => { if (e.name !== 'AbortError') onError(e.message); });
    return () => controller.abort();
  }, [base, onError]);
  function onLoad() {
    const doc = frame.current?.contentDocument; if (!doc?.body) return;
    const fit = () => {
      const slide = doc.querySelector<HTMLElement>('div.slide');
      const width = frame.current!.clientWidth - 16;
      if (slide) doc.body.style.zoom = String(Math.min(1, width / slide.offsetWidth));
    };
    fit();
    if (!passage) return;
    const blocks = [...doc.querySelectorAll<HTMLElement>('div.slide p, div.slide td')];
    const m = bestMatch(passage, blocks.map(b => b.textContent || ''));
    m?.units.forEach(i => { if (blocks[i].textContent?.trim()) blocks[i].classList.add('passage-hit'); });
    if (m) scrollToMiddle(blocks[m.units[0]]);
  }
  return srcDoc ? <iframe ref={frame} className="html-view" title="Slides" sandbox="allow-same-origin" srcDoc={srcDoc} onLoad={onLoad}/> : <p className="preview-note">Rendering slides…</p>;
}

function TableView({ sheets, passage }: { sheets: Sheet[]; passage: string }) {
  // Open on the sheet that best matches the passage, with the matching rows highlighted.
  const matches = useMemo(() => sheets.map(s => passage ? bestMatch(passage, s.rows.map(r => r.join(' '))) : null), [sheets, passage]);
  const initial = useMemo(() => matches.reduce((best, m, i) => (m && m.score > (matches[best]?.score || 0) ? i : best), 0), [matches]);
  const [active, setActive] = useState(initial);
  const ref = useRef<HTMLDivElement>(null);
  const sheet = sheets[Math.min(active, sheets.length - 1)];
  const hits = new Set(matches[active]?.units || []);
  useEffect(() => { scrollToMiddle(ref.current?.querySelector('tr.passage-hit')); }, [active]);
  if (!sheet) return <p className="preview-note">This spreadsheet is empty.</p>;
  const width = Math.max(0, ...sheet.rows.map(r => r.length));
  return <div className="table-view">
    {sheets.length > 1 && <div className="sheet-tabs">{sheets.map((s, i) => <button key={s.name + i} className={i === active ? 'active' : ''} onClick={() => setActive(i)}>{s.name}</button>)}</div>}
    <div ref={ref} className="table-scroll"><table><tbody>{sheet.rows.map((row, r) => <tr key={r} className={`${r === 0 ? 'head' : ''} ${hits.has(r) ? 'passage-hit' : ''}`}><th>{r + 1}</th>{Array.from({ length: width }, (_, c) => <td key={c}>{row[c] ?? ''}</td>)}</tr>)}</tbody></table></div>
    {sheet.truncated && <p className="preview-note">Showing the first {sheet.rows.length} rows. Open the file to see everything.</p>}
  </div>;
}

function SlidesView({ slides, thumbnailUrl, query, passage }: { slides: Slide[]; thumbnailUrl: string | null; query: string; passage: string }) {
  const [thumbOk, setThumbOk] = useState(true);
  const ref = useRef<HTMLDivElement>(null);
  const hits = useMemo(() => new Set(passage ? bestMatch(passage, slides.map(s => s.text))?.units || [] : []), [passage, slides]);
  useEffect(() => { scrollToMiddle(ref.current?.querySelector('.passage-hit')); }, [hits]);
  return <div ref={ref} className="slides-view">
    {thumbnailUrl && thumbOk && <img className="slide-thumb" src={thumbnailUrl} alt="First slide" onError={() => setThumbOk(false)}/>}
    {slides.map((s, i) => <div key={s.number} className={`slide-card ${hits.has(i) ? 'passage-hit' : ''}`}><span className="slide-number">Slide {s.number || '–'}</span><p><Highlighted text={s.text} query={query}/></p></div>)}
    {!slides.length && <p className="preview-note">No text found on these slides.</p>}
  </div>;
}

/** The file itself, rendered by type; falls back to the extracted text if rendering fails. */
export function DocumentView({ api, path, query, passage, fallbackText, zoom }: { api: string; path: string; query: string; passage: string; fallbackText: string; zoom: number }) {
  const [data, setData] = useState<PreviewData | null>(null);
  const [error, setError] = useState('');
  const q = encodeURIComponent(path);
  useEffect(() => {
    const controller = new AbortController();
    setData(null); setError('');
    fetch(`${api}/preview?path=${q}`, { signal: controller.signal })
      .then(async r => { if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `Preview failed (${r.status})`); return r.json(); })
      .then(setData).catch(e => { if (e.name !== 'AbortError') setError(e.message); });
    return () => controller.abort();
  }, [api, q]);
  const onError = useMemo(() => (e: string) => setError(e), []);
  if (error) return <div><p className="preview-note error-note">{error}. Showing the extracted text instead.</p><TextBlocks text={fallbackText} query={query} passage={passage}/></div>;
  if (!data) return <p className="preview-note">Loading preview…</p>;
  if (data.kind === 'pdf') return <PdfView url={`${api}/raw?path=${q}`} zoom={zoom} passage={passage} onError={onError}/>;
  if (data.kind === 'docx') return <DocxView url={`${api}/raw?path=${q}`} zoom={zoom} passage={passage} onError={onError}/>;
  if (data.kind === 'html') return <HtmlView base={new URL(`${api}/quicklook/${data.key}/`, document.baseURI).href} passage={passage} onError={onError}/>;
  if (data.kind === 'table') return <TableView sheets={data.sheets} passage={passage}/>;
  if (data.kind === 'slides') return <SlidesView slides={data.slides} thumbnailUrl={data.thumbnail ? `${api}/thumbnail?path=${q}` : null} query={query} passage={passage}/>;
  if (data.markdown) return <MarkdownView api={api} path={path} text={data.text} query={query} passage={passage}/>;
  return <TextBlocks text={data.text} query={query} passage={passage}/>;
}
