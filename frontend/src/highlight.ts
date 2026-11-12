// Finds where a search result's matched passage sits in a rendered document, so the preview can highlight it.
// The passage comes from the backend's text extraction, which never lines up exactly with what pdf.js, docx-preview
// or Quick Look render (spacing, hyphenation, line breaks), so matching is by words, not characters.

const STOP = new Set(('a an and are as at be but by for from has have he her his i if in into is it its of on or our she ' +
  'so that the their them then there these they this to was we were what when which who will with you your').split(' '));
const MAX_PASSAGE_WORDS = 150;

export const wordsOf = (s: string) => s.toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];

export type Match = { score: number; units: number[] };

/**
 * `units` are the document's text pieces in reading order (PDF text items, paragraphs, table rows…). Slides a window
 * as long as the passage over the document's words, scoring each by how many of its words (ignoring stopwords)
 * occur in the passage, and returns the units covering the best window, trimmed to its first and last hit.
 * Null when nothing overlaps enough to be the same text.
 */
export function bestMatch(passage: string, units: string[]): Match | null {
  const pw = wordsOf(passage).slice(0, MAX_PASSAGE_WORDS);
  const key = new Set(pw.filter(w => w.length > 1 && !STOP.has(w)));
  if (key.size < 2) return null;
  const doc: { hit: boolean; unit: number }[] = [];
  units.forEach((u, i) => { for (const w of wordsOf(u)) doc.push({ hit: key.has(w), unit: i }); });
  if (!doc.length) return null;
  const len = Math.min(pw.length, doc.length);
  let hits = 0, best = -1, bestStart = 0;
  for (let i = 0; i < doc.length; i++) {
    if (doc[i].hit) hits++;
    if (i >= len && doc[i - len].hit) hits--;
    if (i >= len - 1 && hits > best) { best = hits; bestStart = i - len + 1; }
  }
  const needed = Math.max(2, Math.min(6, Math.ceil(key.size * 0.35)));
  if (best < needed) return null;
  let first = bestStart, last = bestStart + len - 1;
  while (first < last && !doc[first].hit) first++;
  while (last > first && !doc[last].hit) last--;
  const out = new Set<number>();
  for (let i = first; i <= last; i++) out.add(doc[i].unit);
  return { score: best, units: [...out] };
}

/**
 * Scrolls an element to the middle of its nearest scrolling container only. (scrollIntoView would also scroll
 * every ancestor, including overflow:hidden app frames and the page around an iframe.)
 */
export function scrollToMiddle(el: Element | null | undefined) {
  if (!el) return;
  const view = el.ownerDocument.defaultView || window;
  let box: Element | null = el.parentElement;
  while (box && !(/(auto|scroll)/.test(view.getComputedStyle(box).overflowY) && box.scrollHeight > box.clientHeight)) box = box.parentElement;
  box ??= el.ownerDocument.scrollingElement;
  if (!box) return;
  const r = el.getBoundingClientRect();
  const top = box === el.ownerDocument.scrollingElement ? 0 : box.getBoundingClientRect().top;
  box.scrollTo({ top: box.scrollTop + r.top - top - Math.max(0, (box.clientHeight - r.height) / 2), behavior: 'smooth' });
}
