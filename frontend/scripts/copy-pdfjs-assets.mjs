// Copies pdf.js runtime data into public/pdfjs so the viewer can load it offline: standard fonts (PDFs that use
// Helvetica/Times without embedding them), CMaps (CJK text), wasm image decoders (JPEG 2000, colour profiles)
// and ICC profiles. Runs after `npm install` and before builds; the copy is gitignored.
import { cpSync, existsSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'node_modules', 'pdfjs-dist');
const dest = join(root, 'public', 'pdfjs');
if (!existsSync(src)) process.exit(0);
rmSync(dest, { recursive: true, force: true });
for (const dir of ['standard_fonts', 'cmaps', 'wasm', 'iccs']) cpSync(join(src, dir), join(dest, dir), { recursive: true });
console.log(`pdf.js assets copied to ${dest}`);
