# ReFinder desktop frontend

The desktop interface uses React, Vite, and Tauri. It connects to the local ReFinder API at `127.0.0.1:8000`; it does not start the Python server itself.

From the repository root, start the backend using your configured Python environment:

```sh
uvicorn evaluation.server:app --host 127.0.0.1 --port 8000
```

In another terminal, run the desktop app:

```sh
cd frontend
npm install
npm run tauri dev
```

For browser-only UI development, use `npm run dev`. The browser version asks for a folder path instead of opening the native folder picker. The backend currently keeps one indexed folder at a time; choosing another folder replaces the existing index. The first search or indexing operation may take longer while the embedding model loads.

Use Settings to choose light, dark, or system appearance. The sidebar can be collapsed, and narrow windows show previews in a separate panel. Search uses Enter or the Search button; `⌘K` (or `Ctrl+K`) focuses the search field and `⌘Y` (or `Ctrl+Y`) opens the selected preview in a narrow window.
