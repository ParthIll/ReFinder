#!/usr/bin/env bash
# Builds the backend into a self-contained folder the desktop app ships and launches:
#   frontend/src-tauri/backend/refinder-backend/refinder-backend
# Needs the project venv (pip install -r requirements.txt pyinstaller). The embedding model is not bundled; it is
# downloaded to the Hugging Face cache on first use, like when running from source.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
OUT=frontend/src-tauri/backend
rm -rf build/pyinstaller "$OUT"
pyinstaller packaging/refinder_backend.py \
  --name refinder-backend --onedir --noconfirm --clean \
  --distpath "$OUT" --workpath build/pyinstaller --specpath build/pyinstaller \
  --paths . \
  --add-data "$PWD/refinder-extractors:refinder-extractors" \
  --hidden-import evaluation.server --hidden-import evaluation.database --hidden-import evaluation.preview \
  --hidden-import evaluation.watcher --hidden-import evaluation.query_meta --hidden-import evaluation.custom --hidden-import evaluation.common \
  --hidden-import refinder.embedder --hidden-import refinder.extract \
  --hidden-import pypdf --hidden-import docx --hidden-import pptx --hidden-import openpyxl \
  --collect-all chromadb --collect-all sentence_transformers --collect-submodules uvicorn \
  --collect-data bm25s --collect-binaries Stemmer --collect-submodules watchdog \
  --exclude-module matplotlib --exclude-module tkinter --exclude-module IPython --exclude-module jupyter
echo "Built $OUT/refinder-backend ($(du -sh "$OUT/refinder-backend" | cut -f1))"
