"""OCR fallback image assets and the matched MobileCLIP2-S0 image/text encoders.

Images are extracted by the format readers, not rendered as full document pages.
No model is loaded during text extraction. Assets and vectors are content-addressed
so repeated images and interrupted/restarted indexing reuse the same work.
"""
from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED = "dfndr2b"
CACHE_ID = f"{MODEL_NAME}:{PRETRAINED}:fp32:v1"
IMAGE_DIM = 512
IMAGE_BATCH = 16
VISUAL_SUFFIXES = frozenset({".pdf", ".docx", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"})


def cache_dir() -> Path:
    base = (Path(os.environ["REFINDER_DATA_DIR"]) if os.environ.get("REFINDER_DATA_DIR")
            else Path(__file__).resolve().parent.parent / ".cache")
    return base / "visual"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(data)
            f.flush()
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def capture_asset(image: Image.Image, location: dict, info: dict) -> dict:
    """Copy a fallback image to the local cache, preserving its parent-file location."""
    upright = ImageOps.exif_transpose(image)
    if "A" in upright.getbands() or "transparency" in upright.info:
        rgba = upright.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        rgba.close()
    else:
        rgb = upright.convert("RGB")
    upright.close()
    try:
        # The encoder consumes 256px crops; cap oversized scans without changing aspect ratio.
        rgb.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        digest = hashlib.sha256(f"{rgb.size}:RGB".encode() + rgb.tobytes()).hexdigest()
        path = cache_dir() / "images" / f"{digest}.png"
        if not path.is_file():
            buffer = io.BytesIO()
            rgb.save(buffer, format="PNG")
            _atomic_write(path, buffer.getvalue())
    finally:
        rgb.close()
    return {"digest": digest, **location, **info}


def asset_path(asset: dict) -> Path:
    digest = asset["digest"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid visual asset digest")
    return cache_dir() / "images" / f"{digest}.png"


def describe_asset(asset: dict) -> str:
    return f"Image {asset.get('image', 1)}" + (f", page {asset['page']}" if "page" in asset else "")


class VisualEmbedder:
    model_name = MODEL_NAME
    cache_id = CACHE_ID
    dim = IMAGE_DIM

    def __init__(self, device: str | None = None):
        import open_clip
        from timm.utils import reparameterize_model
        from refinder.embedder import pick_device

        self.device = device or pick_device()
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, cache_dir=str(cache_dir() / "models"))
        self.model = reparameterize_model(self.model.eval()).eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    def encode_query(self, query: str) -> np.ndarray:
        import torch
        from torch.nn.functional import normalize

        with torch.inference_mode():
            tokens = self.tokenizer([query]).to(self.device)
            return normalize(self.model.encode_text(tokens).float(), dim=-1)[0].cpu().numpy()

    def encode_assets(self, assets: list[dict]) -> np.ndarray:
        """Encode at most IMAGE_BATCH assets; reuse vectors across files and process restarts."""
        import torch
        from torch.nn.functional import normalize

        if not assets:
            return np.empty((0, self.dim), dtype=np.float32)
        if len(assets) > IMAGE_BATCH:
            return np.concatenate([self.encode_assets(assets[i:i+IMAGE_BATCH]) for i in range(0, len(assets), IMAGE_BATCH)])
        version = hashlib.sha256(self.cache_id.encode()).hexdigest()[:16]
        vectors: dict[str, np.ndarray] = {}
        pending = {}
        for asset in assets:
            asset_path(asset)  # validate the digest even when a cached vector already exists
            digest = asset["digest"]
            file = cache_dir() / "vectors" / version / f"{digest}.npy"
            try:
                vector = np.load(file, allow_pickle=False)
                if (vector.shape != (self.dim,) or not np.isfinite(vector).all()
                        or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4)):
                    raise ValueError("Invalid cached image vector")
                vectors[digest] = vector.astype(np.float32)
            except (OSError, ValueError):
                pending[digest] = asset
        if pending:
            tensors = []
            for asset in pending.values():
                with Image.open(asset_path(asset)) as image:
                    tensors.append(self.transform(image.convert("RGB")))
            with torch.inference_mode():
                batch = torch.stack(tensors).to(self.device)
                encoded = normalize(self.model.encode_image(batch).float(), dim=-1).cpu().numpy()
            for digest, vector in zip(pending, encoded):
                vectors[digest] = vector
                buffer = io.BytesIO()
                np.save(buffer, vector, allow_pickle=False)
                _atomic_write(cache_dir() / "vectors" / version / f"{digest}.npy", buffer.getvalue())
        return np.stack([vectors[a["digest"]] for a in assets])
