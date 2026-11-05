"""Runs an embedding model through ONNX Runtime on CPU, exporting it to .onnx on first use."""

import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm
from transformers import AutoModel

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "onnx"


class _LastHiddenState(torch.nn.Module):
    """Makes the model return a plain tensor, which is what the ONNX exporter needs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state


def _export(model_name: str, tokenizer, path: Path) -> None:
    """Save the model as float32 ONNX. Writes to a temp file first so a crash can't leave a broken model behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32, attn_implementation="eager").eval()
    x = tokenizer(["hello world this is a test", "short"], return_tensors="pt", padding=True)
    axes = {name: {0: "batch", 1: "tokens"} for name in ("input_ids", "attention_mask", "last_hidden_state")}
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")  # the exporter warns about every Python branch it traces
        torch.onnx.export(_LastHiddenState(model), (x["input_ids"], x["attention_mask"]), str(tmp), dynamo=False,
                          opset_version=18, dynamic_axes=axes, input_names=["input_ids", "attention_mask"],
                          output_names=["last_hidden_state"])
    tmp.replace(path)


def load_session(model_name: str, tokenizer) -> ort.InferenceSession:
    path = CACHE_DIR / model_name.replace("/", "--") / "model.onnx"
    if not path.exists():
        print(f"  exporting {model_name} to ONNX (first run only)...")
        _export(model_name, tokenizer, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = torch.get_num_threads()
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def encode_last_token(session: ort.InferenceSession, tokenizer, texts: list[str], dim: int, max_length: int,
                      batch_size: int, show_progress: bool = False) -> np.ndarray:
    """L2-normalized float32 vectors, shape (n, dim), using last-token pooling (the only pooling Harrier uses)."""
    vectors = np.zeros((len(texts), dim), np.float32)
    # Similar lengths together, so short chunks don't get padded up to the longest one in the batch.
    order = np.argsort([len(t) for t in texts])
    for start in tqdm(range(0, len(texts), batch_size), desc="Batches", disable=not show_progress):
        batch = order[start : start + batch_size]
        tokens = tokenizer([texts[i] for i in batch], padding=True, truncation=True, max_length=max_length,
                           return_tensors="np")
        mask = tokens["attention_mask"].astype(np.int64)
        hidden = session.run(None, {"input_ids": tokens["input_ids"].astype(np.int64), "attention_mask": mask})[0]
        last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)  # last real token, whichever side is padded
        pooled = hidden[np.arange(len(batch)), last]
        vectors[batch] = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
    return vectors
