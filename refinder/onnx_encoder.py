"""Runs an embedding model through ONNX Runtime on CPU, exporting it to .onnx on first use."""

import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm
from transformers import AutoModel

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "onnx"
POOLING = ("lasttoken", "cls", "mean")  # Sentence Transformers pooling modes this module reproduces


class _LastHiddenState(torch.nn.Module):
    """Makes the model return a plain tensor, which is what the ONNX exporter needs."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Decoder-style models (Harrier, Qwen3) would otherwise build a key/value cache; encoders don't take the flag.
        self.extra = {"use_cache": False} if getattr(model.config, "use_cache", None) is not None else {}

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **self.extra).last_hidden_state


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


def load_session(model_name: str, tokenizer, int8: bool = False) -> ort.InferenceSession:
    """int8: weights quantized to 8-bit integers. Not used by the app (no faster on Apple silicon, slightly less
    accurate); evaluation.index_bench --int8 measures it on other CPUs."""
    path = CACHE_DIR / model_name.replace("/", "--") / "model.onnx"
    if not path.exists():
        print(f"  exporting {model_name} to ONNX (first run only)...")
        _export(model_name, tokenizer, path)
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized = path.with_name("model_int8.onnx")
        if not quantized.exists():
            quantize_dynamic(str(path), str(quantized), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
        path = quantized
    options = ort.SessionOptions()
    options.intra_op_num_threads = torch.get_num_threads()
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def encode(session: ort.InferenceSession, tokenizer, texts: list[str], dim: int, max_length: int, batch_size: int,
           pooling: str, show_progress: bool = False) -> np.ndarray:
    """L2-normalized float32 vectors, shape (n, dim). pooling is the model's own Sentence Transformers mode:
    lasttoken (Harrier, Qwen3), cls (bge) or mean (MiniLM)."""
    vectors = np.zeros((len(texts), dim), np.float32)
    # Similar lengths together, so short chunks don't get padded up to the longest one in the batch.
    order = np.argsort([len(t) for t in texts])
    for start in tqdm(range(0, len(texts), batch_size), desc="Batches", disable=not show_progress):
        batch = order[start : start + batch_size]
        tokens = tokenizer([texts[i] for i in batch], padding=True, truncation=True, max_length=max_length,
                           return_tensors="np")
        mask = tokens["attention_mask"].astype(np.int64)
        hidden = session.run(None, {"input_ids": tokens["input_ids"].astype(np.int64), "attention_mask": mask})[0]
        if pooling == "lasttoken":
            last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)  # last real token, whichever side is padded
            pooled = hidden[np.arange(len(batch)), last]
        elif pooling == "cls":
            pooled = hidden[:, 0]
        else:  # mean over real tokens
            pooled = (hidden * mask[:, :, None]).sum(1) / mask.sum(1, keepdims=True)
        vectors[batch] = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
    return vectors
