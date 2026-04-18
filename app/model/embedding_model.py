import os
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

# MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
# MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_ENCODE_BATCH = 32


def _device() -> str:
    d = (os.environ.get("EMBEDDING_DEVICE") or "").strip().lower()
    if d in ("cpu", "cuda", "mps"):
        return d
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    low_cpu_mem_usage=False avoids loading weights as empty "meta" tensors, which
    breaks .to(device) on some torch + sentence-transformers versions (Windows/CPU).
    """
    return SentenceTransformer(
        MODEL_NAME,
        device=_device(),
        model_kwargs={"low_cpu_mem_usage": False},
    )


def encode_query(text: str) -> list[float]:
    text = f"query: {text.strip()}"
    return get_model().encode([text], normalize_embeddings=True)[0].tolist()


def encode_passage(text: str) -> list[float]:
    text = f"passage: {text.strip()}"
    return get_model().encode([text], normalize_embeddings=True)[0].tolist()


def encode_passages_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    prefixed = [f"passage: {t.strip()}" for t in texts]
    vectors = get_model().encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=_ENCODE_BATCH,
        show_progress_bar=len(texts) > 16,
    )
    return vectors.tolist()