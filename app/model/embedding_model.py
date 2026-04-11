from functools import lru_cache

from sentence_transformers import SentenceTransformer

# MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
_ENCODE_BATCH = 32


@lru_cache
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


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