import logging
import threading
import time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from supabase import Client

from app.deps import get_supabase
from app.pipelines.recommender import RecommenderPipeline
from app.vector_store.pgvector_store import PGVectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

# Short TTL cache: same user + top_k hit repeatedly from the SPA (React Strict Mode, multiple pages).
_CACHE_TTL_SEC = 90.0
_CACHE_MAX = 512
_reco_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_reco_lock = threading.Lock()


def _cache_get(key: str) -> dict[str, Any] | None:
    now = time.monotonic()
    with _reco_lock:
        hit = _reco_cache.get(key)
        if not hit:
            return None
        exp, value = hit
        if now > exp:
            del _reco_cache[key]
            return None
        return value


def _cache_set(key: str, value: dict[str, Any]) -> None:
    with _reco_lock:
        if len(_reco_cache) >= _CACHE_MAX and key not in _reco_cache:
            _reco_cache.pop(next(iter(_reco_cache)))
        _reco_cache[key] = (time.monotonic() + _CACHE_TTL_SEC, value)


def get_vector_store(client: Client = Depends(get_supabase)) -> PGVectorStore:
    return PGVectorStore(client)


def get_recommender(
    client: Client = Depends(get_supabase),
    store: PGVectorStore = Depends(get_vector_store),
) -> RecommenderPipeline:
    return RecommenderPipeline(client, store)


@router.get("/destinations")
def recommend_destinations(
    user_id: UUID = Query(..., description="Traveller / user UUID"),
    top_k: int = Query(5, ge=1, le=50),
    rec: RecommenderPipeline = Depends(get_recommender),
) -> dict[str, Any]:
    key = f"d:{user_id}:{top_k}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        results = rec.recommend_destinations(user_id, top_k=top_k)
    except Exception:
        logger.exception("recommend_destinations failed")
        results = []
    payload = {"user_id": str(user_id), "results": results}
    _cache_set(key, payload)
    return payload


@router.get("/trips")
def recommend_trips(
    user_id: UUID = Query(...),
    top_k: int = Query(5, ge=1, le=50),
    rec: RecommenderPipeline = Depends(get_recommender),
) -> dict[str, Any]:
    key = f"t:{user_id}:{top_k}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        results = rec.recommend_trips(user_id, top_k=top_k)
    except Exception:
        logger.exception("recommend_trips failed")
        results = []
    payload = {"user_id": str(user_id), "results": results}
    _cache_set(key, payload)
    return payload


@router.get("/routes")
def recommend_routes(
    user_id: UUID = Query(...),
    top_k: int = Query(5, ge=1, le=50),
    rec: RecommenderPipeline = Depends(get_recommender),
) -> dict[str, Any]:
    key = f"r:{user_id}:{top_k}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        results = rec.recommend_routes(user_id, top_k=top_k)
    except Exception:
        logger.exception("recommend_routes failed")
        results = []
    payload = {"user_id": str(user_id), "results": results}
    _cache_set(key, payload)
    return payload
