import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from supabase import Client

from app.deps import get_supabase
from app.pipelines.recommender import RecommenderPipeline
from app.vector_store.pgvector_store import PGVectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


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
    try:
        results = rec.recommend_destinations(user_id, top_k=top_k)
    except Exception:
        logger.exception("recommend_destinations failed")
        results = []
    return {"user_id": str(user_id), "results": results}


@router.get("/trips")
def recommend_trips(
    user_id: UUID = Query(...),
    top_k: int = Query(5, ge=1, le=50),
    rec: RecommenderPipeline = Depends(get_recommender),
) -> dict[str, Any]:
    try:
        results = rec.recommend_trips(user_id, top_k=top_k)
    except Exception:
        logger.exception("recommend_trips failed")
        results = []
    return {"user_id": str(user_id), "results": results}


@router.get("/routes")
def recommend_routes(
    user_id: UUID = Query(...),
    top_k: int = Query(5, ge=1, le=50),
    rec: RecommenderPipeline = Depends(get_recommender),
) -> dict[str, Any]:
    try:
        results = rec.recommend_routes(user_id, top_k=top_k)
    except Exception:
        logger.exception("recommend_routes failed")
        results = []
    return {"user_id": str(user_id), "results": results}
