from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from supabase import Client

from app.deps import get_supabase
from app.pipelines.semantic_search import SemanticSearchPipeline
from app.vector_store.pgvector_store import PGVectorStore

router = APIRouter()


def get_vector_store(client: Client = Depends(get_supabase)) -> PGVectorStore:
    return PGVectorStore(client)


def get_semantic_search(
    store: PGVectorStore = Depends(get_vector_store),
) -> SemanticSearchPipeline:
    return SemanticSearchPipeline(store)


@router.get("/destinations")
def search_destinations(
    query: str = Query(..., min_length=1, description="Natural language search"),
    top_k: int = Query(5, ge=1, le=50),
    search: SemanticSearchPipeline = Depends(get_semantic_search),
) -> dict[str, Any]:
    try:
        results = search.search_destinations(query.strip(), top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search failed: {e!s}") from e
    return {"query": query, "results": results}


@router.get("/trips")
def search_trips(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    search: SemanticSearchPipeline = Depends(get_semantic_search),
) -> dict[str, Any]:
    try:
        results = search.search_trips(query.strip(), top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search failed: {e!s}") from e
    return {"query": query, "results": results}


@router.get("/routes")
def search_routes(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    search: SemanticSearchPipeline = Depends(get_semantic_search),
) -> dict[str, Any]:
    try:
        results = search.search_routes(query.strip(), top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Search failed: {e!s}") from e
    return {"query": query, "results": results}
