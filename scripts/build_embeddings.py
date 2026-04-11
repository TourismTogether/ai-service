#!/usr/bin/env python3
"""Build passage embeddings and write to Supabase (destinations, trips, routes).

Run from repo root of ai-service:
  python scripts/build_embeddings.py

If row-level security blocks updates, set SUPABASE_KEY to the service_role key
(never expose it in a browser).

Requires migration 001_pgvector_embeddings_and_search.sql applied on the database.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from supabase import create_client

from app.config import get_settings
from app.model.embedding_model import encode_passages_batch


def _client():
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_key)


def destination_text(row: dict) -> str:
    parts = [
        row.get("name"),
        row.get("description"),
        row.get("category"),
        row.get("best_season"),
        row.get("country"),
    ]
    t = " ".join(str(p).strip() for p in parts if p is not None and str(p).strip())
    return t if t.strip() else "dia diem du lich"


def trip_text(row: dict) -> str:
    t = " ".join(
        str(p).strip()
        for p in (row.get("title"), row.get("description"))
        if p is not None and str(p).strip()
    )
    return t if t.strip() else "chuyen di"


def route_text(row: dict) -> str:
    t = " ".join(
        str(p).strip()
        for p in (row.get("title"), row.get("description"))
        if p is not None and str(p).strip()
    )
    return t if t.strip() else "route"


def fetch_all(client, table: str, select: str = "*") -> list[dict]:
    out: list[dict] = []
    page_size = 1000
    offset = 0
    while True:
        res = (
            client.table(table)
            .select(select)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = res.data or []
        out.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return out


def upsert_embeddings(client, table: str, ids: list[str], vectors: list[list[float]]):
    for rid, vec in zip(ids, vectors):
        literal = "[" + ",".join(str(float(x)) for x in vec) + "]"
        client.table(table).update({"embedding": literal}).eq("id", rid).execute()


def main() -> None:
    client = _client()
    print("Destinations...")
    dests = fetch_all(client, "destinations")
    if dests:
        texts = [destination_text(r) for r in dests]
        ids = [str(r["id"]) for r in dests]
        vecs = encode_passages_batch(texts)
        upsert_embeddings(client, "destinations", ids, vecs)
        print(f"  updated {len(ids)} destinations")
    else:
        print("  no rows")

    print("Trips...")
    trips = fetch_all(client, "trips")
    if trips:
        texts = [trip_text(r) for r in trips]
        ids = [str(r["id"]) for r in trips]
        vecs = encode_passages_batch(texts)
        upsert_embeddings(client, "trips", ids, vecs)
        print(f"  updated {len(ids)} trips")
    else:
        print("  no rows")

    print("Routes...")
    routes = fetch_all(client, "routes")
    if routes:
        texts = [route_text(r) for r in routes]
        ids = [str(r["id"]) for r in routes]
        vecs = encode_passages_batch(texts)
        upsert_embeddings(client, "routes", ids, vecs)
        print(f"  updated {len(ids)} routes")
    else:
        print("  no rows")

    print("Done.")


if __name__ == "__main__":
    main()
