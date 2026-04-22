#!/usr/bin/env python3
"""Measure semantic search latency: embedding encode vs Supabase pgvector RPC vs brute-force.

Use this to back CV claims (p50/p95, speedup vs naive in-memory scan).

Run from ai-service root (with .env: SUPABASE_URL, SUPABASE_KEY):

  python scripts/benchmark_retrieval.py --iterations 50 --top-k 16

Optional: time the full HTTP stack (start `uvicorn app.main:app` elsewhere):

  python scripts/benchmark_retrieval.py --http http://127.0.0.1:8000 --iterations 30

What gets measured
-----------------
1) encode_ms: SentenceTransformer query embedding (same for all paths).
2) rpc_ms: Supabase RPC ``search_destinations`` only (indexed pgvector in DB).
3) naive_cpu_ms: top-k cosine similarity over ALL destination vectors held in RAM
   (simulates an app that loaded embeddings once then scans in-process — fair vs RPC   per query after a cold load).
4) naive_fetch_ms: one-time download of all ``id, embedding`` rows (optional baseline   for "fetch everything every request" anti-patterns).

Speedup for CV: compare median(encode + rpc) vs median(encode + naive_cpu) after
``naive_fetch_ms`` is amortized or cited separately.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from supabase import create_client

from app.config import get_settings
from app.model.embedding_model import encode_query
from app.vector_store.pgvector_store import PGVectorStore


SAMPLE_QUERIES_VI = [
    "bien dao he mat troi",
    "di san the gioi mua thu",
    "trek nui deo cao",
    "am thuc duong pho dem",
    "resort gia dinh gan bien",
    "lich su chua co",
    "city break cuoi tuan",
]


def _parse_embedding(raw) -> np.ndarray:
    if raw is None:
        raise ValueError("null embedding")
    if isinstance(raw, str):
        raw = json.loads(raw)
    return np.asarray(raw, dtype=np.float32)


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def _fetch_destination_embeddings(client):
    out: list[tuple[str, np.ndarray]] = []
    page_size = 1000
    offset = 0
    while True:
        res = (
            client.table("destinations")
            .select("id, embedding")
            .not_.is_("embedding", "null")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = res.data or []
        for row in batch:
            e = row.get("embedding")
            if e is None:
                continue
            out.append((str(row["id"]), _parse_embedding(e)))
        if len(batch) < page_size:
            break
        offset += page_size
    return out


def _topk_cosine_cpu(matrix: np.ndarray, q: np.ndarray, top_k: int) -> None:
    """matrix (n, d) row-normalized, q (d,) normalized -> top-k by dot product."""
    sims = matrix @ q
    k = min(top_k, sims.shape[0])
    # Partial sort is enough for top-k
    _ = np.argpartition(-sims, kth=k - 1)[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pgvector RPC vs naive scan")
    parser.add_argument("--iterations", type=int, default=40, help="Repeated queries")
    parser.add_argument("--top-k", type=int, default=16, dest="top_k")
    parser.add_argument(
        "--http",
        type=str,
        default="",
        help="If set, also GET {base}/search/destinations?query=...&top_k=... (FastAPI)",
    )
    args = parser.parse_args()

    settings = get_settings()
    client = create_client(settings.supabase_url, settings.supabase_key)
    store = PGVectorStore(client)

    print("Loading all destination embeddings for naive baseline...")
    t0 = time.perf_counter()
    pairs = _fetch_destination_embeddings(client)
    fetch_s = time.perf_counter() - t0
    n = len(pairs)
    if n == 0:
        print("No rows with non-null embedding in destinations. Run build_embeddings.py first.")
        sys.exit(1)
    matrix = np.stack([p[1] for p in pairs], axis=0)
    print(f"  rows={n}, fetch_wall_ms={fetch_s * 1000:.1f}, matrix_shape={matrix.shape}")

    print("Warming up model + RPC...")
    _ = encode_query("warmup query")
    q0 = encode_query("warmup query")
    store.search_destinations("warmup query", q0, top_k=args.top_k)
    _topk_cosine_cpu(matrix, np.asarray(q0, dtype=np.float32), args.top_k)

    encode_ms: list[float] = []
    rpc_ms: list[float] = []
    naive_ms: list[float] = []
    http_ms: list[float] = []

    import urllib.request

    for i in range(args.iterations):
        text = SAMPLE_QUERIES_VI[i % len(SAMPLE_QUERIES_VI)]

        t1 = time.perf_counter()
        q = encode_query(text)
        t2 = time.perf_counter()
        _ = store.search_destinations(text, q, top_k=args.top_k)
        t3 = time.perf_counter()

        qv = np.asarray(q, dtype=np.float32)
        t4 = time.perf_counter()
        _topk_cosine_cpu(matrix, qv, args.top_k)
        t5 = time.perf_counter()

        encode_ms.append((t2 - t1) * 1000)
        rpc_ms.append((t3 - t2) * 1000)
        naive_ms.append((t5 - t4) * 1000)

        if args.http:
            base = args.http.rstrip("/")
            url = f"{base}/search/destinations?query={quote(text)}&top_k={args.top_k}"
            t6 = time.perf_counter()
            with urllib.request.urlopen(url, timeout=120) as resp:
                resp.read()
            t7 = time.perf_counter()
            http_ms.append((t7 - t6) * 1000)

    def summarize(name: str, xs: list[float]) -> None:
        s = sorted(xs)
        print(
            f"  {name}: median={statistics.median(s):.2f}ms "
            f"p95={_percentile(s, 95):.2f}ms min={s[0]:.2f}ms max={s[-1]:.2f}ms"
        )

    print(f"\nResults (iterations={args.iterations}, top_k={args.top_k}, n_vectors={n})")
    summarize("encode_query", encode_ms)
    summarize("rpc search_destinations", rpc_ms)
    summarize("naive_cpu top-k (matrix in RAM)", naive_ms)
    if args.http:
        summarize("http GET /search/destinations", http_ms)

    med_enc = statistics.median(sorted(encode_ms))
    med_rpc = statistics.median(sorted(rpc_ms))
    med_naive = statistics.median(sorted(naive_ms))
    total_rpc = med_enc + med_rpc
    total_naive_steady = med_enc + med_naive
    if med_rpc > 0 and med_naive > 0:
        print("\n--- For resume / CV (steady state: embeddings already in app memory) ---")
        print(
            f"  median(encode+rpc)={total_rpc:.1f}ms vs median(encode+naive_cpu)={total_naive_steady:.1f}ms"
        )
        if total_rpc < total_naive_steady:
            print(
                f"  => pgvector RPC path ~{total_naive_steady / total_rpc:.1f}× faster than in-RAM brute force "
                f"(per query after embeddings loaded)."
            )
        else:
            print(
                "  => In-RAM brute force was not slower than RPC here (small N or RPC latency dominates). "
                "Cite instead: one-shot fetch of all vectors + network vs single RPC; or benchmark larger N / "
                "cold-cache naive that re-fetches rows each request."
            )
    print(
        f"\n  One-time fetch of all embeddings: {fetch_s*1000:.1f}ms "
        f"(add to each naive request if you re-fetch every time — avoids that with RPC.)"
    )
    print("\nNote: Enable IVFFlat/HNSW in Supabase (see migrations) for larger tables; "
          "seq scan on ORDER BY <=> still wins vs shipping all vectors to the client.")


if __name__ == "__main__":
    main()
