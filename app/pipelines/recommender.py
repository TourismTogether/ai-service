from __future__ import annotations

from typing import Any
from uuid import UUID

from supabase import Client

from app.model.embedding_model import encode_query
from app.vector_store.pgvector_store import PGVectorStore


def _join_parts(*parts: str | None) -> str:
    return " ".join(p.strip() for p in parts if p and str(p).strip())


def _rows(resp: Any) -> list[dict[str, Any]]:
    return list(resp.data or [])


class RecommenderPipeline:
    """Uses flat Supabase selects (no nested FK embeds) for PostgREST compatibility."""

    def __init__(self, db: Client, store: PGVectorStore):
        self._db = db
        self._store = store

    def _destinations_by_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        ids = [i for i in dict.fromkeys(ids) if i]
        if not ids:
            return {}
        out: dict[str, dict[str, Any]] = {}
        chunk = 50
        for i in range(0, len(ids), chunk):
            part = ids[i : i + chunk]
            dr = (
                self._db.table("destinations")
                .select(
                    "id, name, description, category, best_season, country"
                )
                .in_("id", part)
                .execute()
            )
            for d in _rows(dr):
                out[str(d["id"])] = d
        return out

    def _user_destination_assessment_profile(self, user_id: str) -> str:
        r = (
            self._db.table("assess_destination")
            .select("rating_star, comment, destination_id")
            .eq("traveller_id", user_id)
            .execute()
        )
        rows = _rows(r)
        if not rows:
            return ""
        dest_map = self._destinations_by_ids(
            [str(x["destination_id"]) for x in rows if x.get("destination_id")]
        )
        chunks: list[str] = []
        for row in rows:
            did = str(row.get("destination_id") or "")
            dest = dest_map.get(did, {})
            star = row.get("rating_star")
            comment = row.get("comment") or ""
            header = _join_parts(
                dest.get("name"),
                dest.get("country"),
                dest.get("category"),
                dest.get("best_season"),
            )
            body = _join_parts(dest.get("description"), comment)
            weight = f"(đánh giá {star}/5)" if star else ""
            chunks.append(_join_parts(header, weight, body))
        return _join_parts(*chunks) if chunks else ""

    def _user_trip_context(
        self, user_id: str
    ) -> tuple[str, list[str], list[str]]:
        r = (
            self._db.table("join_trip")
            .select("trip_id")
            .eq("user_id", user_id)
            .execute()
        )
        trip_ids = [
            str(x["trip_id"]) for x in _rows(r) if x.get("trip_id")
        ]
        if not trip_ids:
            return "", [], []

        trips: list[dict[str, Any]] = []
        chunk = 50
        for i in range(0, len(trip_ids), chunk):
            part = trip_ids[i : i + chunk]
            tr = (
                self._db.table("trips")
                .select("id, title, description, destination_id")
                .in_("id", part)
                .execute()
            )
            trips.extend(_rows(tr))

        dest_ids: list[str] = []
        for t in trips:
            if t.get("destination_id"):
                dest_ids.append(str(t["destination_id"]))
        dest_map = self._destinations_by_ids(dest_ids)

        chunks: list[str] = []
        dest_from_trips: list[str] = []
        for trip in trips:
            did = trip.get("destination_id")
            if did:
                ds = str(did)
                dest_from_trips.append(ds)
            dest = dest_map.get(str(did), {}) if did else {}
            chunks.append(
                _join_parts(
                    trip.get("title"),
                    trip.get("description"),
                    dest.get("name"),
                    dest.get("description"),
                    dest.get("category"),
                    dest.get("best_season"),
                )
            )
        profile = _join_parts(*chunks)
        return profile, trip_ids, list(dict.fromkeys(dest_from_trips))

    def _user_route_profile(self, user_id: str) -> tuple[str, list[str]]:
        r = (
            self._db.table("join_trip")
            .select("trip_id")
            .eq("user_id", user_id)
            .execute()
        )
        trip_ids = [
            str(row["trip_id"]) for row in _rows(r) if row.get("trip_id")
        ]
        if not trip_ids:
            return "", []
        rq = (
            self._db.table("routes")
            .select("id, title, description")
            .in_("trip_id", trip_ids)
            .execute()
        )
        route_ids: list[str] = []
        chunks: list[str] = []
        for row in _rows(rq):
            if row.get("id"):
                route_ids.append(str(row["id"]))
            chunks.append(_join_parts(row.get("title"), row.get("description")))
        return _join_parts(*chunks), route_ids

    def recommend_destinations(self, user_id: UUID | str, top_k: int = 5):
        uid = str(user_id)
        assess_profile = self._user_destination_assessment_profile(uid)
        trip_profile, _, dest_from_trips = self._user_trip_context(uid)
        profile = _join_parts(assess_profile, trip_profile)
        if not profile.strip():
            return []
        emb = encode_query(profile)
        r_assess = (
            self._db.table("assess_destination")
            .select("destination_id")
            .eq("traveller_id", uid)
            .execute()
        )
        exclude: list[str] = [
            str(x["destination_id"])
            for x in _rows(r_assess)
            if x.get("destination_id")
        ]
        exclude.extend(dest_from_trips)
        exclude = list(dict.fromkeys(exclude))
        try:
            return self._store.search_destinations(emb, top_k, exclude_ids=exclude)
        except Exception:
            return []

    def recommend_trips(self, user_id: UUID | str, top_k: int = 5):
        uid = str(user_id)
        assess_profile = self._user_destination_assessment_profile(uid)
        trip_profile, joined_trip_ids, _ = self._user_trip_context(uid)
        profile = _join_parts(assess_profile, trip_profile)
        if not profile.strip():
            return []
        emb = encode_query(profile)
        exclude = list(dict.fromkeys(joined_trip_ids))
        try:
            return self._store.search_trips(emb, top_k, exclude_ids=exclude)
        except Exception:
            return []

    def recommend_routes(self, user_id: UUID | str, top_k: int = 5):
        uid = str(user_id)
        route_profile, route_ids = self._user_route_profile(uid)
        trip_profile, _, _ = self._user_trip_context(uid)
        profile = _join_parts(route_profile, trip_profile)
        if not profile.strip():
            return []
        emb = encode_query(profile)
        exclude = list(dict.fromkeys(route_ids))
        try:
            return self._store.search_routes(emb, top_k, exclude_ids=exclude)
        except Exception:
            return []
