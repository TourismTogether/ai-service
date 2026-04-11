from supabase import Client


class PGVectorStore:
    def __init__(self, client: Client):
        self.client = client

    def search_destinations(
        self,
        embedding: list[float],
        top_k: int = 5,
        exclude_ids: list[str] | None = None,
    ):
        return (
            self.client.rpc(
                "search_destinations",
                {
                    "query_embedding": embedding,
                    "match_count": top_k,
                    "exclude_ids": exclude_ids or [],
                },
            )
            .execute()
            .data
            or []
        )

    def search_trips(
        self,
        embedding: list[float],
        top_k: int = 5,
        exclude_ids: list[str] | None = None,
    ):
        return (
            self.client.rpc(
                "search_trips",
                {
                    "query_embedding": embedding,
                    "match_count": top_k,
                    "exclude_ids": exclude_ids or [],
                },
            )
            .execute()
            .data
            or []
        )

    def search_routes(
        self,
        embedding: list[float],
        top_k: int = 5,
        exclude_ids: list[str] | None = None,
    ):
        return (
            self.client.rpc(
                "search_routes",
                {
                    "query_embedding": embedding,
                    "match_count": top_k,
                    "exclude_ids": exclude_ids or [],
                },
            )
            .execute()
            .data
            or []
        )
