from supabase import Client

class PGVectorStore:
    def __init__(self, client: Client):
        self.client = client

    def search_destinations(
        self,
        query_text: str,       # Đã thêm tham số text
        embedding: list[float],
        top_k: int = 5,
        exclude_ids: list[str] | None = None,
    ):
        return (
            self.client.rpc(
                "search_destinations",
                {
                    "query_text": query_text.strip(),  # Truyền text xuống SQL
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
        query_text: str,       # Đã thêm tham số text
        embedding: list[float],
        top_k: int = 5,
        exclude_ids: list[str] | None = None,
    ):
        return (
            self.client.rpc(
                "search_trips",
                {
                    "query_text": query_text.strip(),  # Truyền text xuống SQL
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
        query_text: str,       # Đã thêm tham số text
        embedding: list[float],
        top_k: int = 5,
        exclude_ids: list[str] | None = None,
    ):
        return (
            self.client.rpc(
                "search_routes",
                {
                    "query_text": query_text.strip(),  # Truyền text xuống SQL
                    "query_embedding": embedding,
                    "match_count": top_k,
                    "exclude_ids": exclude_ids or [],
                },
            )
            .execute()
            .data
            or []
        )