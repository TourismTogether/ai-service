from app.model.embedding_model import encode_query
from app.vector_store.pgvector_store import PGVectorStore


class SemanticSearchPipeline:
    def __init__(self, store: PGVectorStore):
        self.store = store

    def search_destinations(self, query: str, top_k: int = 5):
        return self.store.search_destinations(encode_query(query), top_k)

    def search_trips(self, query: str, top_k: int = 5):
        return self.store.search_trips(encode_query(query), top_k)

    def search_routes(self, query: str, top_k: int = 5):
        return self.store.search_routes(encode_query(query), top_k)
