from app.rag.types import RetrieveChunk
from app.rag.vector_store import search_vector_store


def retrieve(query: str, top_k: int = 3) -> list[RetrieveChunk]:
    return search_vector_store(query=query, top_k=top_k)
