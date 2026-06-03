from app.rag.vector_store import search_vector_store, list_all_chunks
from app.llm.client import rerank_chunks, summarize_note


def search_knowledge_base(query: str, top_k: int = 3):
    candidates = search_vector_store(query=query, top_k=top_k * 3)
    return rerank_chunks(question=query, chunks=candidates, top_k=top_k)


def list_chunks(limit: int = 20):
    return list_all_chunks(limit=limit)


def summarize_text(text: str):
    return summarize_note(text)
