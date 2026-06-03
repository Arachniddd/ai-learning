from app.llm.client import (
    decide_tool_use as llm_decide_tool_use,
    final_answer_with_tool_result as llm_final_answer_with_tool_result,
    rerank_chunks,
    rewrite_query,
    summarize_note,
)
from app.rag.retrieve import retrieve
from app.rag.types import Chunk, RetrieveChunk
from app.rag.vector_store import list_all_chunks


def search_knowledge_base(query: str, top_k: int = 3) -> list[RetrieveChunk]:
    rewritten_query = rewrite_query(query)
    candidates = retrieve(query=rewritten_query, top_k=top_k * 3)

    if not candidates:
        return []

    return rerank_chunks(
        question=rewritten_query,
        chunks=candidates,
        top_k=top_k,
    )


def decide_tool_use(message: str) -> dict:
    return llm_decide_tool_use(message=message)


def final_answer_with_tool_result(
    message: str,
    tool_result: list[Chunk],
) -> dict:
    return llm_final_answer_with_tool_result(
        message=message,
        tool_result=tool_result,
    )


def list_chunks(limit: int = 20):
    return list_all_chunks(limit=limit)


def summarize_text(text: str):
    return summarize_note(text)
