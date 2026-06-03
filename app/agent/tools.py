from app.llm.client import (
    decide_tool_use as llm_decide_tool_use,
    final_answer_with_tool_result as llm_final_answer_with_tool_result,
    summarize_note,
)
from app.rag.qa import retrieve_reranked_chunks
from app.rag.types import Chunk, RetrieveChunk
from app.rag.vector_store import list_all_chunks


def search_knowledge_base(query: str, top_k: int = 3) -> list[RetrieveChunk]:
    result = retrieve_reranked_chunks(question=query, top_k=top_k)
    return result["reranked_chunks"]


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
