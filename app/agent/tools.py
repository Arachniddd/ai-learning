from app.llm.client import (
    decide_tool_use as llm_decide_tool_use,
    explain_with_chunks,
    final_answer_with_tool_result as llm_final_answer_with_tool_result,
    generate_quiz_from_chunks,
    rerank_chunks,
    rewrite_query,
    summarize_note,
)
from app.rag.qa import retrieve_reranked_chunks
from app.models.chunk import Chunk
from app.rag.retrieve import retrieve
from app.rag.vector_store import list_all_chunks


def search_knowledge_base(query: str, top_k: int = 3) -> dict:
    result = retrieve_reranked_chunks(question=query, top_k=top_k)

    return {
        "query": query,
        "rewritten_query": result["rewritten_query"],
        "chunks": result["reranked_chunks"],
        "retrieval_debug": {
            "original_query": result["original_query"],
            "rewritten_query": result["rewritten_query"],
            "retrieved_chunks": result["retrieved_chunks"],
            "reranked_chunks": result["reranked_chunks"],
        },
    }


def inspect_retrieval(
    question: str,
    retrieve_top_k: int = 10,
    rerank_top_k: int = 3,
) -> dict:
    rewritten_query = rewrite_query(question)
    retrieved_chunks = retrieve(
        query=rewritten_query,
        top_k=retrieve_top_k,
    )
    reranked_chunks = rerank_chunks(
        question=rewritten_query,
        chunks=retrieved_chunks,
        top_k=rerank_top_k,
    )

    return {
        "original_query": question,
        "rewritten_query": rewritten_query,
        "retrieved_chunks": retrieved_chunks,
        "reranked_chunks": reranked_chunks,
    }


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


def generate_quiz(topic: str, num_questions: int = 5) -> dict:
    retrieval_result = retrieve_reranked_chunks(question=topic, top_k=3)
    chunks = retrieval_result["reranked_chunks"]

    if not chunks:
        return {
            "error": "没有检索到足够相关的资料，无法出题。",
            "topic": topic,
        }

    return generate_quiz_from_chunks(
        topic=topic,
        chunks=chunks,
        num_questions=num_questions,
    )


def explain_concept(concept: str, detail_level: str = "medium") -> dict:
    retrieval_result = retrieve_reranked_chunks(question=concept, top_k=3)
    chunks = retrieval_result["reranked_chunks"]

    result = explain_with_chunks(
        concept=concept,
        chunks=chunks,
        detail_level=detail_level,
    )

    result["retrieval_debug"] = {
        "original_query": retrieval_result["original_query"],
        "rewritten_query": retrieval_result["rewritten_query"],
        "retrieved_count": len(retrieval_result["retrieved_chunks"]),
        "reranked_count": len(chunks),
        "retrieved_chunks": retrieval_result["retrieved_chunks"],
        "reranked_chunks": chunks,
    }

    return result
