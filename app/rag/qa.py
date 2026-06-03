from app.llm.client import answer_with_chunks, rerank_chunks, rewrite_query
from app.rag.retrieve import retrieve
from app.rag.types import RetrieveChunk


def retrieve_reranked_chunks(question: str, top_k: int = 3) -> dict:
    rewritten_query = rewrite_query(question)
    retrieved_chunks = retrieve(
        query=rewritten_query,
        top_k=max(top_k * 3, 10),
    )

    if not retrieved_chunks:
        return {
            "original_query": question,
            "rewritten_query": rewritten_query,
            "retrieved_chunks": [],
            "reranked_chunks": [],
        }

    reranked_chunks = rerank_chunks(
        question=rewritten_query,
        chunks=retrieved_chunks,
        top_k=top_k,
    )

    return {
        "original_query": question,
        "rewritten_query": rewritten_query,
        "retrieved_chunks": retrieved_chunks,
        "reranked_chunks": reranked_chunks,
    }


def answer_question_with_rag(question: str, top_k: int = 3) -> dict:
    retrieval_result = retrieve_reranked_chunks(question=question, top_k=top_k)
    final_chunks: list[RetrieveChunk] = retrieval_result["reranked_chunks"]

    answer_result = answer_with_chunks(
        question=question,
        chunks=final_chunks,
    )

    return {
        "answer": answer_result["answer"],
        "used_chunks": answer_result["used_chunks"],
        "retrieval_debug": {
            "original_query": retrieval_result["original_query"],
            "rewritten_query": retrieval_result["rewritten_query"],
            "retrieved_count": len(retrieval_result["retrieved_chunks"]),
            "reranked_count": len(retrieval_result["reranked_chunks"]),
            "retrieved_chunks": retrieval_result["retrieved_chunks"],
            "reranked_chunks": retrieval_result["reranked_chunks"],
        },
    }
