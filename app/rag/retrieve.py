from app.llm.client import rewrite_query
from app.rag.vector_store import search_vector_store

def retrieve_with_rewrite(question : str, top_k : int = 10):
    rewrited_question = rewrite_query(question=question)

    retrieved_chunks = search_vector_store(
        query=rewrited_question, 
        top_k=top_k
    )

    return {
        "original question": question,
        "rewrited question": rewrited_question,
        "chunks": retrieved_chunks
    }

