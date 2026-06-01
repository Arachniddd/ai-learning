from fastapi import APIRouter

from app.schemas.chat import AskRequest
from app.rag.vector_store import search_vector_store
from app.llm.client import answer_with_context

router = APIRouter()

@router.post("/ask")
def ask(request: AskRequest):
    contexts = search_vector_store(request.question, request.top_k)

    if len(contexts) == 0:
        return {
            "message": "Nothing can be searched. Please try another question or add more documents.",
            "contexts": [],
        }

    result = answer_with_context(request.question, contexts)

    return {
        "answer": result["answer"],
        "used_chunks": result.get("used_chunks", []),
        "contexts": contexts,
    }
