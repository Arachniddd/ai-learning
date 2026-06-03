from fastapi import APIRouter

from app.schemas.chat import AskRequest
from app.rag.qa import answer_question_with_rag

router = APIRouter()

@router.post("/ask")
def ask(request: AskRequest):
    return answer_question_with_rag(
        question=request.question,
        top_k=request.top_k,
    )
