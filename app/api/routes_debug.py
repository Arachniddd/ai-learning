from fastapi import APIRouter

from app.agent.schemas import InspectRequest
from app.agent.tools import inspect_retrieval

router = APIRouter()


@router.post("/retrieval")
def inspect_retrieval_flow(request: InspectRequest):
    return inspect_retrieval(
        question=request.question,
        retrieve_top_k=request.retrieve_top_k,
        rerank_top_k=request.rerank_top_k,
    )
