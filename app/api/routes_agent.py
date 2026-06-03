from fastapi import APIRouter

from app.agent.executor import run_agent
from app.schemas.agent import AgentRequest

router = APIRouter()


@router.post("")
def agent(request: AgentRequest):
    return run_agent(
        message=request.message,
        max_steps=request.max_steps,
    )
