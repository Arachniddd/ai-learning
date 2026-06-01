from fastapi import APIRouter

from app.schemas.agent import AgentRequest
from app.agent.executor import run_agent

router = APIRouter()


@router.post("")
def agent(request: AgentRequest):
    return run_agent(request.message)