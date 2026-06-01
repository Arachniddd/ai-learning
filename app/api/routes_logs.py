from fastapi import APIRouter

from app.observability.logger import read_agent_logs

router = APIRouter()


@router.get("/agent")
def get_agent_logs(limit: int = 20):
    logs = read_agent_logs(limit=limit)
    return {
        "count": len(logs),
        "logs": logs,
    }