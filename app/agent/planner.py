from app.llm.client import decide_tool_use


def plan_next_step(message: str) -> dict:
    return decide_tool_use(message=message)