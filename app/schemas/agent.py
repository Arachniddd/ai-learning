from pydantic import BaseModel, Field

class AgentRequest(BaseModel):
    message: str = Field(description="用户发送给 Agent 的消息")
    max_steps: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Agent 最大执行步骤数",
    )
