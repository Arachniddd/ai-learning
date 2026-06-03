from typing import Literal, Optional, Any

from pydantic import BaseModel, ConfigDict, Field


class AgentAction(BaseModel):
    action_type: Literal["tool_call", "final_answer"] = Field(
        description="Agent 下一步动作类型"
    )

    tool_name: Optional[str] = Field(
        default=None,
        description="需要调用的工具名，仅 tool_call 时使用"
    )

    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="工具调用参数"
    )

    answer: Optional[str] = Field(
        default=None,
        description="最终回答，仅 final_answer 时使用"
    )

    thought: Optional[str] = Field(
        default=None,
        description="简短说明为什么选择这个动作"
    )


class ToolResult(BaseModel):
    success: bool
    tool_name : str
    result : Any = None
    error : Optional[str] = None


class AgentStep(BaseModel):
    step_index : int
    action : AgentAction
    observation : Optional[ToolResult] = None
    retrieval_debug : Optional[dict[str, Any]] = None


class AgentToolResult(BaseModel):
    answer : str
    steps : list[AgentStep]
    mode : str
    error : Optional[str] = None


class SearchKnowledgeBaseArgs(BaseModel):
    query: str = Field(description="用于检索知识库的问题")
    top_k: int = Field(default=3, ge=1, le=10, description="返回的 chunk 数量")


class InspectRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(
        alias="query",
        description="用于调试检索流程的原始问题",
    )
    retrieve_top_k: int = Field(
        default=10,
        ge=1,
        le=30,
        description="直接向量检索返回的 chunk 数量",
    )
    rerank_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="rerank 后返回的 chunk 数量",
    )


class ListChunksArgs(BaseModel):
    limit: int = Field(default=20, ge=1, le=100, description="返回的 chunk 数量")


class SummarizeTextArgs(BaseModel):
    text: str = Field(description="需要总结的文本")
