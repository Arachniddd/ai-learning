from dataclasses import dataclass
from typing import Callable, Type

from pydantic import BaseModel

from app.agent.schemas import (
    SearchKnowledgeBaseArgs,
    ListChunksArgs,
    SummarizeTextArgs,
)
from app.agent.tools import (
    search_knowledge_base,
    list_chunks,
    summarize_text,
)


@dataclass
class ToolSpec:
    name : str
    description : str
    args_schema : Type[BaseModel]
    func : Callable

TOOL_REGISTRY : dict[str, ToolSpec] = {
    "search_knowledge_base": ToolSpec(
        name="search_knowledge_base",
        description="当用户需要查询知识库、课程笔记、上传文档内容时使用。",
        args_schema=SearchKnowledgeBaseArgs,
        func=search_knowledge_base,
    ),
    "list_chunks": ToolSpec(
        name="list_chunks",
        description="当用户想查看当前知识库中已有 chunk 时使用。",
        args_schema=ListChunksArgs,
        func=list_chunks,
    ),
    "summarize_text": ToolSpec(
        name="summarize_text",
        description="当用户明确要求总结一段文本时使用。",
        args_schema=SummarizeTextArgs,
        func=summarize_text,
    ),
}


def get_tool_spec(tool_name: str) -> ToolSpec | None:
    return TOOL_REGISTRY.get(tool_name)

def list_tool_specs() -> list[dict]:
    specs = []

    for tool in TOOL_REGISTRY.values():
        specs.append({
            "name": tool.name,
            "description": tool.description,
            "args_schema": tool.args_schema.model_json_schema(),
        })

    return specs
