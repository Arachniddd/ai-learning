from typing import Any

from pydantic import ValidationError

from app.agent.registry import get_tool_spec
from app.agent.schemas import ToolResult


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> ToolResult:
    tool_spec = get_tool_spec(tool_name)

    if tool_spec is None:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error=f"Unknown tool: {tool_name}",
        )

    try:
        validated_args = tool_spec.args_schema.model_validate(arguments)
        result = tool_spec.func(**validated_args.model_dump())

        return ToolResult(
            success=True,
            tool_name=tool_name,
            result=result,
        )
    except ValidationError as e:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error=f"Tool arguments validation failed: {str(e)}",
        )
    except Exception as e:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error=f"Tool execution failed: {str(e)}",
        )
