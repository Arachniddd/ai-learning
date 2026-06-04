import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

from pydantic import ValidationError

from app.agent.registry import get_tool_spec
from app.agent.schemas import ToolResult


TOOL_TIMEOUT_SECONDS = 20
_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def elapsed_ms(start_time: float) -> int:
    return int((time.monotonic() - start_time) * 1000)


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    timeout_seconds: int = TOOL_TIMEOUT_SECONDS,
) -> ToolResult:
    start_time = time.monotonic()
    tool_spec = get_tool_spec(tool_name)

    if tool_spec is None:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error=f"Unknown tool: {tool_name}",
            duration_ms=elapsed_ms(start_time),
        )

    try:
        validated_args = tool_spec.args_schema.model_validate(arguments)
        future = _TOOL_EXECUTOR.submit(
            tool_spec.func,
            **validated_args.model_dump(),
        )
        result = future.result(timeout=timeout_seconds)

        return ToolResult(
            success=True,
            tool_name=tool_name,
            result=result,
            duration_ms=elapsed_ms(start_time),
        )
    except TimeoutError:
        future.cancel()
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error=f"Tool execution timed out after {timeout_seconds} seconds.",
            duration_ms=elapsed_ms(start_time),
            timed_out=True,
        )
    except ValidationError as e:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error=f"Tool arguments validation failed: {str(e)}",
            duration_ms=elapsed_ms(start_time),
        )
    except Exception as e:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error=f"Tool execution failed: {str(e)}",
            duration_ms=elapsed_ms(start_time),
        )
