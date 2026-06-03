import time
import uuid
from typing import Any

from pydantic import ValidationError

from app.agent.planner import plan_next_action
from app.agent.registry import get_tool_spec
from app.agent.schemas import AgentAction, AgentStep, ToolResult
from app.observability.logger import write_agent_log


MAX_AGENT_STEPS = 5


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())

    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    return value


def execute_tool(action: AgentAction) -> ToolResult:
    tool_name = action.tool_name or ""
    tool_spec = get_tool_spec(tool_name)

    if tool_spec is None:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            error=f"Unknown tool: {tool_name}",
        )

    try:
        args_model = tool_spec.args_schema.model_validate(action.arguments)
        result = tool_spec.func(**args_model.model_dump())

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
            error=str(e),
        )


def run_agent(message: str) -> dict:
    start_time = time.time()
    workflow_id = str(uuid.uuid4())
    steps: list[AgentStep] = []
    latest_tool_output = None
    latest_contexts = None
    latest_retrieval_debug = None

    trace = {
        "workflow_id": workflow_id,
        "message": message,
        "steps": [],
        "answer": None,
        "error": None,
    }

    try:
        for step_index in range(MAX_AGENT_STEPS):
            previous_steps = [to_jsonable(step) for step in steps]
            action = plan_next_action(
                message=message,
                previous_steps=previous_steps,
            )

            if action.action_type == "final_answer":
                step = AgentStep(
                    step_index=step_index,
                    action=action,
                    observation=None,
                )
                steps.append(step)

                answer = action.answer or ""
                trace["answer"] = answer
                trace["steps"] = [to_jsonable(item) for item in steps]
                trace["duration_ms"] = int((time.time() - start_time) * 1000)
                write_agent_log(trace)

                mode = "tool_call" if any(item.observation for item in steps) else "direct_answer"
                response = {
                    "mode": mode,
                    "answer": answer,
                    "steps": trace["steps"],
                    "trace": trace,
                }

                if latest_tool_output is not None:
                    response["tool_result"] = latest_tool_output

                if latest_contexts is not None:
                    response["contexts"] = latest_contexts

                if latest_retrieval_debug is not None:
                    response["retrieval_debug"] = latest_retrieval_debug

                return response

            observation = execute_tool(action)
            step_retrieval_debug = None

            latest_tool_output = observation.result

            if action.tool_name == "search_knowledge_base" and isinstance(observation.result, dict):
                latest_contexts = observation.result.get("chunks")
                latest_retrieval_debug = observation.result.get("retrieval_debug")
                step_retrieval_debug = latest_retrieval_debug

            step = AgentStep(
                step_index=step_index,
                action=action,
                observation=observation,
                retrieval_debug=step_retrieval_debug,
            )
            steps.append(step)

            trace["steps"] = [to_jsonable(item) for item in steps]

            trace["duration_ms"] = int((time.time() - start_time) * 1000)
            write_agent_log(trace)

        answer = f"Agent 已达到最大步骤数 {MAX_AGENT_STEPS}，未获得 final_answer。"
        trace["answer"] = answer
        trace["error"] = answer
        trace["steps"] = [to_jsonable(item) for item in steps]
        trace["duration_ms"] = int((time.time() - start_time) * 1000)
        write_agent_log(trace)

        return {
            "mode": "max_steps_reached",
            "answer": answer,
            "tool_result": latest_tool_output,
            "contexts": latest_contexts,
            "retrieval_debug": latest_retrieval_debug,
            "steps": trace["steps"],
            "trace": trace,
        }

    except Exception as e:
        trace["error"] = str(e)
        trace["steps"] = [to_jsonable(item) for item in steps]
        trace["duration_ms"] = int((time.time() - start_time) * 1000)
        write_agent_log(trace)

        return {
            "error": str(e),
            "steps": trace["steps"],
            "trace": trace,
        }
