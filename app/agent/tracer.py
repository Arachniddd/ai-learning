import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.agent.schemas import AgentAction, AgentStep, ToolResult


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())

    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    return value


@dataclass
class AgentTrace:
    message: str = ""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)
    steps: list[AgentStep] = field(default_factory=list)
    answer: str | None = None
    error: str | None = None
    latest_tool_output: Any = None
    latest_contexts: Any = None
    latest_retrieval_debug: Any = None

    def add_final_step(self, step_index: int, action: AgentAction) -> AgentStep:
        step = AgentStep(
            step_index=step_index,
            action=action,
            observation=None,
        )
        self.steps.append(step)
        self.answer = action.answer or ""
        return step

    def add_tool_step(
        self,
        step_index: int,
        action: AgentAction,
        observation: ToolResult,
    ) -> AgentStep:
        retrieval_debug = self.extract_retrieval_debug(observation.result)
        self.update_latest_tool_state(observation.result)

        step = AgentStep(
            step_index=step_index,
            action=action,
            observation=observation,
            retrieval_debug=retrieval_debug,
        )
        self.steps.append(step)
        return step

    def extract_retrieval_debug(self, tool_result: Any) -> dict[str, Any] | None:
        if isinstance(tool_result, dict):
            return tool_result.get("retrieval_debug")

        return None

    def update_latest_tool_state(self, tool_result: Any) -> None:
        self.latest_tool_output = tool_result

        if not isinstance(tool_result, dict):
            return

        if "chunks" in tool_result:
            self.latest_contexts = tool_result.get("chunks")

        if "retrieval_debug" in tool_result:
            self.latest_retrieval_debug = tool_result.get("retrieval_debug")

    def finish(self, answer: str | None = None, error: str | None = None) -> None:
        if answer is not None:
            self.answer = answer

        if error is not None:
            self.error = error

    def duration_ms(self) -> int:
        return int((time.time() - self.started_at) * 1000)

    def has_tool_observation(self) -> bool:
        return any(step.observation for step in self.steps)

    def response_mode(self) -> str:
        if self.has_tool_observation():
            return "tool_call"

        return "direct_answer"

    def steps_json(self) -> list[dict[str, Any]]:
        return [to_jsonable(step) for step in self.steps]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "message": self.message,
            "steps": self.steps_json(),
            "answer": self.answer,
            "error": self.error,
            "duration_ms": self.duration_ms(),
        }

    def to_response(self, mode: str | None = None) -> dict[str, Any]:
        trace = self.to_dict()
        response = {
            "mode": mode or self.response_mode(),
            "answer": self.answer or "",
            "steps": trace["steps"],
            "trace": trace,
        }

        if self.latest_tool_output is not None:
            response["tool_result"] = self.latest_tool_output

        if self.latest_contexts is not None:
            response["contexts"] = self.latest_contexts

        if self.latest_retrieval_debug is not None:
            response["retrieval_debug"] = self.latest_retrieval_debug

        return response

    def to_error_response(self) -> dict[str, Any]:
        trace = self.to_dict()
        return {
            "error": self.error,
            "steps": trace["steps"],
            "trace": trace,
        }
