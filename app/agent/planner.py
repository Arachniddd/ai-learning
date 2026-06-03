import json
import re

from pydantic import ValidationError

from app.agent.registry import list_tool_specs
from app.agent.schemas import AgentAction
from app.llm.client import chat_with_llm
from app.prompts.agent import build_agent_planner_prompt
from app.prompts.llm import JSON_ONLY_SYSTEM_PROMPT


def parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)

    if not match:
        raise ValueError("No JSON object found in LLM output")

    return json.loads(match.group(0))


def plan_next_action(
    message: str,
    previous_steps: list[dict] | None = None,
) -> AgentAction:
    tools = list_tool_specs()
    prompt = build_agent_planner_prompt(
        message=message,
        tools=tools,
        previous_steps=previous_steps or [],
    )

    raw = chat_with_llm(
        prompt=prompt,
        system_prompt=JSON_ONLY_SYSTEM_PROMPT,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    try:
        data = parse_json_object(raw)
        return AgentAction.model_validate(data)
    except (ValueError, json.JSONDecodeError, ValidationError) as e:
        return AgentAction(
            action_type="final_answer",
            answer=f"Agent 决策解析失败：{str(e)}。模型原始输出：{raw}",
            thought="模型没有输出合法 JSON，因此终止执行。",
        )
