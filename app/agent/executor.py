from app.agent.planner import plan_next_action
from app.agent.tool_executor import execute_tool
from app.agent.tracer import AgentTrace
from app.observability.logger import write_agent_log


MAX_AGENT_STEPS = 5


def run_agent(message: str, max_steps: int = MAX_AGENT_STEPS) -> dict:
    trace = AgentTrace(message=message)

    try:
        for step_index in range(max_steps):
            action = plan_next_action(
                message=message,
                previous_steps=trace.steps_json(),
            )

            if action.action_type == "final_answer":
                trace.add_final_step(step_index=step_index, action=action)
                write_agent_log(trace.to_dict())
                return trace.to_response()

            observation = execute_tool(
                tool_name=action.tool_name or "",
                arguments=action.arguments,
            )
            trace.add_tool_step(
                step_index=step_index,
                action=action,
                observation=observation,
            )

            if observation.timed_out:
                error = observation.error or "Tool execution timed out."
                trace.finish(answer=error, error=error)
                write_agent_log(trace.to_dict())
                return trace.to_error_response()

            write_agent_log(trace.to_dict())

        answer = f"Agent 已达到最大步骤数 {max_steps}，未获得 final_answer。"
        trace.finish(answer=answer, error=answer)
        write_agent_log(trace.to_dict())
        return trace.to_response(mode="max_steps_reached")

    except Exception as e:
        trace.finish(error=str(e))
        write_agent_log(trace.to_dict())
        return trace.to_error_response()
