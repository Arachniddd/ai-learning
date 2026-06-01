import time

from app.agent.planner import plan_next_step
from app.agent.tools import search_knowledge_base, list_chunks, summarize_text
from app.llm.client import final_answer_with_tool_result
from app.observability.logger import write_agent_log


def run_agent(message: str) -> dict:
    start_time = time.time()

    trace = {
        "message": message,
        "decision": None,
        "tool_name": None,
        "tool_arguments": None,
        "tool_result": None,
        "answer": None,
        "error": None,
    }

    try:
        decision = plan_next_step(message)
        trace["decision"] = decision

        if not decision.get("need_tool"):
            answer = decision.get("answer", "")
            trace["answer"] = answer
            trace["duration_ms"] = int((time.time() - start_time) * 1000)
            write_agent_log(trace)

            return {
                "mode": "direct_answer",
                "answer": answer,
                "trace": trace,
            }

        tool_name = decision.get("tool_name")
        args = decision.get("arguments", {})

        trace["tool_name"] = tool_name
        trace["tool_arguments"] = args

        if tool_name == "search_knowledge_base":
            query = args.get("query", message)
            top_k = int(args.get("top_k", 3))

            tool_result = search_knowledge_base(query=query, top_k=top_k)
            trace["tool_result"] = tool_result

            result = final_answer_with_tool_result(
                message=message,
                tool_result=tool_result,
            )

            trace["answer"] = result.get("answer")
            trace["used_chunks"] = result.get("used_chunks", [])
            trace["duration_ms"] = int((time.time() - start_time) * 1000)
            write_agent_log(trace)

            return {
                "mode": "tool_call",
                "tool_name": tool_name,
                "tool_arguments": args,
                "contexts": tool_result,
                "answer": result.get("answer"),
                "used_chunks": result.get("used_chunks", []),
                "trace": trace,
            }

        if tool_name == "list_chunks":
            limit = int(args.get("limit", 20))
            chunks = list_chunks(limit=limit)

            answer = f"当前知识库返回了 {len(chunks)} 个 chunk。"

            trace["answer"] = answer
            trace["tool_result"] = chunks
            trace["duration_ms"] = int((time.time() - start_time) * 1000)
            write_agent_log(trace)

            return {
                "mode": "tool_call",
                "tool_name": tool_name,
                "tool_arguments": args,
                "tool_result": chunks,
                "answer": answer,
                "trace": trace,
            }

        if tool_name == "summarize_text":
            text = args.get("text", "")
            summary = summarize_text(text)

            answer = summary.get("summary", "")

            trace["answer"] = answer
            trace["tool_result"] = summary
            trace["duration_ms"] = int((time.time() - start_time) * 1000)
            write_agent_log(trace)

            return {
                "mode": "tool_call",
                "tool_name": tool_name,
                "tool_arguments": args,
                "tool_result": summary,
                "answer": answer,
                "trace": trace,
            }

        trace["error"] = f"Unknown tool: {tool_name}"
        trace["duration_ms"] = int((time.time() - start_time) * 1000)
        write_agent_log(trace)

        return {
            "error": f"Unknown tool: {tool_name}",
            "trace": trace,
        }

    except Exception as e:
        trace["error"] = str(e)
        trace["duration_ms"] = int((time.time() - start_time) * 1000)
        write_agent_log(trace)

        return {
            "error": str(e),
            "trace": trace,
        }