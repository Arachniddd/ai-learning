import json


def build_tool_decision_prompt(message: str) -> str:
    return f"""
你是一个 AI Agent。你可以使用以下工具：

工具 1：
名称：search_knowledge_base
作用：从用户上传的课程资料中检索相关内容
参数：
{{
  "query": "检索问题",
  "top_k": 3
}}

工具 2：
名称：list_chunks
作用：查看当前知识库中已经存储的文本片段
参数：
{{
  "limit": 20
}}

工具 3：
名称：summarize_text
作用：总结用户直接提供的一段文本
参数：
{{
  "text": "需要总结的文本"
}}

请判断用户的问题是否需要使用工具。

如果需要查询知识库，只返回 JSON：
{{
  "need_tool": true,
  "tool_name": "search_knowledge_base",
  "arguments": {{
    "query": "要检索的问题",
    "top_k": 3
  }}
}}

如果需要查看已有 chunk，只返回 JSON：
{{
  "need_tool": true,
  "tool_name": "list_chunks",
  "arguments": {{
    "limit": 20
  }}
}}

如果需要总结用户直接给出的文本，只返回 JSON：
{{
  "need_tool": true,
  "tool_name": "summarize_text",
  "arguments": {{
    "text": "需要总结的文本"
  }}
}}

如果不需要工具，只返回 JSON：
{{
  "need_tool": false,
  "answer": "直接回答用户的问题"
}}

用户问题：
{message}
""".strip()


def build_agent_planner_prompt(
    message: str,
    tools: list[dict],
    previous_steps: list[dict] | None = None,
) -> str:
    previous_steps = previous_steps or []

    tools_text = json.dumps(tools, ensure_ascii=False, indent=2)
    previous_steps_text = json.dumps(previous_steps, ensure_ascii=False, indent=2)

    return f"""
  你是一个学习助手 Agent 的规划器。

  你需要根据用户消息，决定下一步动作。

  你可以选择两种动作：

  1. tool_call
  表示需要调用工具。
  格式：
  {{
    "action_type": "tool_call",
    "tool_name": "工具名",
    "arguments": {{
      "参数名": "参数值"
    }},
    "thought": "简短说明为什么调用这个工具"
  }}

  2. final_answer
  表示不需要继续调用工具，直接给出最终回答。
  格式：
  {{
    "action_type": "final_answer",
    "answer": "最终回答内容",
    "thought": "简短说明为什么可以直接回答"
  }}

  可用工具：
  {tools_text}

  此前步骤：
  {previous_steps_text}

  决策规则：
  1. 如果用户问题需要查询上传资料、课程笔记、知识库，调用 search_knowledge_base。
  2. 如果用户想查看知识库中已有 chunk，调用 list_chunks。
  3. 如果用户明确要求总结一段给定文本，调用 summarize_text。
  4. 如果工具结果已经足够回答问题，输出 final_answer。
  5. 如果资料不足，不要编造，应在 final_answer 中说明资料不足。
  6. 最多只调用必要工具，不要为了调用工具而调用工具。
  7. 只输出 JSON，不要输出其他解释。

  用户消息：
  {message}

  请输出下一步动作：
  """.strip()
