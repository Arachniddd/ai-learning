import os
import json
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.rag.types import Chunk


load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def chat_with_llm(
    prompt: str,
    system_prompt: str = "你是一个可靠的 AI 助手。",
    model: str = "deepseek-v4-flash",
    temperature: float = 0.2,
    response_format: dict[str, Any] | None = None,
) -> str:
    request_args = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": temperature,
    }

    if response_format is not None:
        request_args["response_format"] = response_format

    response = client.chat.completions.create(**request_args)
    return response.choices[0].message.content or ""


def summarize_note(text : str) -> dict:
    prompt =f"""
    你是一个计算机课程助教。请总结下面这段课程笔记。

    要求返回 JSON，格式如下：
    {{
    "summary": "一段简短总结",
    "key_points": ["重点1", "重点2", "重点3"],
    "questions": ["复习问题1", "复习问题2"]
    }}

    课程笔记：
    {text}
    """
    content = chat_with_llm(
        prompt=prompt,
        system_prompt="你只返回合法 JSON，不要输出 Markdown，不要输出解释。",
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    return json.loads(content)


def answer_with_context(question: str, contexts: list[Chunk]):
    rewritten_question = rewrite_query(question)
    context_text = "\n\n".join(
        [f"chunk {c.id} from {c.source}]\n{c.content}" for c in contexts]
    )

    prompt = f"""
    你是一个计算机课程助教。请只根据给定上下文回答问题。

    如果上下文不足以回答，请明确说“根据当前资料无法确定”。

    上下文：
    {context_text}

    问题：
    {rewritten_question}

    请返回 JSON，格式如下：
    {{
    "answer": "你的回答",
    "used_chunks": [0, 1]
    }}
    """

    content = chat_with_llm(
        prompt=prompt,
        system_prompt="你只返回合法 JSON，不要输出 Markdown，不要输出解释。",
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    return json.loads(content)


def decide_tool_use(message : str) -> dict:
    rewritten_message = rewrite_query(message)
    prompt = f"""
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
    {{ß
    "text": "需要总结的文本"
    }}

    请判断用户的问题是否需要查询知识库。

    如果需要查询知识库，只返回 JSON：
    {{
    "need_tool": true,
    "tool_name": "search_knowledge_base",
    "arguments": {{
        "query": "要检索的问题",
        "top_k": 3
    }}
    }}

    如果不需要查询知识库，只返回 JSON：
    {{
    "need_tool": false,
    "answer": "直接回答用户的问题"
    }}

    用户问题：
    {rewritten_message}
    """

    content = chat_with_llm(
        prompt=prompt,
        system_prompt="你只返回合法 JSON，不要输出 Markdown，不要输出解释。",
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    return json.loads(content)


def final_answer_with_tool_result(message : str, tool_result : list[Chunk]) -> dict:
    rewritten_message = rewrite_query(message)
    context_text = "\n\n".join(
        [
            f"chunk {c.id} from {c.source}\n{c.content}"
            for c in tool_result
        ]
    )

    prompt = f"""
    你是一个计算机课程助教。请根据工具检索到的资料回答用户问题。

    用户问题：
    {rewritten_message}

    工具返回的资料：
    {context_text}

    要求：
    1. 只根据工具资料回答
    2. 如果资料不足，请说“根据当前资料无法确定”
    3. 返回 JSON

    JSON 格式：
    {{
    "answer": "你的回答",
    "used_chunks": ["chunk_id_1", "chunk_id_2"]
    }}
    """

    content = chat_with_llm(
        prompt=prompt,
        model="deepseek-chat",
        system_prompt="你只返回合法 JSON，不要输出 Markdown，不要输出解释。",
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    return json.loads(content)

def rewrite_query(question: str) -> str:
    prompt = f"""
你是一个 RAG 检索查询改写器。

你的任务：
把用户问题改写成更适合向量检索的查询。
要求：
1. 保留用户原始意图。
2. 补充可能相关的专业关键词。
3. 不要回答问题。
4. 只输出改写后的查询，不要输出解释。

用户问题：
{question}
"""

    response = chat_with_llm(prompt)

    rewritten = response.strip()

    if not rewritten:
        return question

    return rewritten
