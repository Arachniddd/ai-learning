import os
import json
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

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
    
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role" : "system",
                "content" : "你只返回合法 JSON，不要输出 Markdown，不要输出解释。"
            },
            {
                "role" : "user",
                "content" : prompt
            }
        ],
        temperature=0.2,
        response_format={"type" : "json_object"}
    )

    content = response.choices[0].message.content
    return json.loads(content)


def answer_with_context(question: str, contexts: list[dict]):
    context_text = "\n\n".join(
        [f"chunk {c['id']} from {c['source']}]\n{c['content']}" for c in contexts]
    )

    prompt = f"""
    你是一个计算机课程助教。请只根据给定上下文回答问题。

    如果上下文不足以回答，请明确说“根据当前资料无法确定”。

    上下文：
    {context_text}

    问题：
    {question}

    请返回 JSON，格式如下：
    {{
    "answer": "你的回答",
    "used_chunks": [0, 1]
    }}
    """

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role" : "system",
                "content" : "你只返回合法 JSON，不要输出 Markdown，不要输出解释。"
            },
            {
                "role" : "user",
                "content" : prompt
            }
        ],
        temperature=0.2,
        response_format={"type":"json_object"}
    )

    content = response.choices[0].message.content
    return json.loads(content)


def decide_tool_use(message : str) -> dict:
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
    {message}
    """

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role" : "system",
                "content" : "你只返回合法 JSON，不要输出 Markdown，不要输出解释。"
            },
            {
                "role" : "user",
                "content" : prompt
            }
        ],
        temperature=0.1,
        response_format={"type" : "json_object"}
    )

    content = response.choices[0].message.content
    return json.loads(content)


def final_answer_with_tool_result(message : str, tool_result : list[dict]) -> dict:
    context_text = "\n\n".join(
        [
            f"chunk{c["id"]} from {c["source"]}\n{c["content"]}"
            for c in tool_result
        ]
    )

    prompt = f"""
    你是一个计算机课程助教。请根据工具检索到的资料回答用户问题。

    用户问题：
    {message}

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

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "你只返回合法 JSON，不要输出 Markdown，不要输出解释。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)