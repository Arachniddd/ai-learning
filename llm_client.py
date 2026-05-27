import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://platform.deepseek.com/"
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
        model="deepseek-chat",
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

