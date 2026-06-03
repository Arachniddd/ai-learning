import os
import json
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.llm.prompt import (
    DEFAULT_SYSTEM_PROMPT,
    JSON_ONLY_SYSTEM_PROMPT,
    RAG_ANSWER_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    build_query_rewrite_prompt,
    build_rag_answer_prompt,
    build_rerank_prompt,
    build_summarize_prompt,
    build_tool_decision_prompt,
)
from app.rag.types import Chunk, RetrieveChunk


load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def chat_with_llm(
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
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


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [chunk.model_dump() for chunk in chunks]


def summarize_note(text : str) -> dict:
    prompt = build_summarize_prompt(text)
    content = chat_with_llm(
        prompt=prompt,
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        temperature=0.2,
    )

    return {
        "summary": content,
        "key_points": [],
        "questions": [],
    }


def answer_with_context(question: str, contexts: list[Chunk]):
    rewritten_question = rewrite_query(question)
    prompt = build_rag_answer_prompt(
        question=rewritten_question,
        chunks=chunks_to_dicts(contexts),
    )

    content = chat_with_llm(
        prompt=prompt,
        system_prompt=RAG_ANSWER_SYSTEM_PROMPT,
        temperature=0.2,
    )

    return {
        "answer": content,
        "used_chunks": [chunk.id for chunk in contexts],
    }


def decide_tool_use(message : str) -> dict:
    rewritten_message = rewrite_query(message)
    prompt = build_tool_decision_prompt(rewritten_message)

    content = chat_with_llm(
        prompt=prompt,
        system_prompt=JSON_ONLY_SYSTEM_PROMPT,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    return json.loads(content)


def final_answer_with_tool_result(message : str, tool_result : list[Chunk]) -> dict:
    rewritten_message = rewrite_query(message)
    prompt = build_rag_answer_prompt(
        question=rewritten_message,
        chunks=chunks_to_dicts(tool_result),
    )

    content = chat_with_llm(
        prompt=prompt,
        model="deepseek-chat",
        system_prompt=RAG_ANSWER_SYSTEM_PROMPT,
        temperature=0.2,
    )

    return {
        "answer": content,
        "used_chunks": [chunk.id for chunk in tool_result],
    }


def rewrite_query(question: str) -> str:
    prompt = build_query_rewrite_prompt(question)

    response = chat_with_llm(prompt)

    rewritten = response.strip()

    if not rewritten:
        return question

    return rewritten

def rerank_chunks(
    question : str,
    chunks : list[RetrieveChunk],
    top_k : int = 3        
)->list[RetrieveChunk]:
    prompt = build_rerank_prompt(
        question=question,
        chunks=chunks_to_dicts(chunks),
    )

    raw = chat_with_llm(
        prompt=prompt,
        system_prompt=JSON_ONLY_SYSTEM_PROMPT,
        temperature=0.1,
    )

    try:
        match = re.search(r"\[.*\]", raw, re.S)

        if not match:
            return chunks[:top_k]
        
        scores = json.loads(match.group(0))

        score_map = {
            item["candidate_index"]: item
            for item in scores
            if "candidate_index" in item and "score" in item
        }

        reranked = []

        for idx, chunk in enumerate(chunks):
            item = score_map.get(idx, {})

            new_chunk = chunk.model_copy(
                update={
                    "rerank_score": item.get("score", 0),
                    "rerank_reason": item.get("reason", ""),
                }
            )
            reranked.append(new_chunk)

        reranked.sort(key=lambda chunk: chunk.rerank_score or 0, reverse=True)

        return [
            chunk for chunk in reranked
            if (chunk.rerank_score or 0) >= 5
        ][:top_k]
    
    except Exception:
        return chunks[:top_k]
