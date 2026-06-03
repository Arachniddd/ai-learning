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


def chunks_to_dicts(chunks: list[Chunk | dict[str, Any]]) -> list[dict[str, Any]]:
    result = []

    for chunk in chunks:
        if hasattr(chunk, "model_dump"):
            result.append(chunk.model_dump())
        else:
            result.append(dict(chunk))

    return result


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


def rewrite_query(question: str) -> str:
    prompt = build_query_rewrite_prompt(question)
    response = chat_with_llm(prompt=prompt)
    rewritten = response.strip()

    if not rewritten:
        return question

    return rewritten


def answer_with_chunks(
    question: str,
    chunks: list[Chunk | dict[str, Any]],
    model: str = "deepseek-v4-flash",
) -> dict:
    if not chunks:
        return {
            "answer": "我没有在知识库中检索到足够相关的内容，无法基于资料回答这个问题。",
            "used_chunks": [],
        }

    chunk_dicts = chunks_to_dicts(chunks)
    prompt = build_rag_answer_prompt(
        question=question,
        chunks=chunk_dicts,
    )

    content = chat_with_llm(
        prompt=prompt,
        model=model,
        system_prompt=RAG_ANSWER_SYSTEM_PROMPT,
        temperature=0.2,
    )

    return {
        "answer": content,
        "used_chunks": [
            {
                "id": chunk.get("id"),
                "source": chunk.get("source", ""),
                "section": chunk.get("section", ""),
                "score": chunk.get("score"),
                "rerank_score": chunk.get("rerank_score"),
            }
            for chunk in chunk_dicts
        ],
    }


def answer_with_context(question: str, contexts: list[Chunk]) -> dict:
    return answer_with_chunks(question=question, chunks=contexts)


def decide_tool_use(message : str) -> dict:
    prompt = build_tool_decision_prompt(message)

    content = chat_with_llm(
        prompt=prompt,
        system_prompt=JSON_ONLY_SYSTEM_PROMPT,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    return json.loads(content)


def final_answer_with_tool_result(message : str, tool_result : list[Chunk]) -> dict:
    return answer_with_chunks(
        question=message,
        chunks=tool_result,
        model="deepseek-chat",
    )


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
