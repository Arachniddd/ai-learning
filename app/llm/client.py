import os
import json
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.core.json_utils import extract_json_array
from app.prompts.agent import build_tool_decision_prompt
from app.prompts.llm import (
    DEFAULT_SYSTEM_PROMPT,
    JSON_ONLY_SYSTEM_PROMPT,
    RAG_ANSWER_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    build_explain_concept_prompt,
    build_generate_quiz_prompt,
    build_query_rewrite_prompt,
    build_rag_answer_prompt,
    build_rerank_prompt,
    build_summarize_prompt,
)
from app.models.chunk import Chunk, RetrieveChunk


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
    chunks: list[Chunk],
    model: str = "deepseek-v4-flash",
) -> dict:
    if not chunks:
        return {
            "answer": "我没有在知识库中检索到足够相关的内容，无法基于资料回答这个问题。",
            "used_chunks": [],
        }

    prompt = build_rag_answer_prompt(
        question=question,
        chunks=chunks,
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
                "id": chunk.id,
                "source": chunk.source,
                "section": chunk.section,
                "score": getattr(chunk, "score", None),
                "rerank_score": getattr(chunk, "rerank_score", None),
            }
            for chunk in chunks
        ],
    }


def answer_with_context(question: str, contexts: list[Chunk]) -> dict:
    return answer_with_chunks(question=question, chunks=contexts)


def generate_quiz_from_chunks(
    topic: str,
    chunks: list[Chunk],
    num_questions: int = 5,
) -> dict:
    prompt = build_generate_quiz_prompt(
        topic=topic,
        chunks=chunks,
        num_questions=num_questions,
    )

    quiz = chat_with_llm(prompt=prompt)

    return {
        "topic": topic,
        "num_questions": num_questions,
        "quiz": quiz,
        "used_chunks": [
            {
                "id": chunk.id,
                "source": chunk.source,
                "section": chunk.section,
                "score": getattr(chunk, "score", None),
                "rerank_score": getattr(chunk, "rerank_score", None),
            }
            for chunk in chunks
        ],
    }


def explain_with_chunks(
    concept: str,
    chunks: list[Chunk],
    detail_level: str = "medium",
) -> dict:
    if not chunks:
        return {
            "concept": concept,
            "detail_level": detail_level,
            "explanation": "没有检索到足够相关的资料，无法基于资料解释这个概念。",
            "used_chunks": [],
        }

    prompt = build_explain_concept_prompt(
        concept=concept,
        chunks=chunks,
        detail_level=detail_level,
    )

    explanation = chat_with_llm(
        prompt=prompt,
        system_prompt=RAG_ANSWER_SYSTEM_PROMPT,
        temperature=0.2,
    )

    return {
        "concept": concept,
        "detail_level": detail_level,
        "explanation": explanation,
        "used_chunks": [
            {
                "id": chunk.id,
                "source": chunk.source,
                "section": chunk.section,
                "score": getattr(chunk, "score", None),
                "rerank_score": getattr(chunk, "rerank_score", None),
            }
            for chunk in chunks
        ],
    }


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
        chunks=chunks,
    )

    raw = chat_with_llm(
        prompt=prompt,
        system_prompt=JSON_ONLY_SYSTEM_PROMPT,
        temperature=0.1,
    )

    try:
        scores = extract_json_array(raw)

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
