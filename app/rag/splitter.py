import json
import sys
import uuid
import re
from pathlib import Path

from app.models.chunk import Chunk

def estimate_token_count(text: str) -> int:
    """
    粗略估算 token 数。
    中文场景可以先粗略用 len(text) / 2。
    后面如果你接 tiktoken，再改成精确计算。
    """
    return max(1, len(text) // 2)



def build_chunk_id(source: str, chunk_index: int) -> str:
    safe_source = source.replace("/", "_").replace(" ", "_")
    return f"{safe_source}:{uuid.uuid4().hex}:{chunk_index}"


def split_by_paragraph(text: str, source: str = "user_input") -> list[Chunk]:
    paragraphs = text.split("\n\n")
    chunks = []

    for index, paragraph in enumerate(paragraphs):
        content = paragraph.strip()

        if content:
            chunks.append(
                Chunk(
                    id=build_chunk_id(source, index),
                    content=content,
                    source=source,
                    chunk_index=index,
                )
            )

    return chunks


def split_file(input_path: str | Path, output_path: str | Path) -> int:
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    if input_file.suffix not in [".txt", ".md"]:
        raise ValueError("Only .txt and .md files are supported now.")

    text = input_file.read_text(encoding="utf-8")
    chunks = split_by_paragraph(text, source=str(input_file))

    data = [
        chunk.model_dump()
        for chunk in chunks
    ]

    output_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return len(chunks)


def split_plain_text(
    text : str,
    source: str = "user_input",
    chunk_size : int = 500,
    overlap : int = 100,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    start = 0
    chunks = []
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        content = text[start:end].strip()

        if content:
            chunks.append(
                Chunk(
                    id=build_chunk_id(source, chunk_index),
                    content=content,
                    source=source,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
        
        start = end - overlap

    return chunks


def split_markdown_by_headers(
    text: str,
    source: str,
    max_section_chars: int = 1000,
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[Chunk]:
    if max_section_chars <= 0:
        raise ValueError("max_section_chars must be positive")

    lines = text.splitlines()

    current_headers = []
    current_content = []
    sections = []

    def flush_section():
        nonlocal current_content

        content = "\n".join(current_content).strip()
        if not content:
            current_content = []
            return

        section_name = " / ".join(current_headers)
        sections.append({
            "section": section_name,
            "content": content,
        })
        current_content = []

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.*)$", line)

        if match:
            flush_section()

            level = len(match.group(1))
            title = match.group(2).strip()

            current_headers = current_headers[:level - 1]
            current_headers.append(title)

            current_content.append(line)
        else:
            current_content.append(line)

    flush_section()

    chunks = []
    chunk_index = 0

    for section in sections:
        content = section["content"]
        section_name = section["section"]

        if len(content) <= max_section_chars:
            chunks.append(
                Chunk(
                    id=build_chunk_id(source, chunk_index),
                    content=content,
                    source=source,
                    section=section_name,
                    chunk_index=chunk_index,
                    token_count=estimate_token_count(content),
                )
            )
            chunk_index += 1
        else:
            sub_chunks = split_plain_text(
                text=content,
                source=source,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for sub in sub_chunks:
                chunks.append(
                    sub.model_copy(
                        update={
                            "id": build_chunk_id(source, chunk_index),
                            "section": section_name,
                            "chunk_index": chunk_index,
                            "token_count": estimate_token_count(sub.content),
                        }
                    )
                )
                chunk_index += 1

    return chunks
    


def split_text(
    text: str,
    source: str,
    max_section_chars: int = 1000,
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[Chunk]:
    if source.endswith(".md") or "#" in text[:200]:
        return split_markdown_by_headers(
            text=text,
            source=source,
            max_section_chars=max_section_chars,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    return split_plain_text(
        text=text,
        source=source,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def main():
    if len(sys.argv) != 3:
        print("Usage: python splitter.py input.txt output.json")
        return

    try:
        chunk_count = split_file(sys.argv[1], sys.argv[2])
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return

    print(f"Saved {chunk_count} data chunks into {sys.argv[2]}.")


if __name__ == "__main__":
    main()
