import json
import sys
import uuid
from pathlib import Path

from app.rag.types import Chunk


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


def split_by_size(
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
