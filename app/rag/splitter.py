import json
import sys
from pathlib import Path


def split_by_paragraph(text: str) -> list[str]:
    paragraphs = text.split("\n\n")
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


def split_file(input_path: str | Path, output_path: str | Path) -> int:
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    if input_file.suffix not in [".txt", ".md"]:
        raise ValueError("Only .txt and .md files are supported now.")

    text = input_file.read_text(encoding="utf-8")
    chunks = split_by_paragraph(text)

    data = [
        {
            "id": index,
            "content": chunk,
            "source": str(input_file),
        }
        for index, chunk in enumerate(chunks)
    ]

    output_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return len(chunks)


def split_by_size(text : str, chunk_size : int = 500, overlap : int = 100):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    start = 0
    chunks = []

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)
        
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
