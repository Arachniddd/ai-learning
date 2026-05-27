from pathlib import Path

text = Path("os_note.txt").read_text(encoding="utf-8")

# paragraphs = text.split("\n\n")

# for i , paragraph in enumerate(paragraphs):
#     print("---- chunk", i, "----")
#     print(paragraph)


def split_by_paragraph(text : str) -> list[str]:
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]

split_by_paragraph(text)