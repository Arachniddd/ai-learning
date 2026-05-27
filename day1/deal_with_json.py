import json
from pathlib import Path
from chunking import split_by_paragraph

text = Path("os_note.txt").read_text(encoding="utf-8")

chunks = split_by_paragraph(text)

data = []

for i, chunk in enumerate(chunks):
    data.append({
        "id" : i,
        "content" : chunk,
        "source" : "os_note.txt"
    })

Path("chunks.json").write_text(
    json.dumps(data, ensure_ascii=False, indent=2),
    encoding="utf-8"
)


