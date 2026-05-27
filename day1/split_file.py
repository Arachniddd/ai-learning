import sys
from pathlib import Path
from chunking import split_by_paragraph
import json

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 split_file.py input.txt output.json")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    text = Path(input_path).read_text("utf-8")

    chunks = split_by_paragraph(text)

    data = []

    for i, chunk in enumerate(chunks):
        data.append({
            "id" : i,
            "content" : chunk,
            "source" : input_path
        })

    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved {len(chunks)} data chunks into {output_path}.")

if __name__ == "__main__":
    main()

    