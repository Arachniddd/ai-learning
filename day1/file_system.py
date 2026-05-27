from pathlib import Path

# path  = Path("os_note.txt")

# text = path.read_text(encoding="utf-8")

# print(text)

path = Path("output.txt")

path.write_text("I am the storm that is appoarching!", encoding="utf-8")