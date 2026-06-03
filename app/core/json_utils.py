import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)

    if not match:
        raise ValueError("No JSON object found")

    return json.loads(match.group(0))


def extract_json_array(text: str) -> list[Any]:
    match = re.search(r"\[.*\]", text, re.S)

    if not match:
        raise ValueError("No JSON array found")

    return json.loads(match.group(0))
