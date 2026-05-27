import os
from typing import Any

import requests


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def get_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)
