import json
import time
from pathlib import Path
from typing import Any

LOG_PATH = Path("logs") / "agent_logs.jsonl"

def json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

def write_agent_log(record : dict[str, Any]) -> None:
    record["timestamp"] = time.time()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")

def read_agent_logs(limit : int = 20) ->list[dict[str,Any]]:
    if not LOG_PATH.exists():
        return []
    
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()

    recent_lines = lines[-limit:]

    logs = []

    for line in recent_lines:
        try:
            logs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    
    return logs
    
