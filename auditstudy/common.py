from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def dumps(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)

def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def write_jsonl(path, rows):
    Path(path).write_text("".join(dumps(row) + "\n" for row in rows))

def config(path=None):
    c = read_json(path or ROOT / "config/study.json")
    # The implemented three bins align with the release threshold. Other
    # thresholds require a protocol amendment, not silently different math.
    if (c["threshold"], c["reward_correct"], c["reward_wrong"]) != (0.8, 1, -4):
        raise ValueError("This protocol implements threshold .8 and rewards +1/-4 only.")
    for name in ("development_per_domain", "evaluation_per_domain"):
        if c[name] % c["batch_size"]:
            raise ValueError(name + " must be divisible by batch_size")
    if c["thinking"]:
        raise ValueError("The pilot is explicitly non-thinking; amend protocol before changing.")
    return c

def load_env():
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "DEEPSEEK_API_KEY":
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

def code_hash():
    return digest({p.name: p.read_text() for p in sorted((ROOT / "auditstudy").glob("*.py"))})

def protocol_fingerprint(c, manifest):
    return digest({"config": c, "manifest_hash": digest(manifest),
                   "protocol_code": {name:(ROOT/"auditstudy"/name).read_text() for name in
                                     ["prompts.py","science.py","analysis.py","runner.py","data.py"]}})
