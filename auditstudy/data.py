from __future__ import annotations
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path
from .common import digest, now, read_json, write_json, write_jsonl

def normalize(text):
    return re.sub(r"\s+", " ", text).strip().casefold()

def preserve_option_order(question, options):
    # Conservative: some matches concern variables rather than choices. Keeping
    # their original order is harmless; permuting a referenced choice is not.
    text = "\n".join([question] + options)
    return bool(re.search(r"\b(all|none|both|either)\b.{0,25}\b(above|below)\b|\b(option|choice|answer)\s+\(?[A-J]\b", text, re.I)
                or re.search(r"\b[A-J]\s+(and|or)\s+[A-J]\b|\([A-J]\)\s*(and|or|,)\s*\([A-J]\)", text))

def fixture_rows(c):
    """Artificial smoke-test inputs; never allowed for a live API study."""
    n = sum(c[x] for x in ("calibration_per_domain", "development_per_domain", "evaluation_per_domain"))
    return [
        {"question_id": f"mock-{domain}-{i}", "category": domain,
         "question": f"MOCK PIPELINE FIXTURE {domain} {i}. Which option names the integer {i % 10}?",
         "options": [str(j) for j in range(10)], "answer_index": i % 10}
        for domain in c["domains"] for i in range(n)
    ]

def prepare(c, directory, source="hf", local=None):
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        old = read_json(directory / "manifest.json")
        if old["preparation_config_hash"] != digest(c):
            raise ValueError("Prepared dataset belongs to different config; use a new data directory.")
        print(f"Already prepared: {directory} ({old['kind']}); preserving splits.")
        return old
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {"dataset": c["dataset"], "revision": c["revision"]}
    if source == "mock":
        rows = fixture_rows(c)
        kind = "synthetic_mock"
    elif local:
        path = Path(local)
        if path.suffix == ".jsonl":
            rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq
            rows = pq.read_table(path).to_pylist()
        else:
            raise ValueError("Local source must be JSONL or Parquet.")
        provenance["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance["note"] = "Local import; revision is asserted by operator, not independently authenticated."
        kind = "local_import"
    else:
        import pyarrow.parquet as pq
        url = f"https://huggingface.co/datasets/{c['dataset']}/resolve/{c['revision']}/data/test-00000-of-00001.parquet"
        path = directory / "source.parquet"
        if not path.exists():
            tmp = path.with_suffix(".part")
            print("Downloading pinned MMLU-Pro item bank...", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
                size = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 200_000_000:
                        raise ValueError("Unexpected dataset download size; stopped.")
                    out.write(chunk)
            tmp.replace(path)
        provenance.update(url=url, file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows = pq.read_table(path).to_pylist()
        kind = "mmlu_pro"
    groups, duplicates, malformed = {d: [] for d in c["domains"]}, 0, 0
    seen = set()
    for row in sorted(rows, key=lambda r: str(r.get("question_id", ""))):
        if row.get("category") not in groups:
            continue
        question, options, index = row.get("question"), row.get("options"), row.get("answer_index")
        if not isinstance(question, str) or not isinstance(options, list) or not 2 <= len(options) <= 26 or not all(isinstance(x, str) for x in options) or type(index) is not int or not 0 <= index < len(options):
            malformed += 1
            continue
        key = normalize(question)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        groups[row["category"]].append(row)
    public, private, split_ids, batches, counts = [], [], {}, [], {}
    preserved_order = 0
    total = c["calibration_per_domain"] + c["development_per_domain"] + c["evaluation_per_domain"]
    for domain in c["domains"]:
        pool = sorted(groups[domain], key=lambda x: normalize(x["question"]))
        counts[domain] = len(pool)
        if len(pool) < total:
            raise ValueError(f"{domain}: only {len(pool)} eligible items, need {total}. Amend before running; never shrink silently.")
        rng = random.Random(f"{c['seed']}:{domain}")
        rng.shuffle(pool)
        split_ids[domain] = {}
        offset = 0
        for split, n in [("calibration", c["calibration_per_domain"]), ("development", c["development_per_domain"]), ("evaluation", c["evaluation_per_domain"])]:
            ids = []
            for row in pool[offset:offset+n]:
                item_id = "q_" + digest([domain, normalize(row["question"])])[:16]
                order = list(range(len(row["options"])))
                preserve = preserve_option_order(row["question"],row["options"])
                if preserve:
                    preserved_order += 1
                else:
                    random.Random(f"{c['seed']}:{item_id}:options").shuffle(order)
                options = [row["options"][j] for j in order]
                record = {"id": item_id, "domain": domain, "split": split,
                          "question": row["question"], "options": options}
                public.append(record)
                private.append({"id": item_id, "gold": chr(65 + order.index(row["answer_index"])),
                                "source_question_id": str(row["question_id"]), "option_permutation": order,
                                "option_order_preserved":preserve})
                ids.append(item_id)
            split_ids[domain][split] = ids
            if split != "calibration":
                for j in range(0, len(ids), c["batch_size"]):
                    batches.append({"id": f"{domain}-{split}-{j//c['batch_size']:03d}",
                                    "domain": domain, "split": split, "ids": ids[j:j+c["batch_size"]]})
            offset += n
    repeats = []
    for domain in c["domains"]:
        eligible = [b["id"] for b in batches if b["domain"] == domain and b["split"] == "evaluation"]
        eligible.sort(key=lambda x: digest([c["seed"], "repeat", x]))
        repeats += eligible[:round(c["repeat_fraction"] * len(eligible))]
    write_jsonl(directory / "public_items.jsonl", public)
    write_jsonl(directory / "private_labels.jsonl", private)
    splits = {"domains": split_ids, "batches": batches, "repeat_batches": repeats}
    write_json(directory / "splits.json", splits)
    manifest = {"created": now(), "kind": kind, "provenance": provenance,
                "preparation_config_hash": digest(c), "eligible_counts": counts,
                "duplicate_questions_removed": duplicates, "malformed_rows_removed": malformed,
                "position_reference_items_preserved":preserved_order,
                "selected_count": len(public), "files": {}}
    for name in ("public_items.jsonl", "private_labels.jsonl", "splits.json"):
        manifest["files"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    write_json(directory / "manifest.json", manifest)
    print(f"Prepared {len(public)} items and {len(batches)} batches: {kind}.", flush=True)
    return manifest

def verify(directory):
    directory = Path(directory)
    m = read_json(directory / "manifest.json")
    for filename, expected in m["files"].items():
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Dataset changed after preparation: {filename}")
    return m
