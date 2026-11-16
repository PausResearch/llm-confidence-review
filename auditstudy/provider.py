from __future__ import annotations
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from .common import digest, dumps, now, write_json
from .prompts import validate
from .store import BudgetExceeded

class FatalAPIError(RuntimeError):
    pass

class Client:
    def __init__(self, c, provider, ledger, store, directory):
        self.c, self.provider, self.ledger, self.store = c, provider, ledger, store
        self.directory = Path(directory)
        self.stop = threading.Event()
        self.stop_reason = None
        self.model_lock = threading.Lock()
        self.model_version = None
        prior = self.directory / "served_model.json"
        if prior.exists():
            self.model_version = json.loads(prior.read_text())["model"]
        if provider == "deepseek":
            if c["base_url"].rstrip("/") != "https://api.deepseek.com":
                raise ValueError("This adapter sends keys only to https://api.deepseek.com")
            self.key = os.environ.get("DEEPSEEK_API_KEY")
            if not self.key:
                raise ValueError("DEEPSEEK_API_KEY is missing. Configure .env locally; never put a key in a CLI argument.")

    def payload(self, messages, maximum):
        return {"model":self.c["model"], "messages":messages,
                "temperature":self.c["temperature"], "max_tokens":maximum,
                "response_format":{"type":"json_object"}, "thinking":{"type":"disabled"}, "stream":False}

    def maximum_cost(self, payload):
        # Conservative upper estimate: one token per UTF-8 request byte,
        # plus generous chat overhead. No reliance on a particular tokenizer.
        input_bound = len(dumps(payload).encode("utf-8")) + 1024
        return 1.1 * (input_bound * self.c["input_usd_per_million"] + payload["max_tokens"] * self.c["output_usd_per_million"]) / 1e6

    def _mock(self, task, kind, context):
        def h(s):
            return int(digest(s)[:8],16)
        if kind == "generate":
            result = {"answer":context["letters"][h(task)%len(context["letters"])],
                      "confidence":[.4,.65,.75,.85,.95][h(task+"p")%5]}
        elif kind == "review_R":
            result = {"revised_probabilities":[{"id":i,"p": [.35,.55,.75,.9][h(i+"R")%4]} for i in context["ids"]]}
        elif kind.startswith("review_"):
            result = {"band_accuracy":[(.2+.15*k if kind == "review_M" else [.4,.7,.9][k]) if n else None for k,n in enumerate(context["counts"])]}
        else:
            result = {"action":"AUDIT" if h(task)%2 else "TRUST"}
        return {"id":"mock_"+digest(task)[:10], "model":"MOCK-NO-LLM", "choices":[{"finish_reason":"stop","message":{"content":dumps(result)}}],
                "usage":{"prompt_tokens":0,"completion_tokens":0}}

    def _post(self, payload, task):
        for attempt in range(self.c["transport_attempts"]):
            if self.stop.is_set():
                raise FatalAPIError("Run stopped before request")
            reserve = self.maximum_cost(payload)
            aid = self.ledger.reserve(str(self.directory.resolve()), task, reserve)
            request = urllib.request.Request(self.c["base_url"].rstrip("/")+"/chat/completions",
                data=dumps(payload).encode(), headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json"},method="POST")
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = response.read(5_000_000)
                decoded = json.loads(body)
                usage = decoded.get("usage") or {}
                if all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens","completion_tokens")):
                    cost = (usage["prompt_tokens"] * self.c["input_usd_per_million"] + usage["completion_tokens"] * self.c["output_usd_per_million"]) / 1e6
                    self.ledger.settle(aid,cost,usage)
                    if cost > reserve:
                        self.stop_reason = "Reported token cost exceeded the conservative request reservation; inspect pricing before continuing."
                        self.stop.set()
                        decoded["local_budget_warning"] = self.stop_reason
                else:
                    self.ledger.settle(aid,status="unknown_usage",detail="Reservation retained; usage missing")
                return decoded
            except urllib.error.HTTPError as error:
                status = error.code
                # Keep reservation for uncertain/server errors; known request
                # rejections are conservatively charged too. Never log headers.
                self.ledger.settle(aid,status="http_error",detail=f"HTTP {status}; reservation retained")
                if status not in (408,409,429,500,502,503,504):
                    self.stop.set()
                    raise FatalAPIError(f"DeepSeek HTTP {status}; stopped. Check credentials, balance, model and request settings.") from None
                if attempt+1 == self.c["transport_attempts"]:
                    raise RuntimeError(f"HTTP {status} after bounded retries") from None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
                self.ledger.settle(aid,status="unknown",detail="Network/response failure; reservation retained")
                if attempt+1 == self.c["transport_attempts"]:
                    raise RuntimeError("Network or response failure after bounded retries; possible bill retained") from None
            time.sleep(min(2 ** attempt, 5))
        raise RuntimeError("Unreachable")

    def execute(self, task, kind, messages, context, maximum, retry_uncertain=False):
        phash = digest({"messages":messages,"maximum":maximum,"model":self.c["model"]})
        original = list(messages)
        start_correction = 0
        existing = self.store.get(task)
        if existing:
            if existing["prompt_hash"] != phash:
                raise FatalAPIError("Task prompt changed: use a new study directory; old answers are immutable.")
            if existing["status"] == "done":
                return existing["result"]
            if existing["status"] == "invalid":
                return None
            # Preserve the formatting retry budget and its exact conversation
            # across restarts, rather than drawing a fresh initial response.
            paths = sorted((self.directory / "raw").glob(digest(task)+"-*.json"), reverse=True)
            saved = json.loads(paths[0].read_text()) if paths else {}
            if saved.get("prompt_hash", phash) != phash:
                raise FatalAPIError("Saved request differs from the current prompt")
            if saved.get("validated_result") is not None:
                result = validate(kind,saved["validated_result"],context)
                self.store.put(task,kind,"done",phash,result)
                return result
            if "validation_error" in saved:
                start_correction = saved["format_retry"] + 1
                if start_correction > self.c["format_retries"]:
                    self.store.put(task,kind,"invalid",phash,detail=saved["validation_error"])
                    return None
                content = saved.get("response",{}).get("choices",[{}])[0].get("message",{}).get("content") or ""
                messages = self._correction(original,content,saved["validation_error"])
            elif existing["status"] == "running" and not retry_uncertain:
                raise FatalAPIError("Uncertain interrupted call: inspect raw logs. Use --retry-uncertain only to accept possible duplicate billing.")
            elif saved.get("request"):
                start_correction = saved["format_retry"]
                messages = saved["request"]["messages"]
        if self.stop.is_set():
            return None
        self.store.put(task,kind,"running",phash)
        try:
            for correction in range(start_correction,self.c["format_retries"]+1):
                payload = self.payload(messages, maximum)
                log = {"task":task,"kind":kind,"timestamp":now(),"provider":self.provider,"request":payload,"prompt_hash":phash,"format_retry":correction}
                path = self.directory / "raw" / (digest(task)+f"-{time.time_ns()}.json")
                write_json(path,log)
                reply = self._mock(task,kind,context) if self.provider == "mock" else self._post(payload,task)
                log["response"] = reply
                write_json(path,log)
                version = reply.get("model")
                if not version:
                    self.stop.set()
                    raise FatalAPIError("Missing served model identity; cannot verify a consistent study.")
                with self.model_lock:
                    if self.model_version and self.model_version != version:
                        self.stop.set()
                        raise FatalAPIError("Served model identity changed mid-study; stopped to avoid mixing versions.")
                    if self.model_version is None:
                        self.model_version = version
                        write_json(self.directory / "served_model.json",{"model":version,"first_seen":now(),"note":"Provider identifier; may be an alias rather than an immutable weight snapshot."})
                content = ""
                try:
                    choice = reply["choices"][0]
                    content = choice["message"]["content"] or ""
                    if choice.get("finish_reason") != "stop":
                        raise ValueError("Response was truncated or not normally completed")
                    if choice["message"].get("reasoning_content"):
                        self.stop.set()
                        raise FatalAPIError("Unexpected thinking output in non-thinking study")
                    value = validate(kind,json.loads(content),context)
                    log["validated_result"] = value
                    write_json(path,log)
                    self.store.put(task,kind,"done",phash,value)
                    return value
                except (ValueError, KeyError, IndexError, TypeError) as error:
                    detail = str(error)[:240]
                    log["validation_error"] = detail
                    write_json(path,log)
                    if correction == self.c["format_retries"]:
                        self.store.put(task,kind,"invalid",phash,detail=detail)
                        return None
                    messages = self._correction(original,content,detail)
        except BudgetExceeded:
            self.stop.set()
            self.store.put(task,kind,"error",phash,detail="Budget limit; no new request sent at limit")
            raise
        except FatalAPIError as error:
            self.stop.set()
            # Preserve running on ambiguous interrupted calls, otherwise known
            # provider/configuration errors may be retried after the issue is fixed.
            self.store.put(task,kind,"error",phash,detail=str(error))
            raise
        except Exception as error:
            self.store.put(task,kind,"error",phash,detail=type(error).__name__+": "+str(error)[:200])
            raise

    @staticmethod
    def _correction(original,content,detail):
        return original + [{"role":"assistant","content":content},
            {"role":"user","content":"Your reply did not satisfy the requested JSON schema: "+detail+". Return the requested JSON only. Preserve substantive estimates where present. No correctness feedback is being supplied."}]
