"""Independent decoder for dsh session logs of the E1-H forced-subagent arm (harness_sub).

Written from scratch for the audit: it does not import the runner. Read-only on the repo.
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import re
from collections import Counter

import zstandard

REPO = "/Users/madili/Desktop/review papaer research"
SESS_ROOT = REPO + "/.tools/dsh-home/sessions"
ALLOWED = [1, 3, 5, 6, 8, 10]

# The three fixed texts of the arm (copied from the runner source so that the audit can
# recognise them in the session logs; they are only ever compared, never sent anywhere).
PRIMER_HEAD = "Do not review yet. This message only delivers the submission"
PROCESS_HEAD = "Now review the submission above, following this required process:"
REMINDER_HEAD = "The required process is not complete."
PRIMER = (
    "\n\nDo not review yet. This message only delivers the submission and the review "
    "criteria. Reply with the single word READY and nothing else; how to proceed is "
    "explained in the next message."
)


# ------------------------------------------------------------------ low level

def session_path(sid: str) -> str | None:
    hits = glob.glob(os.path.join(SESS_ROOT, "*", sid, "session.v4.jsonl.zstd"))
    return hits[0] if hits else None


def decode(path: str) -> list[dict]:
    raw = open(path, "rb").read()
    data = zstandard.ZstdDecompressor().stream_reader(
        io.BytesIO(raw), read_across_frames=True).read().decode("utf-8", "replace")
    out = []
    for line in data.split("\n"):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            out.append({"type": "__unparsed__", "data": line[:200]})
    return out


_CACHE: dict[str, list[dict] | None] = {}


def load(sid: str) -> list[dict] | None:
    if sid not in _CACHE:
        p = session_path(sid)
        _CACHE[sid] = decode(p) if p else None
    return _CACHE[sid]


def blocks(rec: dict) -> list[dict]:
    d = rec.get("data") or {}
    c = d.get("content")
    if c is None:
        c = (d.get("message") or {}).get("content")
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    return [b for b in (c or []) if isinstance(b, dict)]


def text_of(rec: dict) -> str:
    return "".join(b.get("text", "") for b in blocks(rec) if b.get("type") == "text")


def reasoning_of(rec: dict) -> str:
    return "".join(b.get("text", "") for b in blocks(rec) if b.get("type") == "reasoning")


def source_kind(rec: dict) -> str | None:
    d = rec.get("data") or {}
    s = d.get("source") or (d.get("message") or {}).get("source") or {}
    return s.get("kind")


def ws(s: str) -> str:
    return re.sub(r"\s+", " ", s)


# ------------------------------------------------------------------ review parsing

def parse_json_review(text: str) -> dict | None:
    """A dict with the six review fields if `text` holds a JSON review, else None.
    Own implementation: scan for balanced top-level {...} candidates, keep the first
    one that carries all six fields with numeric values."""
    if "{" not in text:
        return None
    cands = []
    s = text.strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", s, re.S)
    if m:
        s = m.group(1).strip()
    try:
        cands.append(json.loads(s))
    except ValueError:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        try:
            cands.append(json.loads(s[i:j + 1]))
        except ValueError:
            pass
    # balanced-brace scan (handles prose + JSON + prose)
    dec = json.JSONDecoder()
    for mm in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[mm.start():])
        except ValueError:
            continue
        cands.append(obj)
        if len(cands) > 40:
            break
    for c in cands:
        if not isinstance(c, dict):
            continue
        try:
            float(c["rating"]); float(c["soundness"]); float(c["presentation"])
            float(c["contribution"]); float(c["confidence"]); c["accept"]
        except (KeyError, TypeError, ValueError):
            continue
        return c
    return None


def runner_style_review(text: str) -> dict | None:
    """What the runner's extract_review + validate_review would accept (re-implemented):
    whole text (fence-stripped) as JSON, else first '{' .. last '}'."""
    s = text.strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", s, re.S)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except ValueError:
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j <= i:
            return None
        try:
            obj = json.loads(s[i:j + 1])
        except ValueError:
            return None
    if not isinstance(obj, dict):
        return None
    try:
        float(obj["rating"]); float(obj["soundness"]); float(obj["presentation"])
        float(obj["contribution"]); float(obj["confidence"]); obj["accept"]
    except (KeyError, TypeError, ValueError):
        return None
    return obj


def snap_rating(x) -> int:
    x = float(x)
    return min(ALLOWED, key=lambda a: (abs(a - x), a))


# ------------------------------------------------------------------ session structure

def split_child(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """(seed records, own records, info). Own cut: the child's `subagent/descriptor`
    record marks the first record the child wrote itself; the `session/end-seed` just
    before it closes the inherited copy."""
    info = {"descriptor_idx": None, "endseed_idx": None, "endseed_inherited": None}
    di = next((i for i, r in enumerate(rows) if r.get("type") == "subagent/descriptor"), None)
    info["descriptor_idx"] = di
    if di is None:
        return [], rows, info
    ei = None
    for i in range(di - 1, -1, -1):
        if rows[i].get("type") == "session/end-seed":
            ei = i
            break
    info["endseed_idx"] = ei
    if ei is None:
        return rows[:di], rows[di:], info
    info["endseed_inherited"] = bool((rows[ei].get("data") or {}).get("inherited"))
    return rows[:ei], rows[ei + 1:], info


def usage_sum(rows: list[dict]) -> dict:
    u = {"input": 0, "cache_read": 0, "output": 0, "calls": 0, "attempts_failed": 0,
         "retries": 0, "cache_write": 0}
    for r in rows:
        t = r.get("type")
        if t == "assistant/message":
            us = (r.get("data") or {}).get("usage") or {}
            u["input"] += us.get("inputTokens", 0) or 0
            u["cache_read"] += us.get("cacheReadTokens", 0) or 0
            u["output"] += us.get("outputTokens", 0) or 0
            u["cache_write"] += us.get("cacheWriteTokens", 0) or 0
            u["calls"] += 1
        elif t == "assistant/attempt":
            u["attempts_failed"] += 1
        elif t == "llm/retry":
            u["retries"] += 1
    return u


def attempt_text(rec: dict) -> dict:
    """Characters streamed in a failed attempt (reasoning / text / tool args)."""
    out = {"reasoning": 0, "text": 0, "tool": 0, "tool_names": []}
    text_parts = []
    for s in (rec.get("data") or {}).get("stream") or []:
        t = s.get("type")
        if t == "reasoning-chunks":
            out["reasoning"] += sum(len(x) for x in s.get("texts") or [])
        elif t == "text-chunks":
            out["text"] += sum(len(x) for x in s.get("texts") or [])
            text_parts.extend(s.get("texts") or [])
        elif t == "tool-call-chunks":
            out["tool"] += sum(len(x) for x in s.get("args") or [])
            if s.get("name"):
                out["tool_names"].append(s["name"])
    out["text_str"] = "".join(text_parts)
    return out


def sha256_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
