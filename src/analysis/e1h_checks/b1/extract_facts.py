"""Stage 1 of the B1 audit: decode every harness_sub session named in the log (parent +
children) and write one facts record per log row.  Read-only on the repo.

    python3 extract_facts.py --logs <sub_log.jsonl> --out facts.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b1lib as L  # noqa: E402


def load_prompt_parts() -> tuple[str, str, str]:
    path = L.REPO + "/src/review/prompts/review_v1.md"
    raw = open(path, encoding="utf-8").read()
    parts = re.split(r"(?m)^#\s+(System|User)\s*$", raw)
    sec = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    return sec["System"], sec["User"], hashlib.sha256(open(path, "rb").read()).hexdigest()


SYSTEM, USER_T, PROMPT_SHA = load_prompt_parts()


def paper_for(row: dict) -> dict:
    safe = row["arxiv_id"].replace("/", "_")
    path = f"{L.REPO}/data/processed/texts/{safe}/v{row['v']}.md"
    out = {"text_path": path, "exists": os.path.exists(path), "sha_matches_row": None,
           "paper": None, "turn1_expected": None, "truncated": False}
    if not out["exists"]:
        return out
    raw = open(path, "rb").read()
    out["sha_matches_row"] = hashlib.sha256(raw).hexdigest() == row.get("text_sha")
    paper = raw.decode("utf-8", errors="replace")
    if len(paper) > 380000:
        paper = paper[:380000] + "\n[truncated]"
        out["truncated"] = True
    out["paper"] = paper
    out["turn1_expected"] = SYSTEM + "\n\n" + USER_T.replace("{PAPER_TEXT}", paper) + L.PRIMER
    return out


def chunk_cov(paper: str, text: str, size: int = 137, offset: int = 53) -> float:
    """Own coverage measure (different chunking from the runner's 200/0)."""
    p, q = L.ws(paper), L.ws(text)
    chunks = [p[i:i + size] for i in range(offset, max(len(p) - size + 1, offset + 1), size)]
    if not chunks:
        return float(p in q)
    return sum(c in q for c in chunks) / len(chunks)


def rec_key(r: dict) -> str:
    return json.dumps([r.get("seq"), r.get("type"), r.get("data")], sort_keys=True,
                      ensure_ascii=False)


def parent_facts(log: list[dict]) -> dict:
    f: dict = {"n_records": len(log)}
    f["turn_ends"] = [((r.get("data") or {}).get("turn"),
                       ((r.get("data") or {}).get("reason") or {}).get("kind"))
                      for r in log if r.get("type") == "turn/end"]
    f["n_turn_starts"] = sum(r.get("type") == "turn/start" for r in log)
    f["children"] = [(r.get("data") or {}).get("childId") for r in log
                     if r.get("type") == "subagent/catalog"]
    results = {}
    for r in log:
        if r.get("type") == "tool/result":
            m = (r.get("data") or {}).get("message") or {}
            results[m.get("toolCallId")] = {
                "text": "".join(b.get("text", "") for b in m.get("content") or []
                                if isinstance(b, dict))[:600],
                "isError": m.get("isError")}
    calls = []
    for i, r in enumerate(log):
        if r.get("type") != "tool/call":
            continue
        d = r.get("data") or {}
        try:
            args = json.loads(d.get("arguments") or "{}")
        except ValueError:
            args = {"__raw__": (d.get("arguments") or "")[:2000]}
        res = results.get(d.get("callId"), {})
        calls.append({"idx": i, "seq": r.get("seq"), "time": r.get("time"),
                      "turn": d.get("turn"), "step": d.get("step"), "name": d.get("name"),
                      "args": args, "result": res.get("text"), "isError": res.get("isError")})
    f["tool_calls"] = calls
    users = []
    for i, r in enumerate(log):
        if r.get("type") != "user/message":
            continue
        t = L.text_of(r)
        d = r.get("data") or {}
        users.append({"idx": i, "seq": r.get("seq"), "time": r.get("time"),
                      "kind": L.source_kind(r),
                      "sender": (d.get("source") or {}).get("senderSessionId"),
                      "len": len(t), "head": t[:160],
                      "is_primer_turn": L.PRIMER_HEAD in t,
                      "is_process": t.startswith(L.PROCESS_HEAD),
                      "is_reminder": t.startswith(L.REMINDER_HEAD),
                      "text": t if L.source_kind(r) in ("agent-message", "subagent-settled")
                      else None})
    f["user_msgs"] = users
    asst = []
    for i, r in enumerate(log):
        if r.get("type") != "assistant/message":
            continue
        d = r.get("data") or {}
        t = L.text_of(r)
        mine = L.parse_json_review(t) if t.strip() else None
        runner = L.runner_style_review(t) if t.strip() else None
        us = d.get("usage") or {}
        asst.append({"idx": i, "seq": r.get("seq"), "time": r.get("time"),
                     "turn": d.get("turn"), "step": d.get("step"),
                     "text_len": len(t), "reasoning_len": len(L.reasoning_of(r)),
                     "tool_names": [b.get("name") for b in L.blocks(r)
                                    if b.get("type") == "tool-call"],
                     "json_review": bool(mine), "runner_review": bool(runner),
                     "rating_raw": (mine or runner or {}).get("rating"),
                     "review": {k: (mine or runner or {}).get(k) for k in
                                ("rating", "soundness", "presentation", "contribution",
                                 "confidence", "accept")} if (mine or runner) else None,
                     "text": t,
                     "text_head": t[:200],
                     "reasoning": L.reasoning_of(r),
                     "model": ((d.get("message") or {}).get("source") or {}).get("model"),
                     "usage": {"input": us.get("inputTokens", 0),
                               "cache_read": us.get("cacheReadTokens", 0),
                               "output": us.get("outputTokens", 0)}})
    f["assistant_msgs"] = asst
    f["attempts"] = []
    for i, r in enumerate(log):
        if r.get("type") == "assistant/attempt":
            a = L.attempt_text(r)
            a["idx"] = i
            a["turn"] = (r.get("data") or {}).get("turn")
            a["json_review_in_text"] = bool(L.parse_json_review(a.pop("text_str")))
            f["attempts"].append(a)
    f["retry_codes"] = [((r.get("data") or {}).get("failure") or {}).get("code")
                        for r in log if r.get("type") == "llm/retry"]
    f["usage"] = L.usage_sum(log)
    f["system_text"] = next((L.text_of(r) for r in log if r.get("type") == "system/message"),
                            None)
    f["runtime_ctx"] = [L.text_of(r) for r in log if r.get("type") == "user/message"
                        and L.source_kind(r) == "runtime-context"]
    f["tools_offered"] = sorted({t.get("name") for r in log if r.get("type") == "request/header"
                                 for t in ((r.get("data") or {}).get("header") or {})
                                 .get("tools") or []})
    cfgs = {json.dumps(((r.get("data") or {}).get("header") or {}).get("config"),
                       sort_keys=True) for r in log if r.get("type") == "request/header"}
    f["request_configs"] = sorted(cfgs)
    return f


def child_facts(cid: str, parent_log: list[dict], paper: dict) -> dict:
    c: dict = {"child_id": cid, "found": False}
    rows = L.load(cid)
    if rows is None:
        return c
    c["found"] = True
    seed, own, info = L.split_child(rows)
    c.update(info)
    c["n_seed"], c["n_own"] = len(seed), len(own)
    # is the seed a verbatim prefix of the parent's log?
    pk = [rec_key(r) for r in parent_log[:len(seed)]]
    sk = [rec_key(r) for r in seed]
    c["seed_is_parent_prefix"] = pk == sk
    c["seed_types"] = [r.get("type") for r in seed]
    su = [r for r in seed if r.get("type") == "user/message"]
    c["seed_user_kinds"] = [L.source_kind(r) for r in su]
    seed_user_text = " ".join(L.text_of(r) for r in su)
    seed_all_text = " ".join(L.text_of(r) + " " + L.reasoning_of(r) for r in seed
                             if r.get("type") in ("user/message", "assistant/message",
                                                  "system/message"))
    c["seed_has_process_text"] = (L.PROCESS_HEAD in seed_all_text
                                  or "subagent_fork` tool exactly once" in seed_all_text)
    c["seed_has_reminder_text"] = L.REMINDER_HEAD in seed_all_text
    c["seed_turns"] = [(r.get("data") or {}).get("turn") for r in seed
                       if r.get("type") == "turn/start"]
    c["seed_assistant_texts"] = [L.text_of(r) for r in seed if r.get("type") == "assistant/message"]
    c["seed_assistant_reasoning_len"] = [len(L.reasoning_of(r)) for r in seed
                                         if r.get("type") == "assistant/message"]
    first_user = next((L.text_of(r) for r in su if L.source_kind(r) == "user"), "")
    c["seed_turn1_len"] = len(first_user)
    if paper.get("paper") is not None:
        p = paper["paper"]
        c["paper_exact_in_seed"] = p in first_user
        c["paper_ws_in_seed"] = L.ws(p) in L.ws(seed_user_text)
        c["turn1_equals_expected"] = first_user == paper["turn1_expected"]
        c["my_coverage"] = round(chunk_cov(p, seed_user_text), 4)
    # own records
    ou = [r for r in own if r.get("type") == "user/message"]
    c["own_user"] = [{"kind": L.source_kind(r), "len": len(L.text_of(r)),
                      "text": L.text_of(r),
                      "blocks": [b.get("text", "") for b in L.blocks(r) if b.get("type") == "text"]}
                     for r in ou]
    own_text = " ".join(L.text_of(r) for r in ou)
    c["own_has_process_text"] = (L.PROCESS_HEAD in own_text
                                 or "subagent_fork` tool exactly once" in own_text)
    calls = []
    for i, r in enumerate(own):
        if r.get("type") != "tool/call":
            continue
        d = r.get("data") or {}
        try:
            args = json.loads(d.get("arguments") or "{}")
        except ValueError:
            args = {"__raw__": d.get("arguments")}
        calls.append({"name": d.get("name"), "args": args, "time": r.get("time"),
                      "seq": r.get("seq")})
    c["tool_calls"] = calls
    c["turn_ends"] = [((r.get("data") or {}).get("reason") or {}).get("kind") for r in own
                      if r.get("type") == "turn/end"]
    c["n_own_turns"] = sum(r.get("type") == "turn/start" for r in own)
    am = [r for r in own if r.get("type") == "assistant/message"]
    c["assistant"] = []
    for r in am:
        t = L.text_of(r)
        rev = L.parse_json_review(t) if t.strip() else None
        us = (r.get("data") or {}).get("usage") or {}
        c["assistant"].append({"time": r.get("time"), "text": t,
                               "prompt_tokens": (us.get("inputTokens", 0) or 0)
                               + (us.get("cacheReadTokens", 0) or 0),
                               "output_tokens": us.get("outputTokens", 0) or 0,
                               "reasoning_len": len(L.reasoning_of(r)),
                               "tool_names": [b.get("name") for b in L.blocks(r)
                                              if b.get("type") == "tool-call"],
                               "json_review": {k: rev.get(k) for k in
                                               ("rating", "soundness", "presentation",
                                                "contribution", "confidence", "accept")}
                               if rev else None})
    c["attempts"] = []
    for r in own:
        if r.get("type") == "assistant/attempt":
            a = L.attempt_text(r)
            a.pop("text_str", None)
            c["attempts"].append(a)
    c["retry_codes"] = [((r.get("data") or {}).get("failure") or {}).get("code")
                        for r in own if r.get("type") == "llm/retry"]
    c["usage"] = L.usage_sum(own)
    c["seed_usage"] = L.usage_sum(seed)
    desc = next(((r.get("data") or {}) for r in rows if r.get("type") == "subagent/descriptor"), {})
    c["descriptor"] = desc
    c["tools_offered"] = sorted({t.get("name") for r in own if r.get("type") == "request/header"
                                 for t in ((r.get("data") or {}).get("header") or {})
                                 .get("tools") or []})
    c["request_configs"] = sorted({json.dumps(((r.get("data") or {}).get("header") or {})
                                              .get("config"), sort_keys=True)
                                   for r in own if r.get("type") == "request/header"})
    c["first_own_time"] = next((r.get("time") for r in own if r.get("time")), None)
    c["last_own_time"] = next((r.get("time") for r in reversed(own) if r.get("time")), None)
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    cfg = ap.parse_args()
    n = 0
    with open(cfg.out, "w", encoding="utf-8") as fh:
        for line_no, line in enumerate(open(cfg.logs, encoding="utf-8")):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("arm") != "harness_sub":
                continue
            rec = {"line_no": line_no, "row": row, "parent": None, "children": [],
                   "paper": None}
            paper = paper_for(row)
            rec["paper"] = {k: paper[k] for k in ("text_path", "exists", "sha_matches_row",
                                                  "truncated")}
            rec["paper"]["paper_chars"] = len(paper["paper"]) if paper["paper"] else None
            rec["paper"]["prompt_sha_matches_row"] = PROMPT_SHA == row.get("prompt_sha")
            sid = row.get("session_id")
            plog = L.load(sid) if sid else None
            rec["parent_found"] = plog is not None
            if plog is not None:
                rec["parent"] = parent_facts(plog)
                first_user = next((L.text_of(r) for r in plog if r.get("type") == "user/message"
                                   and L.source_kind(r) == "user"), "")
                if paper["paper"] is not None:
                    rec["parent"]["turn1_equals_expected"] = first_user == paper["turn1_expected"]
                    rec["parent"]["turn1_len"] = len(first_user)
                    rec["parent"]["turn1_expected_len"] = len(paper["turn1_expected"])
                for cid in rec["parent"]["children"]:
                    rec["children"].append(child_facts(cid, plog, paper))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            L._CACHE.clear()
    print(f"wrote {n} fact records to {cfg.out}")


if __name__ == "__main__":
    main()
