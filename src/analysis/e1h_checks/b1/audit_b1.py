"""Independent audit of the E1-H forced-subagent arm (harness_sub, "B1") from the raw dsh
session logs.  Read-only on the repo; does not import the runner; no network.

    python3 audit_b1.py --logs <e1h_sub_reviews_log.jsonl> [--main-log <e1h_reviews_log.jsonl>]
                        [--out-dir DIR] [--reuse-facts] [--n-sample 25] [--seed 20261112]

Stage 1 (extract_facts.py) decodes every session named in the log plus its children and
writes DIR/facts.jsonl; stage 2 (this file) derives verdicts and statistics, prints a
markdown report and writes DIR/b1_runs.csv and DIR/b1_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics as st
import subprocess
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b1lib as L          # noqa: E402
import promptx as PX       # noqa: E402
import ratingx as RX       # noqa: E402

FORK_TOOLS = ("subagent_fork", "subagent")
BLURB_RX = re.compile(
    r'^Your parent agent id is "(session-[0-9a-f-]+)"\. Before you finish, send your result to '
    r'that agent with send_message\(\{ agent_id: "\1", message: "<self-contained result>" \}\)\. '
    r'The parent shares your workspace but does not automatically receive your transcript, tool '
    r'output, or reasoning\. Send earlier messages as well when a finding changes what the parent '
    r'should do next; sending a message does not end your turn\.$')


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def shingles(text: str, n: int = 6) -> set:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


# ------------------------------------------------------------------ per-run verdict

def derive(rec: dict) -> dict:
    row, p = rec["row"], rec["parent"]
    d: dict = {
        "key": row["key"], "arxiv_id": row["arxiv_id"], "v": row["v"], "rep": row["rep"],
        "session_id": row.get("session_id"), "line_no": rec["line_no"],
        "log_process_ok": row.get("process_ok"), "log_reminded": bool(row.get("reminded")),
        "log_rbr": row.get("review_before_report"), "log_n_children": row.get("n_children"),
        "log_coverage": row.get("coverage"), "log_rating": row.get("rating"),
        "log_parsed_ok": row.get("parsed_ok"), "log_n_turns": row.get("n_turns"),
        "log_n_review_outputs": row.get("n_review_outputs"),
        "log_tool_calls": row.get("tool_calls") or {}, "log_report_via": row.get("report_via"),
        "log_error": (row.get("error") or "")[:160],
        "session_found": bool(rec["parent_found"]),
        "paper_sha_ok": rec["paper"]["sha_matches_row"],
        "log_fields": {k: row.get(k) for k in ("soundness", "presentation", "contribution",
                                               "confidence", "accept")},
    }
    d.update(my_rating=None, my_rating_raw=None, my_review=None, final_time=None, final_text="",
             report_time=None, report_text="", report_chars=0, post_report_ratings=[],
             post_report_ratings_differ=False, child_rating=None, child_rating_lo=None,
             child_rating_hi=None, child_rating_method=None, child_rating_snippet=None,
             child_closing_json_rating=None, report_is_json_review=False,
             child_usage={"input": 0, "cache_read": 0, "output": 0, "calls": 0,
                          "attempts_failed": 0, "retries": 0}, child_reasoning_chars=0,
             final_reasoning_chars=0, final_output_tokens=0, seed_user_kinds=[], seed_turns=[],
             seed_has_process_text=False, seed_has_parent_review=False, child_own_user_kinds=[],
             turn1_reasoning_chars=0, other_tools={}, n_fork_calls=0, n_children=0)
    if not rec["parent_found"]:
        d["my_process_ok"] = None
        return d
    calls = p["tool_calls"]
    forks = [t for t in calls if t["name"] in FORK_TOOLS]
    d["n_fork_calls"] = len(forks)
    d["n_fork_calls_ok"] = sum(not t["isError"] for t in forks)
    d["fork_tool"] = forks[0]["name"] if forks else None
    d["other_tools"] = dict(Counter(t["name"] for t in calls if t["name"] not in FORK_TOOLS))
    d["my_tool_calls"] = dict(Counter(t["name"] for t in calls))
    d["n_children"] = len(p["children"])
    d["n_turns"] = p["n_turn_starts"]
    d["turn_end_kinds"] = [k for _, k in p["turn_ends"]]
    d["reminded"] = any(u["is_reminder"] for u in p["user_msgs"])
    d["turn1_equals_expected"] = p.get("turn1_equals_expected")
    d["turn1_reply_is_READY"] = None
    t1 = [a for a in p["assistant_msgs"] if a["turn"] == 1]
    if t1:
        d["turn1_reply_is_READY"] = t1[-1]["text"].strip() == "READY"
        d["turn1_reasoning_chars"] = sum(a["reasoning_len"] for a in t1)
    child = rec["children"][0] if rec["children"] else None
    cid = child["child_id"] if child else None
    d["child_id"] = cid
    fork_idx = forks[0]["idx"] if forks else None
    d["fork_prompt"] = (forks[0]["args"].get("prompt") or "") if forks else None
    d["fork_description"] = (forks[0]["args"].get("description") or "") if forks else None

    # --- child
    d["child_found"] = bool(child and child["found"])
    if d["child_found"]:
        d["child_paper_exact"] = bool(child.get("paper_exact_in_seed"))
        d["child_turn1_equals_expected"] = bool(child.get("turn1_equals_expected"))
        d["my_coverage"] = child.get("my_coverage")
        d["seed_is_parent_prefix"] = child["seed_is_parent_prefix"]
        d["seed_user_kinds"] = child["seed_user_kinds"]
        d["seed_turns"] = child["seed_turns"]
        d["seed_has_process_text"] = child["seed_has_process_text"] or child["own_has_process_text"]
        d["seed_has_parent_review"] = any(L.parse_json_review(t) is not None
                                          for t in child["seed_assistant_texts"])
        d["child_done"] = "completed" in child["turn_ends"]
        d["child_turn_end"] = child["turn_ends"]
        d["child_tools"] = dict(Counter(t["name"] for t in child["tool_calls"]))
        d["child_own_user_kinds"] = [u["kind"] for u in child["own_user"]]
        first = next((u for u in child["own_user"] if u["kind"] == "user"), None)
        d["child_prompt_blocks"] = len(first["blocks"]) if first else 0
        d["child_prompt_equals_fork_arg"] = bool(first and first["blocks"]
                                                 and first["blocks"][0] == d["fork_prompt"])
        d["child_blurb_standard"] = bool(first and len(first["blocks"]) == 2
                                         and BLURB_RX.match(first["blocks"][1]))
        d["child_extra_parent_msgs"] = [u["text"] for u in child["own_user"]
                                        if u["kind"] == "agent-message"]
        d["child_model"] = child["descriptor"].get("agentModel")
        d["child_effort"] = child["descriptor"].get("agentReasoningEffort")
        sends = [t for t in child["tool_calls"] if t["name"] == "send_message"]
        d["child_n_send"] = len(sends)
        d["child_send_to_parent"] = all(t["args"].get("agent_id") == d["session_id"] for t in sends)
        d["child_closing_text"] = child["assistant"][-1]["text"] if child["assistant"] else ""
        cj = [a["json_review"] for a in child["assistant"] if a["json_review"]]
        d["child_closing_json_rating"] = cj[-1]["rating"] if cj else None
        d["child_usage"] = child["usage"]
        d["child_seed_usage_calls"] = child["seed_usage"]["calls"]
        d["child_first_prompt_tokens"] = (child["assistant"][0].get("prompt_tokens")
                                          if child["assistant"] else None)
        d["child_reasoning_chars"] = sum(x["reasoning_len"] for x in child["assistant"])
    else:
        sends = []

    # --- report delivery
    relays = [u for u in p["user_msgs"] if u["kind"] == "agent-message" and cid
              and u.get("sender") == cid and (fork_idx is None or u["idx"] > fork_idx)]
    closings = [u for u in p["user_msgs"] if u["kind"] == "subagent-settled" and cid
                and f"Background subagent {cid} finished" in (u.get("text") or "")
                and "Its closing message:" in (u.get("text") or "")]
    rep = None
    if relays:
        rep, d["report_via"] = relays[0], "send_message"
    elif closings:
        rep, d["report_via"] = closings[0], "closing_message"
    else:
        d["report_via"] = None
    d["report_idx"] = rep["idx"] if rep else None
    d["report_delivered"] = rep is not None
    d["n_reports_relayed"] = len(relays)
    prefix = f"Agent {cid} sent a message: "
    d["relay_matches_child_send"] = None
    if relays and sends:
        body = relays[0]["text"][len(prefix):] if relays[0]["text"].startswith(prefix) else None
        d["relay_matches_child_send"] = body == sends[0]["args"].get("message")
    settled = [u for u in p["user_msgs"] if u["kind"] == "subagent-settled"]
    d["settled_notice"] = (settled[0]["text"] or "")[:120] if settled else None

    # --- reviews written by the parent
    revs = [a for a in p["assistant_msgs"] if a["json_review"] or a["runner_review"]]
    d["n_review_outputs"] = len(revs)
    d["n_review_outputs_runner_style"] = sum(a["runner_review"] for a in p["assistant_msgs"])
    before = [a for a in revs if d["report_idx"] is None or a["idx"] < d["report_idx"]]
    after = [a for a in revs if d["report_idx"] is not None and a["idx"] > d["report_idx"]]
    d["review_before_report"] = bool(before)
    d["review_before_report_turns"] = [a["turn"] for a in before]
    nonrev_before = [a for a in p["assistant_msgs"]
                     if not (a["json_review"] or a["runner_review"])
                     and (d["report_idx"] is None or a["idx"] < d["report_idx"])
                     and a["turn"] != 1]
    d["max_prose_before_report_chars"] = max([a["text_len"] for a in nonrev_before] or [0])
    d["prose_before_report_mentions_rating"] = any(
        re.search(r"(?i)\brating\b|\bweakness|\bstrengths?\b", a["text"] or "") and a["text_len"] > 250
        for a in nonrev_before)
    d["reasoning_before_fork_chars"] = sum(a["reasoning_len"] for a in p["assistant_msgs"]
                                           if fork_idx is not None and a["idx"] < fork_idx)
    d["final_exists"] = bool(after)
    if t1:
        d["turn1_prompt_tokens"] = t1[-1]["usage"]["input"] + t1[-1]["usage"]["cache_read"]
    if after:
        fr = after[0]
        d["final_reasoning_chars"] = fr["reasoning_len"]
        d["final_output_tokens"] = fr["usage"]["output"]
        d["final_prompt_tokens"] = fr["usage"]["input"] + fr["usage"]["cache_read"]
        d["my_rating"] = L.snap_rating(fr["rating_raw"])
        d["my_rating_raw"] = fr["rating_raw"]
        d["my_review"] = fr["review"]
        d["final_turn"] = fr["turn"]
        d["final_idx"] = fr["idx"]
        d["final_time"] = fr["time"]
        d["final_text"] = fr["text"]
        d["post_report_ratings"] = [L.snap_rating(a["rating_raw"]) for a in after]
        d["post_report_ratings_differ"] = len(set(d["post_report_ratings"])) > 1
    # malformed JSON review after the report (the reminder case)
    d["malformed_review_after_report"] = any(
        (not (a["json_review"] or a["runner_review"])) and a["text"].lstrip().startswith("{")
        and '"rating"' in a["text"] and d["report_idx"] is not None and a["idx"] > d["report_idx"]
        for a in p["assistant_msgs"])

    # --- report text and ratings
    if rep is not None:
        end = d.get("final_idx", 10 ** 9)
        bodies = []
        for u in (relays if relays else [rep]):
            if u["idx"] < end:
                t = u["text"]
                bodies.append(t[len(prefix):] if t.startswith(prefix) else t)
        d["report_text"] = "\n\n".join(bodies)
        d["n_reports_before_final"] = len(bodies)
        cr = RX.child_rating(d["report_text"])
        d["child_rating"] = cr["rating"]
        d["child_rating_method"] = cr["method"]
        d["child_rating_lo"], d["child_rating_hi"] = cr["lo"], cr["hi"]
        d["child_rating_snippet"] = cr["snippet"]
        d["report_is_json_review"] = cr["method"] == "json_object"
        d["report_time"] = rep["time"]
        d["report_chars"] = len(d["report_text"])
    d["fork_time"] = forks[0]["time"] if forks else None

    ok_fork = d["n_fork_calls_ok"] >= 1 and d["n_children"] >= 1
    d["my_process_ok"] = bool(ok_fork and d.get("child_paper_exact") and d.get("child_done")
                              and d["report_delivered"] and d["final_exists"])
    d["strict_violations"] = []
    if d["n_fork_calls"] != 1:
        d["strict_violations"].append(f"fork_calls={d['n_fork_calls']}")
    if d["other_tools"]:
        d["strict_violations"].append("other_tools=" + json.dumps(d["other_tools"], sort_keys=True))
    if d["review_before_report"]:
        d["strict_violations"].append("review_before_report")
    if d["reminded"]:
        d["strict_violations"].append("reminded")
    if d.get("child_extra_parent_msgs"):
        d["strict_violations"].append("parent_messaged_child")

    # --- usage
    d["parent_usage"] = p["usage"]
    d["log_usage"] = row.get("usage") or {}
    d["log_usage_main"] = row.get("usage_main") or {}
    d["log_llm_retries"] = row.get("llm_retries")
    d["parent_attempts_failed"] = p["usage"]["attempts_failed"]
    d["attempt_stream_chars"] = (sum(a["reasoning"] + a["text"] + a["tool"] for a in p["attempts"])
                                 + sum(a["reasoning"] + a["text"] + a["tool"]
                                       for c in rec["children"] if c.get("found")
                                       for a in c["attempts"]))
    d["cwd"] = (re.search(r'Current working directory: "([^"]+)"', (p["runtime_ctx"] or [""])[0])
                or [None, None])[1]
    times = [u["time"] for u in p["user_msgs"] if u["time"]] + \
            [a["time"] for a in p["assistant_msgs"] if a["time"]]
    d["wall_s"] = (max(times) - min(times)) / 1000 if times else None
    d["log_latency_s"] = row.get("latency_s")
    return d


def compare(d: dict) -> list[str]:
    """Disagreements between my reading of the session and the logged verdict."""
    out = []
    if not d["session_found"]:
        return out
    lp = d["log_process_ok"]
    if bool(lp) != bool(d["my_process_ok"]):
        out.append(f"process_ok logged {lp} vs mine {d['my_process_ok']}")
    if d["log_reminded"] != d["reminded"]:
        out.append(f"reminded logged {d['log_reminded']} vs mine {d['reminded']}")
    if d["log_rbr"] is not None and bool(d["log_rbr"]) != d["review_before_report"]:
        out.append(f"review_before_report logged {d['log_rbr']} vs mine {d['review_before_report']}")
    if d["log_n_children"] is not None and d["log_n_children"] != d["n_children"]:
        out.append(f"n_children logged {d['log_n_children']} vs mine {d['n_children']}")
    if d["log_coverage"] is not None:
        mine = 1.0 if d.get("child_paper_exact") else (d.get("my_coverage") or 0.0)
        if abs(float(d["log_coverage"]) - mine) > 1e-9:
            out.append(f"coverage logged {d['log_coverage']} vs mine {mine}")
    if d["log_n_turns"] is not None and d["log_n_turns"] != d["n_turns"]:
        out.append(f"n_turns logged {d['log_n_turns']} vs mine {d['n_turns']}")
    if d["log_tool_calls"] != d["my_tool_calls"] and d["log_process_ok"] is not None:
        out.append(f"tool_calls logged {d['log_tool_calls']} vs mine {d['my_tool_calls']}")
    if lp and d["final_exists"] and d["log_rating"] != d["my_rating"]:
        out.append(f"rating logged {d['log_rating']} vs mine {d['my_rating']}")
    if lp and d.get("my_review"):
        for k, hi in (("soundness", 4), ("presentation", 4), ("contribution", 4), ("confidence", 5)):
            mine = min(hi, max(1, int(round(float(d["my_review"][k])))))
            if d["log_fields"].get(k) != mine:
                out.append(f"{k} logged {d['log_fields'].get(k)} vs mine {mine}")
        acc = d["my_review"]["accept"]
        acc = acc.strip().lower() in ("true", "yes", "1") if isinstance(acc, str) else bool(acc)
        if d["log_fields"].get("accept") != acc:
            out.append(f"accept logged {d['log_fields'].get('accept')} vs mine {acc}")
    if d["log_report_via"] != d["report_via"] and d["log_process_ok"] is not None:
        out.append(f"report_via logged {d['log_report_via']} vs mine {d['report_via']}")
    if d["log_n_review_outputs"] is not None and \
            d["log_n_review_outputs"] != d["n_review_outputs_runner_style"]:
        out.append(f"n_review_outputs logged {d['log_n_review_outputs']} vs mine "
                   f"{d['n_review_outputs_runner_style']}")
    if d["log_usage_main"]:
        pu = d["parent_usage"]
        if any(d["log_usage_main"].get(k) != pu[k] for k in ("input", "cache_read", "output")):
            out.append("usage_main differs from my parent sum")
        cu = d.get("child_usage") or {"input": 0, "cache_read": 0, "output": 0}
        if any(d["log_usage"].get(k) != pu[k] + cu[k] for k in ("input", "cache_read", "output")):
            out.append("usage differs from my parent+child sum")
    return out


def fmt_row(d: dict) -> str:
    return (f"| {d['key']} | {d['session_id']} | {d['log_process_ok']}/{d['log_reminded']}/"
            f"{d['log_rbr']} | {d.get('my_process_ok')}/{d.get('reminded')}/"
            f"{d.get('review_before_report')} | {d.get('n_fork_calls')} | {d.get('n_children')} | "
            f"{d.get('child_paper_exact')} | {d.get('child_done')} | {d.get('report_delivered')} | "
            f"{d.get('final_exists')} | {json.dumps(d.get('other_tools') or {})} | "
            f"{'; '.join(d.get('disagreements') or []) or 'none'} |")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="harness_sub log (jsonl)")
    ap.add_argument("--main-log", default=None,
                    help="optional: log of arms direct/harness, for the reference comparison")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--reuse-facts", action="store_true")
    ap.add_argument("--n-sample", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261112)
    cfg = ap.parse_args()

    facts_path = os.path.join(cfg.out_dir, "facts.jsonl")
    if not (cfg.reuse_facts and os.path.exists(facts_path)):
        subprocess.run([sys.executable, os.path.join(HERE, "extract_facts.py"),
                        "--logs", cfg.logs, "--out", facts_path], check=True)
    recs = [json.loads(line) for line in open(facts_path, encoding="utf-8")]
    D = [derive(r) for r in recs]
    for d in D:
        d["disagreements"] = compare(d)
    S: dict = {}
    P = print

    # ---------------- overview
    n = len(D)
    ok = [d for d in D if d["log_process_ok"] is True]
    clean = [d for d in ok if not d["log_reminded"] and not d["log_rbr"]]
    tech = [d for d in D if d["log_process_ok"] is None]
    pfail = [d for d in D if d["log_process_ok"] is False]
    P(f"# B1 (harness_sub) session audit\n")
    P(f"log: {cfg.logs}")
    P(f"rows {n} | distinct keys {len({d['key'] for d in D})} | process_ok true {len(ok)} "
      f"(clean {len(clean)}) | process failures {len(pfail)} | technical failures {len(tech)} "
      f"(with a session log {sum(d['session_found'] for d in tech)})")
    P(f"sessions decoded: {sum(d['session_found'] for d in D)} parents, "
      f"{sum(bool(d.get('child_found')) for d in D)} children | paper text sha matches row: "
      f"{sum(bool(d['paper_sha_ok']) for d in D)}/{n}")
    S["rows"] = n
    S["process_ok"] = len(ok)
    S["clean"] = len(clean)

    # ---------------- Part A
    P("\n## A. Independent process check\n")
    flagged = [d for d in D if d["log_reminded"] or d["log_rbr"]]
    others = sorted((d for d in ok if d not in flagged), key=lambda d: d["key"])
    rng = random.Random(cfg.seed)
    sample = rng.sample(others, min(cfg.n_sample, len(others)))
    techs = sorted((d for d in tech if d["session_found"]), key=lambda d: d["key"])
    tech_s = rng.sample(techs, min(5, len(techs)))
    P(f"designated sample: {len(flagged)} flagged (reminded or review_before_report) + "
      f"{len(sample)} random other process_ok rows (seed {cfg.seed}) + {len(tech_s)} technical "
      f"failures\n")
    P("| key | session | logged ok/reminded/rbr | mine ok/reminded/rbr | fork calls | children "
      "| paper in child | child done | report delivered | final review after report | other tools "
      "| disagreements |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in flagged + sample + tech_s:
        P(fmt_row(d))
    samp_dis = [d for d in flagged + sample + tech_s if d["disagreements"]]
    P(f"\ndisagreements in the designated sample: {len(samp_dis)} of "
      f"{len(flagged) + len(sample) + len(tech_s)}")

    withs = [d for d in D if d["session_found"]]
    dis = [d for d in withs if d["disagreements"]]
    P(f"\n### all {len(withs)} rows that have a session log\n")
    P(f"rows where my verdict differs from any logged field: {len(dis)}")
    for d in dis:
        P(f"- {d['key']} {d['session_id']}: {'; '.join(d['disagreements'])}")
    S["rows_with_session"] = len(withs)
    S["verdict_disagreements"] = [{"key": d["key"], "session_id": d["session_id"],
                                   "what": d["disagreements"]} for d in dis]

    okd_logged = [d for d in withs if d["log_process_ok"] is True]
    okd = [d for d in okd_logged if d["my_process_ok"]]
    if len(okd) != len(okd_logged):
        P(f"\nNOTE: {len(okd_logged) - len(okd)} rows logged process_ok but not confirmed by this "
          f"audit are left out of the statistics below: "
          f"{[(d['key'], d['session_id']) for d in okd_logged if not d['my_process_ok']]}")
    P(f"\nchecks over the {len(okd)} process_ok rows:")
    checks = [
        ("turn-1 message is exactly rules + paper file + primer", "turn1_equals_expected"),
        ("child seed is a verbatim prefix of the parent log", "seed_is_parent_prefix"),
        ("child seed holds the full paper text (exact substring)", "child_paper_exact"),
        ("child turn ended 'completed'", "child_done"),
        ("report relayed to the parent", "report_delivered"),
        ("relayed text is exactly the child's send_message text", "relay_matches_child_send"),
        ("a valid review JSON exists after the report", "final_exists"),
        ("child's prompt block 1 equals the fork tool argument", "child_prompt_equals_fork_arg"),
        ("child's prompt block 2 is the standard dsh parent-id notice", "child_blurb_standard"),
        ("turn-1 reply is exactly READY", "turn1_reply_is_READY"),
    ]
    for label, k in checks:
        c = Counter(d.get(k) for d in okd)
        P(f"- {label}: {c.get(True, 0)}/{len(okd)}" +
          ("" if c.get(True, 0) == len(okd) else f"  (others: {dict((str(a), b) for a, b in c.items() if a is not True)})"))
    P(f"- fork calls per run: {dict(Counter(d['n_fork_calls'] for d in okd))}; children per run: "
      f"{dict(Counter(d['n_children'] for d in okd))}; fork tool: {dict(Counter(d['fork_tool'] for d in okd))}")
    P(f"- child model/effort: {dict(Counter((d.get('child_model'), d.get('child_effort')) for d in okd))}")
    P(f"- report route: {dict(Counter(d['report_via'] for d in okd))}; reports relayed per run: "
      f"{dict(Counter(d['n_reports_relayed'] for d in okd))}")
    P(f"- child tool calls: {dict(Counter(json.dumps(d.get('child_tools'), sort_keys=True) for d in okd))}")
    ot = Counter()
    for d in okd:
        for k in d["other_tools"]:
            ot[k] += 1
    n_ot = sum(bool(d["other_tools"]) for d in okd)
    P(f"- runs where the main agent called a tool other than subagent_fork: {n_ot}/{len(okd)} "
      f"({dict(ot)}); among clean rows: {sum(bool(d['other_tools']) for d in clean)}/{len(clean)}")
    P(f"- runs with more than one subagent_fork call: "
      f"{[(d['key'], d['session_id'], d['n_fork_calls'], d['n_fork_calls_ok']) for d in okd if d['n_fork_calls'] != 1]}")
    P(f"- runs where the parent messaged the child: "
      f"{[(d['key'], d['child_extra_parent_msgs']) for d in okd if d.get('child_extra_parent_msgs')]}")
    P(f"- reviews written before the report (JSON): "
      f"{[(d['key'], d['session_id'], 'turns', d['review_before_report_turns']) for d in okd if d['review_before_report']]}")
    P(f"- longest non-JSON parent text between the process message and the report: max "
      f"{max(d['max_prose_before_report_chars'] for d in okd)} chars; runs with review-like prose "
      f"there: {sum(d['prose_before_report_mentions_rating'] for d in okd)}")
    P(f"- reminded runs: {[(d['key'], d['session_id'], 'malformed JSON first' if d['malformed_review_after_report'] else '') for d in okd if d['reminded']]}")
    P(f"- post-report review outputs per run: {dict(Counter(len(d['post_report_ratings']) for d in okd))}; "
      f"runs whose repeated outputs carry different ratings: "
      f"{[(d['key'], d['post_report_ratings']) for d in okd if d['post_report_ratings_differ']]}")
    P(f"- raw rating not on the scale (coerced): "
      f"{[(d['key'], d['my_rating_raw']) for d in okd if float(d['my_rating_raw']) not in L.ALLOWED]}")
    order_bad = [d["key"] for d in okd
                 if None in (d["fork_time"], d["report_time"], d["final_time"])
                 or not (d["fork_time"] <= d["report_time"] <= d["final_time"])]
    P(f"- time order fork <= report <= final review violated: {order_bad}")
    P(f"- working directory shown to the model: {dict(Counter(d['cwd'] for d in withs))}")
    slow = [(d["key"], round(d["wall_s"]), d["log_latency_s"]) for d in okd
            if d["wall_s"] and d["log_latency_s"] and d["wall_s"] - d["log_latency_s"] > 120]
    P(f"- runs whose session wall time exceeds the logged latency_s by > 120 s: {slow}")
    S["other_tool_runs"] = n_ot
    S["other_tools"] = dict(ot)

    P(f"\ntechnical failures with a session log ({len(techs)}):")
    for d in techs:
        P(f"- {d['key']} {d['session_id']}: turns {d['turn_end_kinds']}, fork calls "
          f"{d['n_fork_calls']}, child end {d.get('child_turn_end')}, report delivered "
          f"{d['report_delivered']}, final review {d['final_exists']}, settled: {d['settled_notice']}")
    P(f"technical failures that nevertheless hold a complete process: "
      f"{[d['key'] for d in techs if d['my_process_ok']]}")
    keyc = Counter(d["key"] for d in D)
    P(f"keys with more than one row: "
      f"{[(k, [(x['log_process_ok'], x['log_rating']) for x in D if x['key'] == k]) for k, c in keyc.items() if c > 1]}")

    # ---------------- Part B1
    P("\n## B1. Child report rating vs parent final rating\n")

    def rating_block(rows: list[dict], label: str) -> dict:
        res: dict = {"n": len(rows)}
        pt = [d for d in rows if d.get("child_rating") is not None]
        rg = [d for d in rows if d.get("child_rating") is None and d.get("child_rating_lo") is not None]
        no = [d for d in rows if d.get("child_rating") is None and d.get("child_rating_lo") is None]
        P(f"### {label} (n = {len(rows)})")
        P(f"- child report states a point rating: {len(pt)} ({len(pt) / len(rows):.1%}); a range: "
          f"{len(rg)}; no numeric rating: {len(no)} ({len(no) / len(rows):.1%})")
        P(f"  - how it was stated: {dict(Counter(d.get('child_rating_method') for d in rows))}")
        if pt:
            diffs = [d["my_rating"] - L.snap_rating(d["child_rating"]) for d in pt]
            same = sum(x == 0 for x in diffs)
            up = sum(x > 0 for x in diffs)
            dn = sum(x < 0 for x in diffs)
            P(f"- identical rating (parent final == child report): {same}/{len(pt)} = {same / len(pt):.1%}")
            P(f"- mean signed difference parent - child: {mean(diffs):+.3f} rating points "
              f"(SD {st.pstdev(diffs):.3f}); parent higher {up}, parent lower {dn}")
            P(f"  - transitions child -> parent when changed: "
              f"{dict(Counter((L.snap_rating(d['child_rating']), d['my_rating']) for d in pt if d['my_rating'] != L.snap_rating(d['child_rating'])))}")
            P(f"- mean child-report rating {mean(L.snap_rating(d['child_rating']) for d in pt):.3f} vs "
              f"mean parent rating on the same runs {mean(d['my_rating'] for d in pt):.3f}")
            raw_off = [d["key"] for d in pt if float(d["child_rating"]) not in L.ALLOWED]
            P(f"  - child ratings off the allowed scale (snapped for the comparison): {raw_off}")
            res.update(n_point=len(pt), identical=same, agreement=same / len(pt),
                       mean_diff=mean(diffs), parent_higher=up, parent_lower=dn)
        if rg:
            inside = sum(d["child_rating_lo"] <= d["my_rating"] <= d["child_rating_hi"] for d in rg)
            P(f"- child gave a range: parent rating inside it in {inside}/{len(rg)}: "
              f"{[(d['key'], (d['child_rating_lo'], d['child_rating_hi']), d['my_rating']) for d in rg]}")
        if no:
            P(f"- runs whose report has no numeric rating: parent rating distribution "
              f"{dict(sorted(Counter(d['my_rating'] for d in no).items()))}, mean "
              f"{mean(d['my_rating'] for d in no):.3f}")
        P(f"- mean parent rating, all {len(rows)} runs: {mean(d['my_rating'] for d in rows):.3f}; "
          f"distribution {dict(sorted(Counter(d['my_rating'] for d in rows).items()))}")
        cj = [d for d in rows if d.get("child_closing_json_rating") is not None]
        if cj:
            s2 = sum(L.snap_rating(d["child_closing_json_rating"]) == d["my_rating"] for d in cj)
            P(f"- child's own closing message (not shown to the parent) holds a JSON review in "
              f"{len(cj)} runs; its rating equals the parent's final rating in {s2}/{len(cj)} = "
              f"{s2 / len(cj):.1%}; mean parent - child-closing "
              f"{mean(d['my_rating'] - L.snap_rating(d['child_closing_json_rating']) for d in cj):+.3f}")
            both = [d for d in cj if d.get("child_rating") is not None]
            P(f"  - where both exist ({len(both)}), report rating == closing-JSON rating in "
              f"{sum(L.snap_rating(d['child_rating']) == L.snap_rating(d['child_closing_json_rating']) for d in both)}")
        anyc = [d for d in rows if d.get("child_rating") is not None
                or d.get("child_closing_json_rating") is not None]
        if anyc:
            def cr(d):
                return L.snap_rating(d["child_rating"] if d.get("child_rating") is not None
                                     else d["child_closing_json_rating"])
            s3 = sum(cr(d) == d["my_rating"] for d in anyc)
            P(f"- using report rating, else the child's closing JSON: {len(anyc)} runs, identical "
              f"{s3} = {s3 / len(anyc):.1%}, mean parent - child "
              f"{mean(d['my_rating'] - cr(d) for d in anyc):+.3f}")
            res.update(n_any=len(anyc), agreement_any=s3 / len(anyc),
                       mean_diff_any=mean(d["my_rating"] - cr(d) for d in anyc))
        jr = [d for d in rows if d.get("report_is_json_review")]
        if jr:
            P(f"- report is itself a full review JSON in {len(jr)} runs ({len(jr) / len(rows):.1%})")
        ov = []
        for d in rows:
            a, b = shingles(d["final_text"]), shingles(d["report_text"])
            if a:
                ov.append(len(a & b) / len(a))
        if ov:
            P(f"- share of the parent's final-review 6-word sequences that occur verbatim in the "
              f"child's report: median {st.median(ov):.2f}, mean {mean(ov):.2f}, "
              f">= 0.5 in {sum(x >= 0.5 for x in ov)}/{len(ov)} runs")
            res["overlap_median"] = st.median(ov)
        P("")
        return res

    S["rating_all_ok"] = rating_block(okd, "all process_ok rows")
    S["rating_clean"] = rating_block([d for d in okd if not d["log_reminded"] and not d["log_rbr"]],
                                     "clean rows (not reminded, no review before the report)")

    # ---------------- Part B2
    P("## B2. The prompt the main agent wrote for the subagent\n")
    cls = []
    for d in okd:
        c = PX.classify(d["fork_prompt"])
        c["no_inflate"] = bool(re.search(r"(?i)do not inflate|don't inflate", d["fork_prompt"]))
        c["key"] = d["key"]
        d["prompt_cls"] = c
        cls.append(c)
    m = len(cls)
    chars = [c["chars"] for c in cls]
    words = [c["words"] for c in cls]
    P(f"- {m} prompts; length chars: median {st.median(chars):.0f}, mean {mean(chars):.0f}, min "
      f"{min(chars)}, max {max(chars)}; words: median {st.median(words):.0f}, mean {mean(words):.0f}")
    P(f"- distinct prompt texts: {len({d['fork_prompt'] for d in okd})} of {m}")

    def line(k, label):
        v = sum(bool(c[k]) for c in cls)
        P(f"- {label}: {v}/{m} = {v / m:.1%}")
        return v / m
    P("\nrestating the reviewing instructions")
    for k, label in [("reviewer_role", "names a reviewer role"), ("iclr", "mentions ICLR"),
                     ("refers_to_criteria", "points to the 'criteria' already in the conversation"),
                     ("names_review_dimensions", "lists >= 3 of novelty/soundness/rigor/clarity/significance"),
                     ("rating_scale_values", "spells out the rating values 1/3/5/6/8/10"),
                     ("rating_scale_labels", "gives the verbal labels of the scale (strong reject ... )"),
                     ("score_ranges_1_4", "mentions the 1-4 sub-score range"),
                     ("asks_scores", "asks the subagent for scores / a rating / a recommendation"),
                     ("forbids_scores", "tells the subagent NOT to give scores"),
                     ("asks_json", "asks for JSON"), ("forbids_json", "tells it NOT to write the JSON review"),
                     ("no_author_identification", "repeats the no-author-identification rule"),
                     ("ignore_embedded_instructions", "repeats the ignore-embedded-instructions rule"),
                     ("paper_specific_refs", "contains paper-specific pointers (numbered table/section, quoted title, e.g.-list)")]:
        S.setdefault("prompt", {})[k] = line(k, label)
    P("\nlanguage that could shift scores")
    for k, label in [("critical", "'critical(ly)'"), ("skeptical", "'skeptical'"),
                     ("scrutinize", "'scrutinize'"), ("harsh_strict", "harsh/strict/tough"),
                     ("no_inflate", "'do not inflate (scores)'"),
                     ("stance1_any", "ANY explicit stance word above (critical/skeptical/scrutinize/harsh)"),
                     ("flag_note_any", "flag / note any / point out ..."),
                     ("overclaim_family", "overclaim / overstate / unsupported / outrun ..."),
                     ("inconsistency_error", "inconsistencies / contradictions / errors"),
                     ("missing_lack", "missing / lack / absence"),
                     ("confound_unfair", "confound / unfair / cherry-pick"), ("flaws", "'flaw(s)'"),
                     ("stance2_any", "ANY directed fault-finding term (previous six lines)"),
                     ("stance_any", "ANY of the above (explicit stance or directed fault-finding)"),
                     ("rigorous", "'rigorous(ly)' (also in the fixed system prompt: 'rigorously but fairly')"),
                     ("actually_supports", "intensifier 'actually/truly/genuinely/really' (supports ...)"),
                     ("fair_stance", "counter-weight: 'but fair' / balanced / objective / constructive"),
                     ("strengths", "asks for strengths"),
                     ("limitations_weaknesses", "asks for limitations / weaknesses")]:
        S["prompt"][k] = line(k, label)
    both = sum(c["stance_any"] and c["fair_stance"] for c in cls)
    only_neg = sum(c["stance_any"] and not c["fair_stance"] for c in cls)
    only_fair = sum(c["fair_stance"] and not c["stance_any"] for c in cls)
    P(f"- stance/fault-finding with a fairness counter-weight {both}, without one {only_neg}; "
      f"fairness wording only {only_fair}; neither {m - both - only_neg - only_fair}")
    # association with ratings (observational)
    for k in ("stance1_any", "stance2_any", "asks_scores", "fair_stance"):
        a = [d["my_rating"] for d in okd if d["prompt_cls"][k]]
        b = [d["my_rating"] for d in okd if not d["prompt_cls"][k]]
        P(f"- mean parent rating when prompt has {k}: {mean(a):.3f} (n={len(a)}) vs without "
          f"{mean(b):.3f} (n={len(b)})  [raw, not paper-adjusted]")
    S["prompt"]["chars_median"] = st.median(chars)
    S["prompt"]["words_median"] = st.median(words)

    if cfg.main_log and os.path.exists(cfg.main_log):
        cell = defaultdict(list)
        seen = {}
        for line_ in open(cfg.main_log, encoding="utf-8"):
            if line_.strip():
                r = json.loads(line_)
                if r.get("parsed_ok"):
                    seen[r["key"]] = r          # last usable row per key
        for r in seen.values():
            cell[(r["arm"], r["arxiv_id"], r["v"])].append(r["rating"])
        P("\npaper-adjusted comparison (B1 parent rating minus the mean rating of the reference arm "
          "for the same paper-version; clean rows only; CI = percentile bootstrap over papers, "
          "4000 draws; observational - the main agent chose the wording):")
        import numpy as np
        cl = [d for d in okd if not d["log_reminded"] and not d["log_rbr"]]

        def boot_diff(pairs_a, pairs_b, seed=7):
            """mean(a) - mean(b) with a paper-cluster bootstrap; pairs = [(paper, value)]"""
            papers = sorted({p_ for p_, _ in pairs_a} | {p_ for p_, _ in pairs_b})
            ia = defaultdict(list)
            ib = defaultdict(list)
            for p_, x in pairs_a:
                ia[p_].append(x)
            for p_, x in pairs_b:
                ib[p_].append(x)
            rng_ = np.random.default_rng(seed)
            out = []
            for _ in range(4000):
                pick = [papers[i] for i in rng_.integers(0, len(papers), len(papers))]
                xa = [x for p_ in pick for x in ia[p_]]
                xb = [x for p_ in pick for x in ib[p_]]
                if xa and xb:
                    out.append(np.mean(xa) - np.mean(xb))
            return np.percentile(out, 2.5), np.percentile(out, 97.5)

        S["adjusted"] = {}
        for arm in ("direct", "harness"):
            dl = [(d, d["my_rating"] - mean(cell[(arm, d["arxiv_id"], d["v"])])) for d in cl
                  if cell.get((arm, d["arxiv_id"], d["v"]))]
            P(f"- vs {arm}: {len(dl)} runs with a reference cell, mean delta {mean(x for _, x in dl):+.3f}")
            S["adjusted"][arm] = {"n": len(dl), "mean_delta": mean(x for _, x in dl)}
            for k in ("stance1_any", "stance2_any", "stance_any", "asks_scores", "fair_stance",
                      "rigorous"):
                a_ = [(d["arxiv_id"], x) for d, x in dl if d["prompt_cls"][k]]
                b_ = [(d["arxiv_id"], x) for d, x in dl if not d["prompt_cls"][k]]
                lo, hi = boot_diff(a_, b_)
                P(f"  - {k}: with {mean(x for _, x in a_):+.3f} (n={len(a_)}) | without "
                  f"{mean(x for _, x in b_):+.3f} (n={len(b_)}) | difference "
                  f"{mean(x for _, x in a_) - mean(x for _, x in b_):+.3f} [{lo:+.3f}, {hi:+.3f}]")
                S["adjusted"][arm][k] = {"diff": mean(x for _, x in a_) - mean(x for _, x in b_),
                                         "ci": [float(lo), float(hi)]}
            cr = [(d, L.snap_rating(d["child_rating"]) - mean(cell[(arm, d["arxiv_id"], d["v"])]))
                  for d in cl if d.get("child_rating") is not None
                  and cell.get((arm, d["arxiv_id"], d["v"]))]
            P(f"  - child-report rating minus {arm} cell mean: {mean(x for _, x in cr):+.3f} "
              f"(n={len(cr)})")
            nr = [x for d, x in dl if d.get("child_rating") is None and d.get("child_rating_lo") is None]
            yr = [x for d, x in dl if d.get("child_rating") is not None]
            P(f"  - parent delta when the report carried a rating {mean(yr):+.3f} (n={len(yr)}) | "
              f"no rating in the report {mean(nr):+.3f} (n={len(nr)})")
            sv = [(d["arxiv_id"], x) for d, x in dl if d["other_tools"] or d["n_fork_calls"] != 1]
            sc = [(d["arxiv_id"], x) for d, x in dl if not (d["other_tools"] or d["n_fork_calls"] != 1)]
            lo, hi = boot_diff(sv, sc)
            P(f"  - strict-protocol sensitivity: runs with an extra tool call or a second fork call "
              f"{mean(x for _, x in sv):+.3f} (n={len(sv)}) | fully compliant "
              f"{mean(x for _, x in sc):+.3f} (n={len(sc)}) | difference "
              f"{mean(x for _, x in sv) - mean(x for _, x in sc):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        P("within the same paper-version (B1 clean replicates that differ in prompt wording; "
          "weighted mean of per-cell differences, paper-bootstrap CI):")
        for k in ("stance_any", "stance1_any", "stance2_any", "fair_stance", "asks_scores", "rigorous"):
            cells = defaultdict(lambda: {True: [], False: []})
            for d in cl:
                cells[(d["arxiv_id"], d["v"])][bool(d["prompt_cls"][k])].append(d["my_rating"])
            per = defaultdict(list)
            for (aid, _v), c in cells.items():
                if c[True] and c[False]:
                    per[aid].append((mean(c[True]) - mean(c[False]),
                                     1 / (1 / len(c[True]) + 1 / len(c[False]))))
            ids = sorted(per)
            if not ids:
                continue
            allv = [x for i in ids for x in per[i]]
            est = sum(a_ * w for a_, w in allv) / sum(w for _, w in allv)
            rng_ = np.random.default_rng(1)
            bs = []
            for _ in range(4000):
                smp = [x for i in rng_.integers(0, len(ids), len(ids)) for x in per[ids[i]]]
                bs.append(sum(a_ * w for a_, w in smp) / sum(w for _, w in smp))
            P(f"  - {k}: {len(allv)} cells / {len(ids)} papers, with minus without {est:+.3f} "
              f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
            S.setdefault("within_cell", {})[k] = {"cells": len(allv), "est": est,
                                                  "ci": [float(np.percentile(bs, 2.5)),
                                                         float(np.percentile(bs, 97.5))]}

    # prompt wording and parent-vs-child changes by version position (manifest + text files)
    try:
        import numpy as np2
        pos, nver = {}, {}
        for line_ in open(L.REPO + "/data/processed/pilot_manifest.jsonl", encoding="utf-8"):
            if not line_.strip():
                continue
            pm = json.loads(line_)
            safe = pm["arxiv_id"].replace("/", "_")
            vs = sorted(v["v"] for v in pm["versions"]
                        if os.path.exists(f"{L.REPO}/data/processed/texts/{safe}/v{v['v']}.md"))
            for i, v in enumerate(vs):
                pos[(pm["arxiv_id"], v)] = i
            nver[pm["arxiv_id"]] = len(vs)
        cl = [d for d in okd if not d["log_reminded"] and not d["log_rbr"]
              and (d["arxiv_id"], d["v"]) in pos]
        P("\nprompt wording by version position (clean rows; position from manifest + text files):")
        for label, sel in (("first version (position 0)", lambda d: pos[(d["arxiv_id"], d["v"])] == 0),
                           ("later versions (position >= 1)", lambda d: pos[(d["arxiv_id"], d["v"])] >= 1)):
            xs_ = [d for d in cl if sel(d)]
            cr_ = [d for d in xs_ if d.get("child_rating") is not None]
            P(f"- {label}: n={len(xs_)}, stance/fault-finding wording "
              f"{mean(d['prompt_cls']['stance_any'] for d in xs_):.1%}, explicit stance word "
              f"{mean(d['prompt_cls']['stance1_any'] for d in xs_):.1%}, fairness counter-weight "
              f"{mean(d['prompt_cls']['fair_stance'] for d in xs_):.1%}; parent changed the child's "
              f"rating in {sum(d['my_rating'] != L.snap_rating(d['child_rating']) for d in cr_)}/{len(cr_)}")
        pp = defaultdict(lambda: {"first": [], "last": []})
        for d in cl:
            ps = pos[(d["arxiv_id"], d["v"])]
            if ps == 0:
                pp[d["arxiv_id"]]["first"].append(d["prompt_cls"]["stance_any"])
            elif ps == nver[d["arxiv_id"]] - 1:
                pp[d["arxiv_id"]]["last"].append(d["prompt_cls"]["stance_any"])
        dd = [mean(c["last"]) - mean(c["first"]) for c in pp.values() if c["first"] and c["last"]]
        rg = np2.random.default_rng(3)
        bs = [float(np2.mean(rg.choice(dd, len(dd)))) for _ in range(4000)]
        P(f"- within paper, last minus first version, share of prompts with stance/fault-finding "
          f"wording: {mean(dd):+.3f} over {len(dd)} papers, bootstrap CI "
          f"[{np2.percentile(bs, 2.5):+.3f}, {np2.percentile(bs, 97.5):+.3f}]")
    except OSError as exc:
        P(f"(position analysis skipped: {exc})")

    # ---------------- Part B3
    P("\n## B3. What the child saw\n")
    P(f"- seed user messages (kinds): {dict(Counter(tuple(d['seed_user_kinds']) for d in okd))}")
    P(f"- seed turns: {dict(Counter(tuple(d['seed_turns']) for d in okd))}")
    P(f"- seed or child input contains the turn-2 process instruction: "
      f"{sum(d['seed_has_process_text'] for d in okd)}/{len(okd)}")
    P(f"- seed contains a review already written by the parent in turn 1: "
      f"{[(d['key'], d['session_id']) for d in okd if d['seed_has_parent_review']]}")
    P(f"- child's own user messages (kinds): "
      f"{dict(Counter(tuple(d['child_own_user_kinds']) for d in okd))}")
    P(f"- turn-1 reasoning of the parent (part of the seeded assistant record), chars: median "
      f"{st.median(d['turn1_reasoning_chars'] for d in okd):.0f}, max "
      f"{max(d['turn1_reasoning_chars'] for d in okd)}; runs > 2000 chars: "
      f"{sum(d['turn1_reasoning_chars'] > 2000 for d in okd)}")
    P(f"- parent reasoning before the fork call (turns 1+2), chars: median "
      f"{st.median(d['reasoning_before_fork_chars'] for d in okd):.0f}, max "
      f"{max(d['reasoning_before_fork_chars'] for d in okd)}; runs > 2000 chars: "
      f"{sum(d['reasoning_before_fork_chars'] > 2000 for d in okd)}")
    ratio = [d["child_first_prompt_tokens"] / d["turn1_prompt_tokens"] for d in okd
             if d.get("child_first_prompt_tokens") and d.get("turn1_prompt_tokens")]
    P(f"- child's first model call prompt tokens / parent's turn-1 prompt tokens: min "
      f"{min(ratio):.3f}, median {st.median(ratio):.3f} (n={len(ratio)}) - the child's request "
      f"was at least as large as the paper turn")
    import numpy as _np
    xs, ys = [], []
    for d in okd:
        if d.get("turn1_reply_is_READY") and d.get("child_first_prompt_tokens"):
            xs.append(len(d["fork_prompt"]))
            ys.append(d["child_first_prompt_tokens"] - d["turn1_prompt_tokens"])
    if len(xs) > 10:
        slope, icpt = _np.polyfit(_np.array(xs, float), _np.array(ys, float), 1)
        P(f"- token accounting: (child first-call prompt tokens - parent turn-1 prompt tokens) = "
          f"{icpt:.0f} + {slope:.3f} x fork-prompt chars (n={len(xs)}, r="
          f"{_np.corrcoef(xs, ys)[0, 1]:.3f}); the fork prompt is counted once "
          f"({1 / slope:.2f} chars/token) and the constant matches the dsh notice (about 570 chars) "
          f"plus the child's runtime-context block (1,245 chars), leaving no room for the 846-char "
          f"turn-2 process message")
    P(f"- child closing message holds a full JSON review: "
      f"{sum(d.get('child_closing_json_rating') is not None for d in okd)}/{len(okd)}")

    # ---------------- Part B4
    P("\n## B4. Token usage and model calls (process_ok rows)\n")
    for k in ("input", "cache_read", "output"):
        pm = mean(d["parent_usage"][k] for d in okd)
        cm = mean(d["child_usage"][k] for d in okd)
        P(f"- {k}: parent mean {pm:,.0f} | child mean {cm:,.0f} | child share {cm / (pm + cm):.1%}")
    tot_p = mean(sum(d["parent_usage"][k] for k in ("input", "cache_read", "output")) for d in okd)
    tot_c = mean(sum(d["child_usage"][k] for k in ("input", "cache_read", "output")) for d in okd)
    P(f"- all tokens: parent mean {tot_p:,.0f} | child mean {tot_c:,.0f} | child share {tot_c / (tot_p + tot_c):.1%}")
    pc = [d["parent_usage"]["calls"] for d in okd]
    cc = [d["child_usage"]["calls"] for d in okd]
    P(f"- completed model calls per run: parent mean {mean(pc):.2f} (distribution "
      f"{dict(sorted(Counter(pc).items()))}), child mean {mean(cc):.2f} (distribution "
      f"{dict(sorted(Counter(cc).items()))}), total mean {mean(a + b for a, b in zip(pc, cc)):.2f}")
    fa = [d["parent_usage"]["attempts_failed"] + d["child_usage"]["attempts_failed"] for d in okd]
    P(f"- failed attempts (no usage recorded) per run: mean {mean(fa):.2f}; runs with any "
      f"{sum(x > 0 for x in fa)}; characters streamed in failed attempts, total "
      f"{sum(d['attempt_stream_chars'] for d in okd):,} (about "
      f"{sum(d['attempt_stream_chars'] for d in okd) / 4 / max(sum(d['parent_usage']['output'] + d['child_usage']['output'] for d in okd), 1):.1%} "
      f"of counted output tokens at 4 chars/token)")
    um = sum(any(d["log_usage"].get(k) != d["parent_usage"][k] + d["child_usage"][k]
                 for k in ("input", "cache_read", "output")) for d in okd)
    P(f"- logged usage equals my parent + child sum in {len(okd) - um}/{len(okd)} rows")
    rm = sum(d["log_llm_retries"] != d["parent_usage"]["retries"] + d["child_usage"]["retries"]
             for d in okd)
    P(f"- logged llm_retries equals my count in {len(okd) - rm}/{len(okd)} rows")

    def q(xs):
        xs = sorted(xs)
        return f"p10 {xs[int(0.1 * (len(xs) - 1))]:,} / median {st.median(xs):,.0f} / p90 {xs[int(0.9 * (len(xs) - 1))]:,}"
    P(f"- parent, final-review step: reasoning chars {q([d['final_reasoning_chars'] for d in okd])}; "
      f"output tokens {q([d['final_output_tokens'] for d in okd])}; runs with < 500 reasoning chars "
      f"{sum(d['final_reasoning_chars'] < 500 for d in okd)}/{len(okd)}")
    P(f"- child: reasoning chars {q([d['child_reasoning_chars'] for d in okd])}; output tokens "
      f"{q([d['child_usage']['output'] for d in okd])}; report chars {q([d['report_chars'] for d in okd])}")
    S["usage"] = {"parent_tokens_mean": tot_p, "child_tokens_mean": tot_c,
                  "child_share": tot_c / (tot_p + tot_c), "parent_calls_mean": mean(pc),
                  "child_calls_mean": mean(cc)}

    # ---------------- outputs
    cols = ["key", "session_id", "child_id", "log_process_ok", "log_reminded", "log_rbr",
            "my_process_ok", "reminded", "review_before_report", "n_fork_calls", "n_children",
            "child_paper_exact", "child_done", "report_delivered", "report_via", "final_exists",
            "log_rating", "my_rating", "child_rating", "child_rating_method", "child_rating_lo",
            "child_rating_hi", "child_closing_json_rating", "report_is_json_review", "report_chars",
            "n_review_outputs", "post_report_ratings", "other_tools", "strict_violations",
            "disagreements", "cwd", "wall_s", "log_latency_s"]
    with open(os.path.join(cfg.out_dir, "b1_runs.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols + ["prompt_chars", "stance1_any", "stance2_any", "asks_scores", "fair_stance"])
        for d in D:
            c = d.get("prompt_cls") or {}
            w.writerow([json.dumps(d.get(k)) if isinstance(d.get(k), (list, dict)) else d.get(k)
                        for k in cols] + [c.get("chars"), c.get("stance1_any"), c.get("stance2_any"),
                                          c.get("asks_scores"), c.get("fair_stance")])
    with open(os.path.join(cfg.out_dir, "b1_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(S, fh, indent=1, default=str)
    with open(os.path.join(cfg.out_dir, "b1_prompts.jsonl"), "w", encoding="utf-8") as fh:
        for d in okd:
            fh.write(json.dumps({"key": d["key"], "session_id": d["session_id"],
                                 "prompt": d["fork_prompt"], "cls": d["prompt_cls"],
                                 "child_rating": d.get("child_rating"),
                                 "child_rating_snippet": d.get("child_rating_snippet"),
                                 "parent_rating": d["my_rating"]}, ensure_ascii=False) + "\n")
    P(f"\nwrote {cfg.out_dir}/b1_runs.csv, b1_summary.json, b1_prompts.jsonl")


if __name__ == "__main__":
    main()
