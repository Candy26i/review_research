"""Shared helpers for data collection: polite HTTP with retries, JSONL checkpointing."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

# arXiv API terms of use: at most 1 request every 3 seconds.
ARXIV_API_DELAY_S = 3.0
USER_AGENT = "review_research-pilot/0.1 (+https://github.com/Candy26i/review_research)"


class RateLimiter:
    """Enforce a minimum interval between calls (per process)."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        sleep_for = self._last + self.min_interval_s - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last = time.monotonic()


_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = USER_AGENT
    return _session


def http_get(
    url: str,
    *,
    params: dict | None = None,
    limiter: RateLimiter | None = None,
    timeout: int = 60,
    retries: int = 4,
    backoff_s: float = 5.0,
) -> requests.Response:
    """GET with retries on 429/5xx and connection errors. Raises on final failure."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        if limiter:
            limiter.wait()
        try:
            resp = session().get(url, params=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = float(resp.headers.get("Retry-After", backoff_s * (attempt + 1)))
                time.sleep(min(retry_after, 120))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(backoff_s * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts") from last_exc


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def done_keys(path: str | Path, key: str) -> set:
    """Keys already present in a checkpoint JSONL — makes long runs resumable."""
    return {row[key] for row in read_jsonl(path) if key in row}


def norm_title(title: str) -> str:
    """Normalize a paper title for matching: lowercase, alphanumeric+space, collapsed."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()
