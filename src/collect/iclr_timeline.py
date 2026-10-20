"""ICLR per-year review timelines and stage labeling for arXiv version timestamps.

Dates were read from official iclr.cc pages (Dates / CallForPapers / AuthorGuide,
plus web.archive.org snapshots of the official pages for 2018-2020) in Oct 2026.
All instants are UTC. Submission cutoffs use the officially stated instant where
known (2023+ deadlines are AoE = UTC-12, so date D ends at (D+1)T11:59Z); decision
cutoffs conservatively use the end of the announced decision day in UTC.

Stage semantics for an arXiv version posted at time t for an ICLR-{year} paper:
  coarse: pre_submission | during_review | post_decision | unknown
  fine:   pre_submission | before_reviews | during_rebuttal | awaiting_decision
          | post_decision | unknown
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class YearTimeline:
    year: int
    submission_cutoff: datetime          # last instant the reviewed version could be posted
    reviews_released: datetime | None    # rebuttal window opens
    rebuttal_end: datetime | None        # last day authors could respond/revise
    decision_cutoff: datetime            # end of decision-announcement day (UTC)
    camera_ready: datetime | None        # approximate for some years


TIMELINES: dict[int, YearTimeline] = {
    2018: YearTimeline(2018, _utc("2017-10-28T00:00"), _utc("2017-11-27T23:59"),
                       _utc("2018-01-05T23:59"), _utc("2018-01-29T23:59"), None),
    2019: YearTimeline(2019, _utc("2018-09-27T22:00"), _utc("2018-11-05T23:59"),
                       _utc("2018-11-23T23:59"), _utc("2018-12-21T23:59"), None),
    2020: YearTimeline(2020, _utc("2019-09-25T15:00"), _utc("2019-11-04T23:59"),
                       _utc("2019-11-15T23:59"), _utc("2019-12-19T23:59"), _utc("2020-02-14T23:59")),
    2021: YearTimeline(2021, _utc("2020-10-02T15:00"), _utc("2020-11-10T23:59"),
                       _utc("2020-11-24T23:59"), _utc("2021-01-14T23:59"), _utc("2021-03-17T23:59")),
    2022: YearTimeline(2022, _utc("2021-10-06T00:00"), _utc("2021-11-09T07:59"),
                       _utc("2021-11-22T23:59"), _utc("2022-01-24T23:59"), _utc("2022-03-31T23:59")),
    2023: YearTimeline(2023, _utc("2022-09-29T11:59"), _utc("2022-11-05T01:00"),
                       _utc("2022-11-18T23:59"), _utc("2023-01-21T23:59"), _utc("2023-03-01T23:59")),
    2024: YearTimeline(2024, _utc("2023-09-29T11:59"), _utc("2023-11-10T23:59"),
                       _utc("2023-11-22T23:59"), _utc("2024-01-15T23:59"), _utc("2024-04-14T23:59")),
    2025: YearTimeline(2025, _utc("2024-10-02T11:59"), _utc("2024-11-12T23:59"),
                       _utc("2024-11-26T23:59"), _utc("2025-01-22T23:59"), _utc("2025-03-14T23:59")),
    2026: YearTimeline(2026, _utc("2025-09-25T11:59"), _utc("2025-11-11T21:00"),
                       _utc("2025-12-03T21:00"), _utc("2026-01-25T23:59"), None),
}


def label_stage(iclr_year: int, t: datetime) -> tuple[str, str]:
    """Return (coarse, fine) review-stage labels for an arXiv version timestamp.

    Timestamps on or within a boundary day are labeled conservatively (a version
    posted on the decision day itself still counts as during_review).
    """
    tl = TIMELINES.get(iclr_year)
    if tl is None:
        return "unknown", "unknown"
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    if t <= tl.submission_cutoff:
        return "pre_submission", "pre_submission"
    if t <= tl.decision_cutoff:
        if tl.reviews_released and t <= tl.reviews_released:
            return "during_review", "before_reviews"
        if tl.rebuttal_end and t <= tl.rebuttal_end:
            return "during_review", "during_rebuttal"
        return "during_review", "awaiting_decision"
    return "post_decision", "post_decision"
