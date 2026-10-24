"""Out-of-time evaluation: do 2022 prescribing outlier scores rank providers who were excluded by OIG
from 2023 onward above everyone else?

Labels: an NPI appears in the LEIE with an exclusion date on or after 2023-01-01. Exclusion is a weak,
delayed proxy for misconduct (it also covers license revocations and non-prescribing offences), so
these numbers measure enrichment, not accuracy.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import psycopg

KS = (100, 500, 1000, 5000, 10000)

RANKINGS = {
    "composite (mean of top-2 peer surprise scores)": "score",
    "max single peer surprise score": "max_z",
    "raw opioid share (no peer adjustment)": "opioid_share",
    "raw cost per beneficiary (no peer adjustment)": "cost_per_beneficiary",
    "prescribing volume (total claims)": "claims",
}


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(labels)
    neg = len(labels) - pos
    s = sum(r for r, y in zip(ranks, labels) if y)
    return (s - pos * (pos + 1) / 2) / (pos * neg)


def load(dsn: str) -> list[dict]:
    sql = """
    SELECT s.npi, s.score::float, z.max_z::float, f.opioid_share::float, f.cost_per_beneficiary::float,
           p.claims::float,
           EXISTS (SELECT 1 FROM core.exclusion e WHERE e.npi = s.npi AND e.excluded_on >= DATE '2023-01-01') AS y
    FROM mart.scores s
    JOIN mart.features f USING (npi)
    JOIN core.prescriber p USING (npi)
    JOIN (SELECT npi, max(surprise) AS max_z FROM mart.peer_scores GROUP BY npi) z USING (npi)
    """
    with psycopg.connect(dsn) as conn:
        cols = ["npi", "score", "max_z", "opioid_share", "cost_per_beneficiary", "claims", "y"]
        return [dict(zip(cols, r)) for r in conn.execute(sql)]


def evaluate(rows: list[dict], seed: int = 7) -> dict:
    n = len(rows)
    pos = sum(r["y"] for r in rows)
    base = pos / n
    out = {"population": n, "later_excluded": pos, "base_rate": base, "rankings": {}}
    rng = random.Random(seed)
    for name, col in RANKINGS.items():
        # missing values (e.g. no opioid claims) rank last; random tie-break so ties do not favour labels
        keyed = [((r[col] if r[col] is not None else -math.inf), rng.random(), r["y"]) for r in rows]
        keyed.sort(reverse=True)
        res = {"auc": round(auc([k[0] if k[0] != -math.inf else -1e18 for k in keyed], [k[2] for k in keyed]), 4)}
        for k in KS:
            hits = sum(x[2] for x in keyed[:k])
            lo, hi = wilson(hits, k)
            res[f"top_{k}"] = {"hits": hits, "precision": round(hits / k, 5),
                               "precision_95ci": [round(lo, 5), round(hi, 5)],
                               "lift": round((hits / k) / base, 2), "recall": round(hits / pos, 4)}
        out["rankings"][name] = res
    out["random_expected_hits"] = {f"top_{k}": round(k * base, 2) for k in KS}
    return out


def main(dsn: str, out: Path) -> dict:
    res = evaluate(load(dsn))
    out.write_text(json.dumps(res, indent=2) + "\n")
    return res
