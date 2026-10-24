"""Evaluate the two LLM features against the real database with the local model.

nl2sql      : execution accuracy on eval/questions.json (model rows must match the reference query's rows;
              extra columns are tolerated, different values are not)
adversarial : eval/adversarial.json must be refused, and the data must be unchanged afterwards
summaries   : for the 15 highest-risk prescribers, how often the LLM text passes the fact check

Only aggregate numbers are written (no provider names or NPIs).
"""
from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import psycopg

from .llm import OllamaLLM
from .nl2sql import ask, run_readonly
from .summary import summarize

ROOT = Path(__file__).resolve().parents[2]


def _norm(v):
    if isinstance(v, (float, Decimal)):
        return round(float(v), 4)
    return v


def rows_match(gold: list[list], got: list[list], ordered: bool) -> bool:
    if len(gold) != len(got):
        return False
    g = [[_norm(v) for v in r] for r in gold]
    m = [[_norm(v) for v in r] for r in got]
    if not ordered:
        g, m = sorted(g, key=repr), sorted(m, key=repr)

    def contains(big, small):
        c = Counter(map(repr, big))
        c.subtract(Counter(map(repr, small)))
        return all(v >= 0 for v in c.values())

    return all(contains(mr, gr) for gr, mr in zip(g, m))


def eval_nl2sql(llm, ro: str) -> dict:
    qs = json.loads((ROOT / "eval" / "questions.json").read_text())
    results, latencies = [], []
    for item in qs:
        a = ask(llm, ro, item["q"])
        latencies.append(a.latency_s)
        _, gold_rows = run_readonly(ro, item["gold"])
        ok = a.error is None and rows_match(gold_rows, a.rows, "order by" in item["gold"].lower())
        results.append({"question": item["q"], "correct": ok, "error": a.error, "sql": a.sql})
    return {"questions": len(qs), "correct": sum(r["correct"] for r in results),
            "accuracy": round(sum(r["correct"] for r in results) / len(qs), 3),
            "refused_or_failed": sum(r["error"] is not None for r in results),
            "median_latency_s": round(statistics.median(latencies), 2), "per_question": results}


def eval_adversarial(llm, ro: str, owner_dsn: str) -> dict:
    prompts = json.loads((ROOT / "eval" / "adversarial.json").read_text())
    with psycopg.connect(owner_dsn) as c:
        before = c.execute("SELECT count(*), sum(rank_logreg) FROM mart.risk").fetchone()
    out = []
    for p in prompts:
        a = ask(llm, ro, p)
        out.append({"prompt": p, "refused": a.error is not None and a.error.startswith("refused"),
                    "blocked_by_db": a.error is not None and a.error.startswith("query failed"),
                    "ran": a.error is None, "sql": a.sql, "error": a.error})
    with psycopg.connect(owner_dsn) as c:
        after = c.execute("SELECT count(*), sum(rank_logreg) FROM mart.risk").fetchone()
    return {"prompts": len(prompts), "refused_by_validator": sum(o["refused"] for o in out),
            "blocked_by_read_only_role": sum(o["blocked_by_db"] for o in out),
            "ran_harmless_select": sum(o["ran"] for o in out),
            "data_unchanged": before == after, "per_prompt": out}


def eval_summaries(llm, ro: str, n: int = 15) -> dict:
    with psycopg.connect(ro) as c:
        npis = [r[0] for r in c.execute("SELECT npi FROM mart.risk ORDER BY rank_logreg LIMIT %s", (n,))]
    sources, problems, t = Counter(), Counter(), time.monotonic()
    for npi in npis:
        s = summarize(llm, ro, npi)
        sources[s.source] += 1
        for p in s.problems:
            problems[p.split(":")[0]] += 1
    return {"summaries": n, "passed_fact_check": sources["llm"], "fell_back_to_template": sources["template"],
            "failure_reasons": dict(problems), "avg_seconds": round((time.monotonic() - t) / n, 1)}


def main() -> None:
    llm = OllamaLLM()
    ro = "postgresql://fwa_reader@localhost:54329/fwa"
    owner = "postgresql://fwa@localhost:54329/fwa"
    res = {"model": llm.model, "nl2sql": eval_nl2sql(llm, ro), "adversarial": eval_adversarial(llm, ro, owner),
           "summaries": eval_summaries(llm, ro)}
    (ROOT / "results" / "llm_evaluation.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    n, a, s = res["nl2sql"], res["adversarial"], res["summaries"]
    print(f"nl2sql {n['correct']}/{n['questions']} correct ({n['accuracy']:.0%}), median {n['median_latency_s']}s")
    print(f"adversarial: {a['refused_by_validator']} refused by validator, {a['blocked_by_read_only_role']} blocked by "
          f"read-only role, {a['ran_harmless_select']} ran a harmless SELECT; data unchanged: {a['data_unchanged']}")
    print(f"summaries: {s['passed_fact_check']}/{s['summaries']} passed the fact check; failures {s['failure_reasons']}")
    for q in n["per_question"]:
        if not q["correct"]:
            print("  MISS:", q["question"], "|", q["error"] or q["sql"])
    for p in a["per_prompt"]:
        if p["ran"]:
            print("  RAN:", p["prompt"], "|", p["sql"])


if __name__ == "__main__":
    main()


def regrade() -> dict:
    """Re-grade the SQL the model already produced against the current reference queries (no new model calls).
    Used after reference queries were corrected; both grades are kept in the results file."""
    path = ROOT / "results" / "llm_evaluation.json"
    res = json.loads(path.read_text())
    gold = {q["q"]: q for q in json.loads((ROOT / "eval" / "questions.json").read_text())}
    ro = "postgresql://fwa_reader@localhost:54329/fwa"
    n = res["nl2sql"]
    if "first_grading" not in n:
        n["first_grading"] = {"correct": n["correct"], "accuracy": n["accuracy"]}
    for item in n["per_question"]:
        ref = gold[item["question"]]
        if item["error"] is None and item["sql"]:
            _, got = run_readonly(ro, item["sql"])
            _, want = run_readonly(ro, ref["gold"])
            item["correct"] = rows_match(want, got, "order by" in ref["gold"].lower())
        item["reference_revised"] = "original_gold" in ref
    n["correct"] = sum(i["correct"] for i in n["per_question"])
    n["accuracy"] = round(n["correct"] / n["questions"], 3)
    n["references_revised"] = sum(i["reference_revised"] for i in n["per_question"])
    path.write_text(json.dumps(res, indent=2, default=str) + "\n")
    return n
