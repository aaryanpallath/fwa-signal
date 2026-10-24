"""Plain-language case summaries for investigators, written by the LLM but only shown if every number in
the text matches a fact from the database and the language stays non-accusatory. Otherwise a deterministic
template is returned (and the reason is reported), so a reviewer never sees an unverified figure.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import psycopg

from .llm import LLM

ACCUSATORY = re.compile(r"\b(fraud\w*|illegal\w*|criminal\w*|guilty|scam\w*|kickback\w*|pill mill|abus(e|ing|ive)|"
                        r"committed|wrongdoing\s+(was|is)\s+(found|proven|confirmed))\b", re.IGNORECASE)
DISCLAIMER = ("This is a statistical lead for review, not a finding of wrongdoing; unusual prescribing often has "
              "legitimate clinical explanations.")
SYSTEM = ("You write a short, neutral case summary (3-5 sentences) for a Medicare Part D program-integrity analyst. "
          "Use ONLY the facts provided, quote numbers exactly as given, and do not speculate about intent. "
          "Never state or imply fraud, crime or wrongdoing: describe the statistics only.")

FEATURE_LABELS = {
    "opioid_share": ("opioid share of claims", "pct"), "long_acting_opioid_share": ("long-acting share of opioid claims", "pct"),
    "cost_per_beneficiary": ("drug cost per beneficiary", "usd"), "brand_share": ("brand-name share of claims", "pct"),
    "days_per_claim": ("days of supply per claim", "num"), "antipsychotic_65_per_bene": ("antipsychotic claims per beneficiary 65+", "num"),
}


def _fmt(v: float, kind: str) -> str:
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "usd":
        return f"${v:,.0f}"
    return f"{v:.1f}"


def facts(dsn: str, npi: int) -> dict[str, Any] | None:
    with psycopg.connect(dsn) as conn:
        cur = conn.execute("SELECT * FROM mart.provider_risk WHERE npi = %s", (npi,))
        row = cur.fetchone()
        if row is None:
            return None
        p = dict(zip([d.name for d in cur.description], row))
        zs = conn.execute("""SELECT feature, value::float, peer_median::float, share_at_or_above::float,
                                    peer_group, peer_n
                             FROM mart.peer_scores WHERE npi = %s ORDER BY surprise DESC LIMIT 3""",
                          (npi,)).fetchall()
        total = conn.execute("SELECT count(*) FROM mart.risk").fetchone()[0]
    f = {"prescribing_year": 2022, "npi": p["npi"], "name": " ".join(x for x in (p["first_name"], p["last_or_org_name"]) if x),
         "specialty": p["specialty"], "location": f"{p['city']}, {p['state']}",
         "claims": f"{p['claims']:,}", "beneficiaries": f"{p['beneficiaries']:,}",
         "drug_cost": f"${float(p['drug_cost']):,.0f}",
         "risk_rank": f"{p['rank_logreg']:,} of {total:,}" if p["rank_logreg"] else None,
         "excluded_but_prescribing": bool(p["excluded_but_prescribing"]), "unusual_measures": []}
    for feat, val, med, tail, grp, n in zs:
        label, kind = FEATURE_LABELS[feat]
        f["unusual_measures"].append({"measure": label, "value": _fmt(val, kind), "peer_median": _fmt(med, kind),
                                      "peer_group": f"{p['specialty']} in {p['state']}" if grp == "state"
                                      else f"{p['specialty']} nationally", "peer_count": f"{n:,}",
                                      "higher_than_pct_of_peers": f"{100 * (1 - tail):.1f}%"})
    return f


_NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


def _numbers(text: str) -> set[str]:
    return {m.group(0).rstrip(".,").lstrip("$").replace(",", "") for m in _NUM.finditer(text)}


def check(text: str, fx: dict[str, Any]) -> list[str]:
    """Problems with a generated summary; empty means it may be shown."""
    problems = []
    allowed = _numbers(json.dumps(fx)) | {"65"}  # "65+" appears in a measure name
    unknown = sorted(n for n in _numbers(text) if n not in allowed)
    if unknown:
        problems.append(f"numbers not in the facts: {unknown[:6]}")
    if m := ACCUSATORY.search(text):
        problems.append(f"accusatory language: '{m.group(0)}'")
    if len(text) > 1200:
        problems.append("too long")
    return problems


def template(fx: dict[str, Any]) -> str:
    parts = [(f"{fx['name']} ({fx['specialty']}, {fx['location']}) had {fx['claims']} Part D claims for "
             f"{fx['beneficiaries']} beneficiaries in {fx['prescribing_year']}, totalling {fx['drug_cost']}.")]
    for u in fx["unusual_measures"][:2]:
        parts.append(f"Their {u['measure']} was {u['value']} versus a median of {u['peer_median']} among "
                     f"{u['peer_count']} {u['peer_group']} peers, higher than {u['higher_than_pct_of_peers']} of them.")
    if fx["risk_rank"]:
        parts.append(f"The exclusion-risk model ranks them {fx['risk_rank']}.")
    if fx["excluded_but_prescribing"]:
        parts.append(f"They appear on the OIG exclusion list with no reinstatement, yet billed Part D in "
                     f"{fx['prescribing_year']}.")
    return " ".join(parts)


@dataclass
class Summary:
    npi: int
    text: str
    source: str  # "llm" or "template"
    problems: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


def summarize(llm: LLM, dsn: str, npi: int) -> Summary | None:
    fx = facts(dsn, npi)
    if fx is None:
        return None
    c = llm.complete(SYSTEM, "Facts (JSON):\n" + json.dumps(fx, indent=1))
    text = c.text.strip()
    problems = check(text, fx)
    if problems:
        return Summary(npi, template(fx), "template", problems)
    return Summary(npi, text, "llm")
