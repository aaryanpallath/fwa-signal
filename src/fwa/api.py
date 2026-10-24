"""FastAPI service over the marts.

GET  /health
GET  /providers/{npi}                     profile, most unusual measures vs peers, model risk
GET  /providers/{npi}/summary             LLM case summary, shown only if it passes the fact check
GET  /leads?state=&specialty=&by=risk|outlier&limit=
GET  /compliance/excluded-but-prescribing
POST /ask {"question": "..."}             plain-English question -> validated, read-only SQL -> rows

All database reads run as the read-only role except the fixed queries below, which use bound parameters.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Literal

import psycopg
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .llm import LLM, OllamaLLM
from .nl2sql import ask
from .summary import facts, summarize

# The API only ever connects as the restricted reader; it never falls back to an owner connection string.
READONLY_DSN = os.environ.get("FWA_READONLY_URL", "postgresql://fwa_reader@localhost:54329/fwa")


def assert_restricted(dsn: str) -> None:
    """Refuse to serve LLM-written SQL through a role that could read or change more than the API views."""
    with psycopg.connect(dsn) as conn:
        user, superuser, core, raw, create = conn.execute(
            """SELECT current_user, rolsuper, has_schema_privilege(current_user, 'core', 'USAGE'),
                      has_schema_privilege(current_user, 'raw', 'USAGE'),
                      has_schema_privilege(current_user, 'mart', 'CREATE')
               FROM pg_roles WHERE rolname = current_user""").fetchone()
    if superuser or core or raw or create:
        raise RuntimeError(f"FWA_READONLY_URL connects as {user!r}, which is not the restricted read-only role")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def _rows(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with psycopg.connect(READONLY_DSN, options="-c statement_timeout=5000") as conn:
        conn.read_only = True
        cur = conn.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def create_app(llm: LLM | None = None) -> FastAPI:
    app = FastAPI(title="fwa-signal", version="0.1.0",
                  description="Medicare Part D prescribing leads for program-integrity review. "
                              "Scores are statistical leads, not findings of wrongdoing.")
    assert_restricted(READONLY_DSN)
    model = llm or OllamaLLM()
    llm_slot = threading.Semaphore(1)  # one local-model call at a time

    @app.get("/health")
    def health():
        return {"ok": bool(_rows("SELECT 1 AS ok"))}

    @app.get("/providers/{npi}")
    def provider(npi: int):
        fx = facts(READONLY_DSN, npi)
        if fx is None:
            raise HTTPException(404, "NPI not found among scored individual prescribers")
        return fx

    @app.get("/providers/{npi}/summary")
    def provider_summary(npi: int):
        with llm_slot:
            s = summarize(model, READONLY_DSN, npi)
        if s is None:
            raise HTTPException(404, "NPI not found among scored individual prescribers")
        return s.__dict__

    @app.get("/leads")
    def leads(state: str | None = Query(None, min_length=2, max_length=2), specialty: str | None = None,
              by: Literal["risk", "outlier"] = "risk", limit: int = Query(50, ge=1, le=500)):
        order = "rank_logreg" if by == "risk" else "rank"
        return _rows(f"""SELECT npi, first_name, last_or_org_name, specialty, city, state, claims,
                                round(score, 2) AS outlier_score, rank AS outlier_rank,
                                rank_logreg AS risk_rank, reasons, excluded_but_prescribing
                         FROM mart.provider_risk
                         WHERE (%s::text IS NULL OR state = upper(%s)) AND (%s::text IS NULL OR specialty = %s)
                           AND {order} IS NOT NULL
                         ORDER BY {order} LIMIT %s""", (state, state, specialty, specialty, limit))

    @app.get("/compliance/excluded-but-prescribing")
    def excluded_but_prescribing():
        return _rows("""SELECT npi, first_name, last_or_org_name, specialty, state, claims, drug_cost
                        FROM mart.provider_risk WHERE excluded_but_prescribing ORDER BY claims DESC""")

    @app.post("/ask")
    def ask_question(q: Question):
        with llm_slot:
            a = ask(model, READONLY_DSN, q.question)
        return {"question": a.question, "sql": a.sql, "columns": a.columns, "rows": a.rows, "error": a.error,
                "latency_s": round(a.latency_s, 2)}

    return app


if os.environ.get("FWA_NO_APP") != "1":  # tests build their own app against a throwaway database
    app = create_app()
