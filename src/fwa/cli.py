"""fwa command line.

  fwa build --year 2022 --partd data/raw/partd_prescribers_2022.csv --leie data/raw/leie_UPDATED.csv
  fwa evaluate          # unsupervised scores vs later exclusions -> results/evaluation.json
  fwa model             # temporal-holdout models (needs the 2021 and 2022 databases) -> results/model_evaluation.json
  fwa eval-llm          # question answering, adversarial prompts, summaries -> results/llm_evaluation.json
  fwa serve             # FastAPI on 127.0.0.1:8000

Databases default to postgresql://fwa@localhost:54329/fwa (2022) and .../fwa2021; override with
FWA_DATABASE_URL / FWA_TRAIN_DATABASE_URL.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DB = os.environ.get("FWA_DATABASE_URL", "postgresql://fwa@localhost:54329/fwa")
TRAIN_DB = os.environ.get("FWA_TRAIN_DATABASE_URL", "postgresql://fwa@localhost:54329/fwa2021")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fwa")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--partd", type=Path, required=True)
    b.add_argument("--leie", type=Path, required=True)
    b.add_argument("--db", default=DB)
    sub.add_parser("evaluate")
    sub.add_parser("model")
    sub.add_parser("eval-llm")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    if a.cmd == "build":
        from .pipeline import build
        print(json.dumps(build(a.db, a.partd, a.leie), indent=1))
    elif a.cmd == "evaluate":
        from .evaluate import main as ev
        ev(DB, Path("results/evaluation.json"))
        print("wrote results/evaluation.json")
    elif a.cmd == "model":
        from .model import _fmt, run
        print(_fmt(run(TRAIN_DB, DB, Path("results/model_evaluation.json"))))
    elif a.cmd == "eval-llm":
        from .eval_llm import main as el
        el()
    else:
        import uvicorn
        uvicorn.run("fwa.api:app", host="127.0.0.1", port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
