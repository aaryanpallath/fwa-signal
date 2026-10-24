"""Load the public files into PostgreSQL, build the marts, and enforce data-quality gates.

raw  : both CSVs loaded verbatim (every column text) with COPY
core : typed tables (sql/10_core.sql)
mart : features, peer statistics, robust z-scores, composite score, compliance check (sql/20_mart.sql)
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[2]
SQL = ROOT / "sql"


class QualityError(RuntimeError):
    pass


def load_raw(conn: psycopg.Connection, table: str, csv_path: Path, encoding: str = "latin-1") -> int:
    with open(csv_path, encoding=encoding, newline="") as f:
        header = next(csv.reader(f))
    cols = ", ".join(f'"{c}" text' for c in header)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS raw.{table} CASCADE")
        cur.execute(f"CREATE TABLE raw.{table} ({cols})")
        with open(csv_path, "rb") as f, cur.copy(
                f"COPY raw.{table} FROM STDIN WITH (FORMAT csv, HEADER true, ENCODING '{encoding.upper()}')") as cp:
            while chunk := f.read(1 << 20):
                cp.write(chunk)
        cur.execute(f"SELECT count(*) FROM raw.{table}")
        return cur.fetchone()[0]


def checks(conn: psycopg.Connection) -> dict:
    """Data-quality gates. Any failure raises; results are returned for the run report."""
    q = lambda sql: conn.execute(sql).fetchone()[0]
    res = {
        "raw_prescribers": q("SELECT count(*) FROM raw.partd_prescribers"),
        "core_prescribers": q("SELECT count(*) FROM core.prescriber"),
        "npi_not_10_digits": q("SELECT count(*) FROM core.prescriber WHERE npi NOT BETWEEN 1000000000 AND 9999999999"),
        "claims_unparsed": q("SELECT count(*) FROM core.prescriber WHERE claims IS NULL"),
        "ratios_out_of_range": q("""SELECT count(*) FROM mart.features WHERE opioid_share > 1 OR brand_share > 1
                                    OR long_acting_opioid_share > 1 OR opioid_share < 0"""),
        "scored_prescribers": q("SELECT count(*) FROM mart.scores"),
        "peer_shares_out_of_range": q("SELECT count(*) FROM mart.peer_scores WHERE share_at_or_above <= 0 OR share_at_or_above > 1"),
        "exclusions_with_npi": q("SELECT count(*) FROM core.exclusion"),
        "excluded_but_prescribing": q("SELECT count(*) FROM mart.excluded_but_prescribing"),
    }
    failures = []
    if res["core_prescribers"] != res["raw_prescribers"]:
        failures.append("row count changed between raw and core (duplicate or dropped NPIs)")
    if res["npi_not_10_digits"]:
        failures.append(f"{res['npi_not_10_digits']} invalid NPIs")
    if res["claims_unparsed"]:
        failures.append(f"{res['claims_unparsed']} prescribers with unparseable total claims")
    if res["ratios_out_of_range"]:
        failures.append(f"{res['ratios_out_of_range']} share ratios outside [0, 1]")
    if res["peer_shares_out_of_range"]:
        failures.append("peer shares outside (0, 1]")
    if res["scored_prescribers"] < 0.5 * res["core_prescribers"]:
        failures.append("fewer than half of prescribers were scored")
    if failures:
        raise QualityError("; ".join(failures))
    return res


def build(dsn: str, partd_csv: Path, leie_csv: Path) -> dict:
    timings: dict[str, float] = {}
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
        t = time.monotonic()
        load_raw(conn, "partd_prescribers", partd_csv)
        load_raw(conn, "leie", leie_csv)
        timings["load_s"] = round(time.monotonic() - t, 1)
        for f in (SQL / "10_core.sql", SQL / "20_mart.sql"):  # 30_readonly runs after the model (needs mart.risk)
            t = time.monotonic()
            conn.execute(f.read_text())
            timings[f"{f.stem}_s"] = round(time.monotonic() - t, 1)
        res = checks(conn)
    return {**res, "timings": timings}
