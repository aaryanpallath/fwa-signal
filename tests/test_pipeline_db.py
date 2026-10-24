"""End-to-end test on a small synthetic dataset with the real CMS/LEIE column layout.

Needs PostgreSQL: FWA_TEST_ADMIN_URL (default postgresql://fwa@localhost:54329/postgres). Creates and drops a
throwaway database. Skipped when no server is reachable.
"""
import csv
import os
import random
import uuid
from pathlib import Path

import psycopg
import pytest

ADMIN = os.environ.get("FWA_TEST_ADMIN_URL", "postgresql://fwa@localhost:54329/postgres")
PARTD_COLS = ["Prscrbr_NPI", "Prscrbr_Last_Org_Name", "Prscrbr_First_Name", "Prscrbr_Crdntls", "Prscrbr_Ent_Cd",
              "Prscrbr_City", "Prscrbr_State_Abrvtn", "Prscrbr_Type", "Tot_Clms", "Tot_Benes", "Tot_Drug_Cst",
              "Tot_Day_Suply", "Brnd_Tot_Clms", "Opioid_Tot_Clms", "Opioid_LA_Tot_Clms", "Antpsyct_GE65_Tot_Clms",
              "GE65_Tot_Benes", "Bene_Avg_Risk_Scre"]
LEIE_COLS = ["LASTNAME", "FIRSTNAME", "MIDNAME", "BUSNAME", "GENERAL", "SPECIALTY", "UPIN", "NPI", "DOB", "ADDRESS",
             "CITY", "STATE", "ZIP", "EXCLTYPE", "EXCLDATE", "REINDATE", "WAIVERDATE", "WVRSTATE"]
OUTLIER = 1999999999
EXCLUDED_ACTIVE = 1888888888
EXCLUDED_REINSTATED = 1777777777


def _server_up() -> bool:
    try:
        psycopg.connect(ADMIN, connect_timeout=2).close()
        return True
    except psycopg.Error:
        return False


pytestmark = [pytest.mark.db, pytest.mark.skipif(not _server_up(), reason="no PostgreSQL server")]


def _write_fixtures(tmp_path):
    rng = random.Random(3)
    partd = tmp_path / "partd.csv"
    with partd.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PARTD_COLS)
        npi = 1000000000
        for spec in ("Family Practice", "Dentist"):
            for state in ("IN", "OH"):
                for _ in range(60):
                    npi += 1
                    claims = rng.randint(80, 900)
                    opioid = int(claims * rng.uniform(0.01, 0.08)) if spec == "Family Practice" else int(claims * 0.1)
                    w.writerow([npi, f"L{npi}", "F", "MD", "I", "Town", state, spec, claims, rng.randint(20, 200),
                                round(claims * rng.uniform(40, 90), 2), claims * 30, int(claims * 0.2), opioid,
                                "" if opioid < 11 else int(opioid * 0.1), "", "", "1.1"])
        # one extreme prescriber (opioid share and cost far above peers), two excluded ones
        w.writerow([OUTLIER, "Extreme", "X", "MD", "I", "Town", "IN", "Family Practice", 800, 40, 900000, 24000, 600,
                    700, 300, "", "", "1.0"])
        for n in (EXCLUDED_ACTIVE, EXCLUDED_REINSTATED):
            w.writerow([n, "Exc", "E", "MD", "I", "Town", "OH", "Family Practice", 200, 50, 9000, 6000, 40, 8, "", "",
                        "", "1.0"])
        w.writerow([1555555555, "Org", "", "", "O", "Town", "IN", "Pharmacy", 5000, 900, 1, 1, 1, 1, "", "", "", ""])
    leie = tmp_path / "leie.csv"
    with leie.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LEIE_COLS)
        base = {c: "" for c in LEIE_COLS}
        for npi, excl, rein in ((EXCLUDED_ACTIVE, "20190505", "00000000"), (EXCLUDED_REINSTATED, "20180101", "20200101"),
                                (0, "20240101", "00000000")):  # NPI 0 = unlinkable
            w.writerow([{**base, "NPI": str(npi), "EXCLTYPE": "1128b4", "EXCLDATE": excl, "REINDATE": rein}[c]
                        for c in LEIE_COLS])
    return partd, leie


@pytest.fixture
def db(tmp_path):
    name = f"fwa_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(ADMIN, autocommit=True) as c:
        c.execute(f"CREATE DATABASE {name}")
    dsn = ADMIN.rsplit("/", 1)[0] + "/" + name
    try:
        yield dsn, *_write_fixtures(tmp_path)
    finally:
        with psycopg.connect(ADMIN, autocommit=True) as c:
            c.execute(f"DROP DATABASE {name} WITH (FORCE)")


def test_pipeline_gates_scores_and_compliance(db):
    from fwa.pipeline import build

    dsn, partd, leie = db
    res = build(dsn, partd, leie)
    assert res["core_prescribers"] == res["raw_prescribers"] == 244
    assert res["exclusions_with_npi"] == 2           # the NPI-0 row cannot be linked
    with psycopg.connect(dsn) as c:
        top = c.execute("SELECT npi, reasons FROM mart.scores ORDER BY rank LIMIT 1").fetchone()
        assert top[0] == OUTLIER and "opioid_share" in top[1]
        assert c.execute("SELECT count(*) FROM mart.scores WHERE npi = 1555555555").fetchone()[0] == 0  # orgs unscored
        flagged = [r[0] for r in c.execute("SELECT npi FROM mart.excluded_but_prescribing")]
        assert flagged == [EXCLUDED_ACTIVE]          # the reinstated provider is not flagged
        shares = c.execute("SELECT min(share_at_or_above), max(share_at_or_above) FROM mart.peer_scores").fetchone()
        assert 0 < shares[0] and shares[1] <= 1


def test_quality_gate_stops_bad_data(db, tmp_path):
    from fwa.pipeline import QualityError, build

    dsn, partd, leie = db
    bad = tmp_path / "bad.csv"
    rows = partd.read_text().splitlines()
    rows.append(rows[1])  # duplicate NPI
    bad.write_text("\n".join(rows) + "\n")
    with pytest.raises(Exception) as e:
        build(dsn, bad, leie)
    assert isinstance(e.value, (QualityError, psycopg.Error))


def test_read_only_role_and_api(db):
    from fastapi.testclient import TestClient

    from fwa import api as api_mod
    from fwa.llm import ScriptedLLM
    from fwa.pipeline import build

    dsn, partd, leie = db
    build(dsn, partd, leie)
    with psycopg.connect(dsn, autocommit=True) as c:   # stand-in for the model's risk table
        c.execute("CREATE TABLE mart.risk AS SELECT npi, score::float AS risk_logreg, rank AS rank_logreg, "
                  "score::float AS risk_gbm, rank AS rank_gbm FROM mart.scores")
        c.execute((Path(__file__).resolve().parents[1] / "sql" / "30_readonly.sql").read_text())
    ro = dsn.replace("://fwa@", "://fwa_reader@")
    with psycopg.connect(ro) as c, pytest.raises(psycopg.errors.InsufficientPrivilege):
        c.execute("SELECT count(*) FROM core.prescriber")
    with psycopg.connect(ro) as c, pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        c.execute("CREATE TABLE mart.x (a int)")

    api_mod.READONLY_DSN = ro
    llm = ScriptedLLM(['{"sql": "SELECT count(*) FROM mart.provider_risk WHERE state = \'IN\'"}',
                       '{"sql": "DELETE FROM mart.risk"}',
                       "Extreme X, a Family Practice prescriber in Town, IN, had 800 claims.",
                       "This looks like fraud."])
    client = TestClient(api_mod.create_app(llm))
    assert client.get("/health").json() == {"ok": True}
    assert client.get("/leads", params={"by": "outlier", "limit": 1}).json()[0]["npi"] == OUTLIER
    assert [r["npi"] for r in client.get("/compliance/excluded-but-prescribing").json()] == [EXCLUDED_ACTIVE]
    assert client.post("/ask", json={"question": "how many in Indiana"}).json()["rows"] == [[121]]
    assert client.post("/ask", json={"question": "delete risk"}).json()["error"].startswith("refused")
    ok = client.get(f"/providers/{OUTLIER}/summary").json()
    assert ok["source"] == "llm"
    bad = client.get(f"/providers/{OUTLIER}/summary").json()
    assert bad["source"] == "template" and "accusatory" in bad["problems"][0]
    assert client.get("/providers/1").status_code == 404
