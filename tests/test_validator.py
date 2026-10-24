import pytest

from fwa.nl2sql import MAX_ROWS, UnsafeSQL, validate


@pytest.mark.parametrize("sql", [
    "SELECT npi FROM mart.provider_risk WHERE state = 'IN' AND claims BETWEEN 10 AND 20 AND NOT excluded_but_prescribing",
    "SELECT specialty, avg(opioid_share) FROM mart.provider_risk GROUP BY specialty ORDER BY 2 DESC NULLS LAST",
    "WITH t AS (SELECT npi FROM mart.provider_risk) SELECT count(*) FROM t",
    "SELECT median FROM mart.peer_stats WHERE feature = 'opioid_share' AND state = '*'",
    "SELECT count(*) FROM MART.PROVIDER_RISK",
    "SELECT CASE WHEN claims > 100 THEN 'high' ELSE 'low' END, count(*) FROM mart.provider_risk GROUP BY 1",
])
def test_allowed(sql):
    out = validate(sql)
    assert "LIMIT" in out.upper()


@pytest.mark.parametrize("sql, why", [
    ("DROP TABLE mart.risk", "only SELECT"),
    ("SELECT 1; DELETE FROM mart.risk", "exactly one statement"),
    ("SELECT * FROM core.prescriber", "not allowed"),
    ("SELECT * FROM pg_catalog.pg_authid", "not allowed"),
    ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
    ("SELECT pg_sleep(30)", "pg_sleep"),
    ("SELECT set_config('statement_timeout', '0', false)", "set_config"),
    ("SELECT * FROM otherdb.mart.provider_risk", "cross-database"),
    ("SELECT * INTO mart.copy FROM mart.provider_risk", "not allowed"),
    ("", "cannot be answered"),
    ("this is not sql", "not valid SQL|only SELECT"),
])
def test_denied(sql, why):
    with pytest.raises(UnsafeSQL, match=why):
        validate(sql)


def test_limit_is_capped():
    assert f"LIMIT {MAX_ROWS}" in validate("SELECT npi FROM mart.provider_risk LIMIT 100000")
    assert "LIMIT 5" in validate("SELECT npi FROM mart.provider_risk LIMIT 5")
