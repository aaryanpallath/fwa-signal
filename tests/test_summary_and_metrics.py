import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fwa.eval_llm import rows_match
from fwa.evaluate import auc, wilson
from fwa.summary import check, template

FACTS = {"prescribing_year": 2022, "npi": 1234567890, "name": "A B", "specialty": "Family Practice", "location": "Gary, IN",
         "claims": "2,000", "beneficiaries": "150", "drug_cost": "$95,000", "risk_rank": "12 of 1,000",
         "excluded_but_prescribing": False,
         "unusual_measures": [{"measure": "opioid share of claims", "value": "41.0%", "peer_median": "6.2%",
                               "peer_group": "Family Practice in IN", "peer_count": "812",
                               "higher_than_pct_of_peers": "99.6%"}]}


def test_summary_with_only_known_numbers_passes():
    text = ("A B, a Family Practice prescriber in Gary, IN, had 2,000 claims for 150 beneficiaries ($95,000). "
            "Their opioid share of claims was 41.0%, against a peer median of 6.2% among 812 peers.")
    assert check(text, FACTS) == []


def test_summary_with_invented_number_or_accusation_is_rejected():
    assert any("numbers not in the facts" in p for p in check("Opioid share was 55.0% of claims.", FACTS))
    assert any("accusatory" in p for p in check("This pattern suggests fraudulent prescribing.", FACTS))


def test_template_only_uses_facts():
    assert check(template(FACTS), FACTS) == []


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 5, 500).astype(float)
    y = (rng.random(500) < 0.2).astype(int)
    assert auc(list(s), list(y)) == pytest.approx(roc_auc_score(y, s))


def test_wilson_interval_contains_estimate():
    lo, hi = wilson(3, 1000)
    assert lo < 0.003 < hi and lo > 0


def test_rows_match_tolerates_extra_columns_not_wrong_values():
    assert rows_match([["TX"]], [["TX", 812]], ordered=True)
    assert not rows_match([["TX"]], [["CA", 812]], ordered=True)
    assert rows_match([[1], [2]], [[2], [1]], ordered=False)
    assert not rows_match([[1], [2]], [[2], [1]], ordered=True)
