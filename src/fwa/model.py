"""Supervised exclusion-risk model with a strict temporal holdout.

    train: 2021 prescribing  ->  label = excluded by OIG during 2022
    test : 2022 prescribing  ->  label = excluded by OIG on or after 2023-01-01

Providers already excluded before a window starts are removed from that window (they cannot be
"later" excluded). Hyperparameters are fixed up front; early stopping uses a slice of the 2021 data
only, so nothing is tuned on the test year. The test year is scored once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluate import KS, wilson

RATIOS = ["opioid_share", "long_acting_opioid_share", "cost_per_beneficiary", "brand_share", "days_per_claim",
          "antipsychotic_65_per_bene"]
VOLUME = ["log_claims", "log_beneficiaries", "log_drug_cost"]
ZS = [f"surprise_{r}" for r in RATIOS]
NUMERIC = VOLUME + RATIOS + ZS + ["avg_risk_score"]
CATEGORICAL = ["specialty", "state"]


def load_year(dsn: str, label_from: str, label_to: str | None) -> pd.DataFrame:
    """Features for one prescribing year plus the exclusion label for the following window."""
    to_clause = f"AND e.excluded_on <= DATE '{label_to}'" if label_to else ""
    sql = f"""
    SELECT f.npi, f.specialty, f.state, p.claims, p.beneficiaries, p.drug_cost, p.avg_risk_score,
           {", ".join(f"f.{r}" for r in RATIOS)},
           {", ".join(f"max(z.surprise) FILTER (WHERE z.feature = '{r}') AS surprise_{r}" for r in RATIOS)},
           EXISTS (SELECT 1 FROM core.exclusion e WHERE e.npi = f.npi
                   AND e.excluded_on >= DATE '{label_from}' {to_clause}) AS y,
           EXISTS (SELECT 1 FROM core.exclusion e WHERE e.npi = f.npi
                   AND e.excluded_on < DATE '{label_from}'
                   AND (e.reinstated_on IS NULL OR e.reinstated_on >= DATE '{label_from}')) AS already_excluded
    FROM mart.features f
    JOIN core.prescriber p USING (npi)
    LEFT JOIN mart.peer_scores z USING (npi)
    GROUP BY f.npi, f.specialty, f.state, p.claims, p.beneficiaries, p.drug_cost, p.avg_risk_score,
             {", ".join(f"f.{r}" for r in RATIOS)}
    """
    with psycopg.connect(dsn) as conn:
        cur = conn.execute(sql)
        df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    df = df[~df["already_excluded"]].drop(columns="already_excluded").copy()
    for c in ["claims", "beneficiaries", "drug_cost"]:
        df[f"log_{c}"] = np.log1p(df[c].astype(float))
    for c in RATIOS + ZS + ["avg_risk_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["y"] = df["y"].astype(int)
    return df


def gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        categorical_features="from_dtype", random_state=7)


def logreg():
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), CATEGORICAL),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1))


def _as_categorical(df: pd.DataFrame, cats: dict[str, list[str]]) -> pd.DataFrame:
    out = df[NUMERIC + CATEGORICAL].copy()
    for c in CATEGORICAL:
        out[c] = pd.Categorical(out[c].where(out[c].isin(cats[c]), "Other"), categories=cats[c] + ["Other"])
    return out


def topk(scores: np.ndarray, y: np.ndarray, base: float, rng: np.random.Generator) -> dict:
    order = np.lexsort((rng.random(len(scores)), -scores))  # random tie-break
    res = {}
    for k in KS:
        hits = int(y[order[:k]].sum())
        lo, hi = wilson(hits, k)
        res[f"top_{k}"] = {"hits": hits, "precision": round(hits / k, 5), "precision_95ci": [round(lo, 5), round(hi, 5)],
                           "lift": round(hits / k / base, 2), "recall": round(hits / y.sum(), 4)}
    return res


def bootstrap_auc_diff(y, a, b, n=200, seed=7) -> dict:
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = []
    for _ in range(n):  # stratified: keep the number of positives fixed
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return {"mean": round(float(np.mean(diffs)), 4), "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                                                             round(float(np.percentile(diffs, 97.5)), 4)]}


def run(dsn_train: str, dsn_test: str, out: Path) -> dict:
    train = load_year(dsn_train, "2022-01-01", "2022-12-31")
    test = load_year(dsn_test, "2023-01-01", None)
    cats = {c: sorted(train[c].value_counts()[lambda s: s >= 200].index.tolist()) for c in CATEGORICAL}
    Xtr, Xte = _as_categorical(train, cats), _as_categorical(test, cats)
    ytr, yte = train["y"].to_numpy(), test["y"].to_numpy()
    base = yte.mean()
    rng = np.random.default_rng(7)

    g = gbm().fit(Xtr, ytr)
    lr = logreg().fit(train[NUMERIC + CATEGORICAL], ytr)
    preds = {
        "gradient boosting (all features)": g.predict_proba(Xte)[:, 1],
        "logistic regression (all features)": lr.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1],
        "baseline: prescribing volume": test["claims"].to_numpy(float),
        "baseline: raw opioid share": test["opioid_share"].fillna(-1).to_numpy(),
        "baseline: peer composite (unsupervised)": test[ZS].apply(
            lambda r: np.mean(sorted(r.dropna(), reverse=True)[:2]) if r.notna().any() else 0, axis=1).to_numpy(),
    }
    res = {"train": {"year": 2021, "label": "excluded in 2022", "n": len(train), "positives": int(ytr.sum())},
           "test": {"year": 2022, "label": "excluded 2023-01-01 or later", "n": len(test), "positives": int(yte.sum()),
                    "base_rate": round(float(base), 6)},
           "gbm_iterations": int(g.n_iter_), "models": {}}
    for name, s in preds.items():
        res["models"][name] = {"auc": round(float(roc_auc_score(yte, s)), 4),
                               "average_precision": round(float(average_precision_score(yte, s)), 5),
                               **topk(s, yte, base, rng)}
    res["auc_difference_gbm_minus_volume"] = bootstrap_auc_diff(
        yte, preds["gradient boosting (all features)"], preds["baseline: prescribing volume"])
    res["auc_difference_logreg_minus_volume"] = bootstrap_auc_diff(
        yte, preds["logistic regression (all features)"], preds["baseline: prescribing volume"])
    # Persist both models' 2022 risk scores for the API (rank 1 = highest risk).
    with psycopg.connect(dsn_test, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS mart.risk")
        conn.execute("CREATE TABLE mart.risk (npi bigint PRIMARY KEY, risk_logreg double precision, "
                     "risk_gbm double precision, rank_logreg int, rank_gbm int)")
        df = pd.DataFrame({"npi": test["npi"].astype("int64"), "lr": preds["logistic regression (all features)"],
                           "gb": preds["gradient boosting (all features)"]})
        df["rl"] = df["lr"].rank(ascending=False, method="first").astype(int)
        df["rg"] = df["gb"].rank(ascending=False, method="first").astype(int)
        with conn.cursor() as cur, cur.copy("COPY mart.risk FROM STDIN") as cp:
            for r in df.itertuples(index=False):
                cp.write_row((int(r.npi), float(r.lr), float(r.gb), int(r.rl), int(r.rg)))
        conn.execute("CREATE INDEX ON mart.risk (rank_logreg)")
        conn.execute((Path(__file__).resolve().parents[2] / "sql" / "30_readonly.sql").read_text())

    # What the boosted model relies on (permutation importance on a stratified test subsample).
    sub = np.concatenate([np.flatnonzero(yte == 1), rng.choice(np.flatnonzero(yte == 0), 100_000, replace=False)])
    pi = permutation_importance(g, Xte.iloc[sub], yte[sub], scoring="roc_auc", n_repeats=5, random_state=7)
    res["gbm_permutation_importance_auc_drop"] = dict(sorted(
        {c: round(float(m), 4) for c, m in zip(Xte.columns, pi.importances_mean)}.items(), key=lambda kv: -kv[1])[:8])
    coefs = lr[-1].coef_[0]
    names = lr[0].get_feature_names_out()
    res["logreg_largest_coefficients"] = {n.split("__", 1)[1]: round(float(c), 3) for n, c in
                                          sorted(zip(names, coefs), key=lambda nc: -abs(nc[1]))[:10]}
    out.write_text(json.dumps(res, indent=2) + "\n")
    return res


def _fmt(res: dict) -> str:
    lines = [f"test base rate {res['test']['base_rate']:.5%} ({res['test']['positives']} of {res['test']['n']:,})"]
    for name, m in res["models"].items():
        lines.append(f"{name:42} AUC {m['auc']:.3f}  AP {m['average_precision']:.4f}  "
                     + "  ".join(f"top{k[4:]}: {m[k]['hits']} ({m[k]['lift']}x)" for k in ("top_100", "top_1000", "top_10000")))
    return "\n".join(lines)


if __name__ == "__main__":
    r = run("postgresql://fwa@localhost:54329/fwa2021", "postgresql://fwa@localhost:54329/fwa",
            Path("results/model_evaluation.json"))
    print(_fmt(r))
    print("AUC diff GBM - volume:", r["auc_difference_gbm_minus_volume"])
    print("AUC diff LR - volume:", r["auc_difference_logreg_minus_volume"])
    print("GBM permutation importance:", r["gbm_permutation_importance_auc_drop"])
    print("LR largest coefficients:", r["logreg_largest_coefficients"])
