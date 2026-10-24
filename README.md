# fwa-signal — Medicare Part D prescribing leads with a validated LLM layer

A fraud, waste and abuse (FWA) analytics service built on two public government datasets. It loads
**1.33 million Medicare Part D prescribers** (CMS) and the **HHS-OIG exclusion list** into PostgreSQL,
scores how unusual each prescriber is against their specialty-and-state peers, and **tests whether those
scores predict who OIG excludes later**. A FastAPI service lets analysts ask questions in plain English
(the LLM writes SQL that is validated and run read-only) and read case summaries whose every number is
checked against the database.

> Scores here are **statistical leads, not findings of wrongdoing**. Unusual prescribing often has
> legitimate clinical reasons, and most high-scoring prescribers in the evaluation below were never
> excluded. This repository therefore contains **only aggregate results**: no lists of named or
> NPI-identified providers.

## Results

### 1. Do the scores predict later exclusion? (out-of-time test)
Scores are computed from **2022** prescribing. A prescriber counts as a positive if OIG excluded them on
or after **2023-01-01**: 538 of 1,023,331 scored individual prescribers, a base rate of **0.053%**.
Models were trained only on **2021** prescribing with 2022 exclusions as labels (90 positives), so
nothing from the test period was used ([`results/model_evaluation.json`](results/model_evaluation.json)).

| ranking | AUC | top 100 | top 1,000 | top 10,000 |
|---|---:|---:|---:|---:|
| logistic regression (specialty, state, volume, ratios, peer scores) | **0.651** | 0 | 1 (1.9×) | 14 (2.7×) |
| gradient boosting (same features) | 0.626 | 0 | 3 (5.7×) | 23 (4.4×) |
| peer-outlier composite (unsupervised) | 0.582 | 0 | 3 (5.7×) | 20 (3.8×) |
| prescribing volume (total claims) | 0.576 | 2 (38×) | **5 (9.5×)** | **28 (5.3×)** |
| raw opioid share | 0.548 | 0 | 1 (1.9×) | 9 (1.7×) |
| random | 0.500 | 0.05 | 0.53 | 5.3 |

Cells show the number of later-excluded prescribers in each top-k, with lift over random in parentheses.

* **Logistic regression ranks the whole population better than volume**: AUC +0.076, stratified
  bootstrap 95% CI [+0.035, +0.107]. Gradient boosting is +0.050 [+0.012, +0.084].
* **No method beats raw prescribing volume at the very top of the list**, which is where reviewers
  actually work. With 538 positives, the top-1,000 counts are small and their intervals overlap.
* Specialty is the strongest predictor: it has the highest permutation importance in gradient
  boosting. Interventional and general pain management are excluded at roughly 6–10× the base rate.
  The large per-state coefficients rest on very few positives and should not be read as real
  regional effects.
* Logistic regression was not chosen as "the" model in advance; that it scored best is an
  observation, not a pre-registered result.

### 2. The first version failed, and why
The first peer-outlier score used median/MAD robust z-scores and scored **AUC 0.459**, worse than random.
Two things were wrong:
* **A statistics bug.** For zero-inflated rates (in 89% of peer groups most prescribers have *no*
  long-acting opioid claims) the MAD is 0, so any tiny value produced an enormous z-score. **4% of all
  z-scores hit the cap of 20.** The fix replaced z-scores with a distribution-free measure:
  −log₁₀(share of peers at or above the prescriber). That moved the unsupervised AUC from 0.459 to
  **0.582**.
* **A design mismatch.** Comparing prescribers only within their specialty removes the between-specialty
  signal that exclusions are concentrated in, which is why the supervised models, which can use
  specialty, do better.

### 3. Compliance check
Prescribers excluded before 2022, never reinstated, but still billing Part D in 2022: **2**
(in 2021 data: 33). These are reported as an exact rule, not a score.

### 4. LLM features ([`results/llm_evaluation.json`](results/llm_evaluation.json))
A local model (gemma4 through Ollama; no data leaves the machine). Median 2.9 s per question on an Apple M4.

| check | result |
|---|---|
| plain-English questions → SQL, execution accuracy on 24 questions with reference queries | **20 / 24 (83%)** |
| … as first graded, before correcting 4 reference queries (see below) | 16 / 24 (67%) |
| 10 adversarial prompts (drop / delete / update, read `pg_authid` or files, `pg_sleep`, `COPY`, privilege grants) | 9 refused by the SQL validator; 1 became a harmless count query; **data unchanged** |
| 15 case summaries for the highest-risk prescribers | **15 / 15** passed the fact check (every number traced to the database, no accusatory language) |

Grading notes: after the first run I found that 4 of my reference answers were wrong, not the model.
Three returned an extra count column the question never asked for, and one silently excluded
prescribers with no opioid claims. I fixed those references and re-graded the SQL the model had
already produced. Both scores and every change are recorded. The 4 remaining misses are genuine model
errors:
* a missing state filter;
* "Psychiatrist" instead of the data's value "Psychiatry";
* ignoring an instruction to sort NULLs last;
* not excluding national rows from a peer-group count.

## How it works

```
CMS Part D by Provider (2021, 2022) ─┐  COPY          sql/10_core.sql           sql/20_mart.sql
OIG LEIE exclusion list ─────────────┴─▶ raw.* ──▶ core.prescriber / exclusion ──▶ mart.features (6 ratios)
                                                                                     mart.peer_stats (median, MAD)
                                         quality gates (row counts, NPI format,      mart.peer_scores (peer position)
                                         numeric parsing, ratio bounds) ─ stop ─▶    mart.scores · excluded_but_prescribing
                                                                                     mart.risk (model scores)
FastAPI ──▶ /leads /providers/{npi} /compliance/... ──┐
        └─▶ /ask: question ─▶ LLM ─▶ sqlglot validator ─▶ run as fwa_reader (read-only, 5 s timeout, 200 rows)
        └─▶ /providers/{npi}/summary: facts ─▶ LLM ─▶ fact check ─▶ text, or template fallback
```

* **Features:**
  * opioid share of claims;
  * long-acting share of opioid claims;
  * drug cost per beneficiary;
  * brand-name share;
  * days of supply per claim;
  * antipsychotic claims per beneficiary aged 65+.

  Only individual prescribers with at least 50 claims are scored. CMS-suppressed small counts stay NULL.
* **Peers:** the same specialty in the same state when there are at least 30 of them, otherwise the
  same specialty nationally.
* **Defence in depth for LLM-written SQL:** the API refuses to start unless it connects as the restricted
  reader. The validator allows one `SELECT` over two allow-listed
  views and a short list of functions, and caps rows. Behind it, `fwa_reader` can only `SELECT` those
  views, with `default_transaction_read_only` and `statement_timeout` set. Question text only ever
  becomes part of the prompt, never part of the SQL.
* **Case summaries:** the LLM gets a small JSON of facts. Any number in its text that is not in those
  facts, or any accusatory wording, triggers a deterministic template instead, so a reviewer never sees
  an unverified figure.

## Setup and usage

Requires PostgreSQL 16, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) with a model
(`FWA_LLM_MODEL`, default `gemma4`).

```bash
uv sync
# data (public): CMS "Medicare Part D Prescribers - by Provider" for 2021 and 2022 from data.cms.gov,
# and the OIG LEIE "UPDATED.csv" from oig.hhs.gov/exclusions, into data/raw/
createdb fwa && createdb fwa2021
uv run fwa build --partd data/raw/partd_prescribers_2021.csv --leie data/raw/leie_UPDATED.csv --db postgresql://fwa@localhost:54329/fwa2021
uv run fwa build --partd data/raw/partd_prescribers_2022.csv --leie data/raw/leie_UPDATED.csv
uv run fwa evaluate     # unsupervised scores vs later exclusions
uv run fwa model        # temporal-holdout models; also creates the read-only role and API views
uv run fwa eval-llm     # question answering, adversarial prompts, summaries
uv run fwa serve        # http://127.0.0.1:8000/docs (FWA_READONLY_URL must be the fwa_reader role; checked at startup)
uv run pytest -q        # unit + end-to-end tests (PostgreSQL on FWA_TEST_ADMIN_URL)
```

Each year's build takes about 90 s on an M4 laptop (loading 1.3 M rows takes 6 s).

## Design decisions
* **Evaluate against something real.** Outlier detection is easy to demo and hard to validate. OIG
  exclusions are a delayed, imperfect proxy (they include license revocations and non-prescribing
  offences), but they are an outcome that happened after the data. That turned a demo into a measurable
  question, and exposed the first version's failure.
* **Strict temporal holdout.** Training labels had to be known before the test predictions were made,
  which is why training has only 90 positives. Widening the label window would have leaked the future.
* **Distribution-free peer scores.** Real prescribing rates are skewed and zero-inflated; percentile
  position needs no normality or symmetry assumptions.
* **The LLM never touches data directly.** It proposes SQL or prose; deterministic code decides whether
  either is shown.
* **No identified providers in the repository.** Names stay in the local database.

## Limitations
* Exclusion is a weak label. Many exclusions have nothing to do with prescribing, and many unusual
  prescribers are never investigated. Precision is very low in absolute terms (at best about 1 in 200
  in a top-1,000 list).
* Only 538 positives in the test year and 90 in training, so all intervals are wide.
* LEIE rows without an NPI (about 89% of all exclusions) cannot be linked to prescribers.
* The question-answering evaluation set is small (24 questions) and written by me; the model is a
  general local model, not tuned for SQL.
* Not deployed to a cloud provider; the service runs locally. It has no authentication, so it must
  not be exposed beyond localhost as is.

## Testing
29 tests:
* the SQL validator (allowed and blocked queries, including the adversarial cases);
* the summary fact check and template;
* AUC (checked against scikit-learn, with ties), Wilson intervals and the grading function;
* end-to-end tests that build a synthetic dataset with the real CMS and LEIE column layout in a fresh
  PostgreSQL database. They check that a planted outlier ranks first, that the excluded-but-prescribing
  rule ignores reinstated providers, that a duplicate NPI trips a quality gate, that the read-only role
  cannot read other schemas or write, and every API endpoint with a scripted LLM.

CI runs lint and all tests against a PostgreSQL 16 service container on every push.

## Data & licensing
CMS Medicare Part D Prescribers (data.cms.gov) and the OIG List of Excluded Individuals/Entities
(oig.hhs.gov) are U.S. government public data. Raw data is not committed. Code is MIT licensed.

## AI assistance disclosure
Built with AI assistance (Claude Code). The out-of-time evaluation is what exposed the failed first
version and the z-score bug; the reference-query corrections are documented above, with both grades
kept. Every number in this README comes from the committed result files.
