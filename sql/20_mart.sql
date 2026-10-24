-- Analytics marts: per-prescriber ratios, peer statistics, robust z-scores and a composite score.
DROP SCHEMA IF EXISTS mart CASCADE;
CREATE SCHEMA mart;

-- 1) Ratios for individual prescribers with enough volume for a rate to mean something.
CREATE TABLE mart.features AS
SELECT npi, specialty, state,
       opioid_claims::numeric / NULLIF(claims, 0)                   AS opioid_share,
       opioid_la_claims::numeric / NULLIF(opioid_claims, 0)         AS long_acting_opioid_share,
       drug_cost / NULLIF(beneficiaries, 0)                         AS cost_per_beneficiary,
       brand_claims::numeric / NULLIF(claims, 0)                    AS brand_share,
       day_supply::numeric / NULLIF(claims, 0)                      AS days_per_claim,
       antipsychotic_65_claims::numeric / NULLIF(beneficiaries_65, 0) AS antipsychotic_65_per_bene
FROM core.prescriber
WHERE entity_code = 'I' AND claims >= 50 AND specialty IS NOT NULL AND state IS NOT NULL;
ALTER TABLE mart.features ADD PRIMARY KEY (npi);

-- 2) Long format, one row per prescriber and feature (NULL ratios = suppressed or not applicable).
CREATE TABLE mart.feature_long AS
SELECT f.npi, f.specialty, f.state, v.feature, v.value
FROM mart.features f
CROSS JOIN LATERAL (VALUES
    ('opioid_share', f.opioid_share),
    ('long_acting_opioid_share', f.long_acting_opioid_share),
    ('cost_per_beneficiary', f.cost_per_beneficiary),
    ('brand_share', f.brand_share),
    ('days_per_claim', f.days_per_claim),
    ('antipsychotic_65_per_bene', f.antipsychotic_65_per_bene)
) AS v(feature, value)
WHERE v.value IS NOT NULL;

-- 3) Peer statistics: median and MAD within specialty x state, and within specialty nationally
--    (used when the state peer group is smaller than 30).
CREATE TABLE mart.peer_stats AS
WITH med AS (
    SELECT specialty, state, feature, count(*) AS n,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY value) AS median
    FROM mart.feature_long GROUP BY specialty, state, feature
    UNION ALL
    SELECT specialty, '*' AS state, feature, count(*),
           percentile_cont(0.5) WITHIN GROUP (ORDER BY value)
    FROM mart.feature_long GROUP BY specialty, feature
), dev AS (
    SELECT m.specialty, m.state, m.feature,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(l.value - m.median)) AS mad,
           avg(abs(l.value - m.median)) AS meanad
    FROM med m
    JOIN mart.feature_long l
      ON l.specialty = m.specialty AND l.feature = m.feature AND (m.state = '*' OR l.state = m.state)
    GROUP BY m.specialty, m.state, m.feature
)
SELECT m.specialty, m.state, m.feature, m.n, m.median, d.mad, d.meanad
FROM med m JOIN dev d USING (specialty, state, feature);
CREATE INDEX ON mart.peer_stats (specialty, state, feature);

-- 4) Peer position per feature: the fraction of peers whose value is at or above this prescriber's
--    (state peer group when it has at least 30 members, otherwise the specialty nationally), and the
--    "surprise" -log10 of that fraction (2 = top 1%, 3 = top 0.1%). This is distribution-free, so it
--    stays meaningful for zero-inflated rates (most peers have no long-acting opioid or antipsychotic
--    claims) where median/MAD z-scores blow up (an earlier version capped 4% of all z-scores at 20).
CREATE TABLE mart.peer_scores AS
WITH w AS (
    SELECT npi, specialty, state, feature, value,
           count(*) OVER (PARTITION BY specialty, state, feature) AS n_state,
           count(*) OVER (PARTITION BY specialty, feature) AS n_nat,
           cume_dist() OVER (PARTITION BY specialty, state, feature ORDER BY value DESC) AS tail_state,
           cume_dist() OVER (PARTITION BY specialty, feature ORDER BY value DESC) AS tail_nat
    FROM mart.feature_long
)
SELECT w.npi, w.feature, w.value,
       CASE WHEN w.n_state >= 30 THEN 'state' ELSE 'national' END AS peer_group,
       CASE WHEN w.n_state >= 30 THEN w.n_state ELSE w.n_nat END AS peer_n,
       CASE WHEN w.n_state >= 30 THEN w.tail_state ELSE w.tail_nat END AS share_at_or_above,
       -log(CASE WHEN w.n_state >= 30 THEN w.tail_state ELSE w.tail_nat END) AS surprise,
       COALESCE(s.median, n.median) AS peer_median
FROM w
LEFT JOIN mart.peer_stats s ON w.n_state >= 30 AND s.specialty = w.specialty AND s.state = w.state AND s.feature = w.feature
JOIN mart.peer_stats n ON n.specialty = w.specialty AND n.state = '*' AND n.feature = w.feature
WHERE w.n_nat >= 30;  -- specialties too small nationally are not scored
CREATE INDEX ON mart.peer_scores (npi);

-- 5) Composite: mean of a prescriber's two largest surprise values (one extreme ratio can be an
--    artefact; two unusual ratios together are a stronger signal). Ties broken by NPI for stability.
CREATE TABLE mart.scores AS
WITH ranked AS (
    SELECT npi, feature, surprise, share_at_or_above, peer_n,
           row_number() OVER (PARTITION BY npi ORDER BY surprise DESC) AS r
    FROM mart.peer_scores
)
SELECT npi,
       round((sum(surprise) FILTER (WHERE r <= 2) / 2)::numeric, 3) AS score,
       -- a reason = top 1% of peers, or the single highest value when the group has under 100 members
       array_agg(feature ORDER BY surprise DESC)
           FILTER (WHERE share_at_or_above <= GREATEST(0.01, 1.0 / peer_n)) AS reasons
FROM ranked GROUP BY npi;
ALTER TABLE mart.scores ADD PRIMARY KEY (npi);
ALTER TABLE mart.scores ADD COLUMN rank int;
UPDATE mart.scores s SET rank = r.rk
FROM (SELECT npi, row_number() OVER (ORDER BY score DESC, npi) AS rk FROM mart.scores) r WHERE r.npi = s.npi;
CREATE INDEX ON mart.scores (rank);

-- 6) Compliance check: excluded before the claims year, not reinstated, yet prescribing in 2022.
CREATE TABLE mart.excluded_but_prescribing AS
SELECT p.npi, p.last_or_org_name, p.first_name, p.specialty, p.state, p.claims, p.drug_cost,
       e.excluded_on, e.exclusion_type
FROM core.prescriber p
JOIN core.exclusion e ON e.npi = p.npi
WHERE e.excluded_on < DATE '2022-01-01'
  AND (e.reinstated_on IS NULL OR e.reinstated_on > DATE '2022-12-31');

-- 7) One wide row per scored prescriber: what the API and the question-answering feature read.
CREATE VIEW mart.provider_profile AS
SELECT p.npi, p.last_or_org_name, p.first_name, p.credentials, p.city, p.state, p.specialty,
       p.claims, p.beneficiaries, p.drug_cost,
       f.opioid_share, f.long_acting_opioid_share, f.cost_per_beneficiary, f.brand_share,
       f.days_per_claim, f.antipsychotic_65_per_bene,
       s.score, s.rank, s.reasons
FROM core.prescriber p
JOIN mart.features f USING (npi)
LEFT JOIN mart.scores s USING (npi);
