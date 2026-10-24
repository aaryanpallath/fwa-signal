-- Typed core tables from the raw text loads. CMS suppresses small counts (blank cells); those become NULL.
DROP SCHEMA IF EXISTS core CASCADE;
CREATE SCHEMA core;

CREATE FUNCTION core.num(t text) RETURNS numeric LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE WHEN t ~ '^-?[0-9]+(\.[0-9]+)?$' THEN t::numeric END
$$;

CREATE TABLE core.prescriber AS
SELECT
    "Prscrbr_NPI"::bigint                         AS npi,
    NULLIF("Prscrbr_Last_Org_Name", '')           AS last_or_org_name,
    NULLIF("Prscrbr_First_Name", '')              AS first_name,
    NULLIF("Prscrbr_Crdntls", '')                 AS credentials,
    "Prscrbr_Ent_Cd"                              AS entity_code,     -- I individual, O organization
    NULLIF("Prscrbr_City", '')                    AS city,
    NULLIF("Prscrbr_State_Abrvtn", '')            AS state,
    NULLIF("Prscrbr_Type", '')                    AS specialty,
    core.num("Tot_Clms")::int                     AS claims,
    core.num("Tot_Benes")::int                    AS beneficiaries,
    core.num("Tot_Drug_Cst")                      AS drug_cost,
    core.num("Tot_Day_Suply")::bigint             AS day_supply,
    core.num("Brnd_Tot_Clms")::int                AS brand_claims,
    core.num("Opioid_Tot_Clms")::int              AS opioid_claims,
    core.num("Opioid_LA_Tot_Clms")::int           AS opioid_la_claims,
    core.num("Antpsyct_GE65_Tot_Clms")::int       AS antipsychotic_65_claims,
    core.num("GE65_Tot_Benes")::int               AS beneficiaries_65,
    core.num("Bene_Avg_Risk_Scre")                AS avg_risk_score
FROM raw.partd_prescribers;
ALTER TABLE core.prescriber ADD PRIMARY KEY (npi);

-- LEIE: keep rows with a real 10-digit NPI (most individual exclusions have none and cannot be linked).
CREATE TABLE core.exclusion AS
SELECT
    "NPI"::bigint                                                    AS npi,
    to_date("EXCLDATE", 'YYYYMMDD')                                  AS excluded_on,
    "EXCLTYPE"                                                       AS exclusion_type,
    CASE WHEN "REINDATE" ~ '^[0-9]{8}$' AND "REINDATE" <> '00000000'
         THEN to_date("REINDATE", 'YYYYMMDD') END                    AS reinstated_on,
    NULLIF("GENERAL", '')                                            AS general,
    NULLIF("SPECIALTY", '')                                          AS leie_specialty
FROM raw.leie
WHERE "NPI" ~ '^[0-9]{10}$' AND "NPI" <> '0000000000'
  AND "EXCLDATE" ~ '^[0-9]{8}$';
CREATE INDEX ON core.exclusion (npi);
