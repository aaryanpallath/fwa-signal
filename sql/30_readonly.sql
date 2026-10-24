-- What the API and the question-answering feature may read, and a role that can only read it.
CREATE OR REPLACE VIEW mart.provider_risk AS
SELECT pp.*, r.risk_logreg, r.rank_logreg, r.risk_gbm, r.rank_gbm,
       EXISTS (SELECT 1 FROM mart.excluded_but_prescribing x WHERE x.npi = pp.npi) AS excluded_but_prescribing
FROM mart.provider_profile pp
LEFT JOIN mart.risk r USING (npi);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fwa_reader') THEN
        CREATE ROLE fwa_reader LOGIN;
    END IF;
END $$;

REVOKE ALL ON SCHEMA raw, core FROM fwa_reader;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA mart TO fwa_reader;
REVOKE ALL ON ALL TABLES IN SCHEMA mart FROM fwa_reader;
GRANT SELECT ON mart.provider_risk, mart.peer_stats, mart.peer_scores, mart.risk TO fwa_reader;  -- the last two back /providers
-- The read-only role is the last line of defence behind the SQL validator: even a query that slipped
-- through could not write, run long, or read other schemas.
ALTER ROLE fwa_reader SET default_transaction_read_only = on;
ALTER ROLE fwa_reader SET statement_timeout = '5s';
