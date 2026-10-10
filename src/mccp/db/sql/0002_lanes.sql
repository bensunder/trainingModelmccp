-- Work lanes (fixes S1: a long training run no longer blocks other jobs).
--   gpu      training-host stages: data staging, training, registration
--   serving  stages that need the vLLM adapter volume / private vLLM
--   control  pure control-plane stages: validation, policy, approvals
-- Must match mccp.domain.status.LANE (tested).

CREATE OR REPLACE FUNCTION mccp_lane(p_lifecycle text) RETURNS text
LANGUAGE sql IMMUTABLE AS $$
  SELECT CASE
    WHEN p_lifecycle IN ('queued','provisioning','running','paused','registering') THEN 'gpu'
    WHEN p_lifecycle IN ('publish_guard','evaluating','approved','canary','deployed','superseded') THEN 'serving'
    ELSE 'control'
  END
$$;

DROP FUNCTION IF EXISTS mccp_lease_job(text, integer);

-- Lease the next ready job whose current stage belongs to one of p_lanes.
CREATE OR REPLACE FUNCTION mccp_lease_job(p_owner text, p_lease_seconds integer, p_lanes text[])
RETURNS TABLE (job_id text, tenant_id uuid, attempts integer, lane text)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
  RETURN QUERY
  WITH pick AS (
    SELECT q2.job_id, mccp_lane(j.lifecycle_status) AS lane
      FROM job_queue q2
      JOIN jobs j ON j.id = q2.job_id
     WHERE q2.run_after <= now()
       AND (q2.lease_owner IS NULL OR q2.lease_expires_at < now())
       AND mccp_lane(j.lifecycle_status) = ANY(p_lanes)
     ORDER BY q2.run_after, q2.job_id
     FOR UPDATE OF q2 SKIP LOCKED
     LIMIT 1)
  UPDATE job_queue q
     SET lease_owner = p_owner,
         lease_expires_at = now() + make_interval(secs => p_lease_seconds),
         attempts = q.attempts + 1
    FROM pick
   WHERE q.job_id = pick.job_id
  RETURNING q.job_id, q.tenant_id, q.attempts, pick.lane;
END
$$;

REVOKE ALL ON FUNCTION mccp_lease_job(text, integer, text[]) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION mccp_lease_job(text, integer, text[]) TO mccp_app;
GRANT EXECUTE ON FUNCTION mccp_lane(text) TO mccp_app;
