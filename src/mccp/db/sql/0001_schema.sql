-- MCCP schema v1. Applied by the owner role (mccp_owner).
-- Every tenant-scoped table has RLS keyed on current_setting('app.tenant_id').

CREATE OR REPLACE FUNCTION mccp_current_tenant() RETURNS uuid
LANGUAGE sql STABLE AS $$
  SELECT NULLIF(current_setting('app.tenant_id', true), '')::uuid
$$;

CREATE TABLE tenants (
  id             uuid PRIMARY KEY,
  name           text NOT NULL UNIQUE,
  idp_tenant_id  text NOT NULL UNIQUE,
  region         text NOT NULL DEFAULT 'us',
  created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE projects (
  tenant_id   uuid NOT NULL REFERENCES tenants(id),
  name        text NOT NULL CHECK (name ~ '^[a-z][a-z0-9-]{0,62}$'),
  created_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, name)
);

-- Recipes are platform-wide and admin-curated (not tenant data).
CREATE TABLE recipes (
  ref         text PRIMARY KEY,
  body        jsonb NOT NULL,
  body_sha256 text NOT NULL,
  created_by  text NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE dataset_versions (
  tenant_id        uuid NOT NULL REFERENCES tenants(id),
  project          text NOT NULL,
  name             text NOT NULL,
  version          integer NOT NULL CHECK (version >= 1),
  ref              text NOT NULL,
  manifest_sha256  text NOT NULL,
  artifact_prefix  text NOT NULL,
  card             jsonb NOT NULL,
  created_by       text NOT NULL,
  created_at       timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, project, name, version),
  FOREIGN KEY (tenant_id, project) REFERENCES projects(tenant_id, name)
);

CREATE TABLE eval_suites (
  tenant_id    uuid NOT NULL REFERENCES tenants(id),
  project      text NOT NULL,
  name         text NOT NULL,
  version      integer NOT NULL CHECK (version >= 1),
  ref          text NOT NULL,
  body         jsonb NOT NULL,
  body_sha256  text NOT NULL,
  created_by   text NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, project, name, version),
  FOREIGN KEY (tenant_id, project) REFERENCES projects(tenant_id, name)
);

CREATE TABLE jobs (
  id                  text PRIMARY KEY CHECK (id ~ '^ftjob-[0-9a-f]{32}$'),
  tenant_id           uuid NOT NULL REFERENCES tenants(id),
  project             text NOT NULL,
  idempotency_key     text NOT NULL CHECK (length(idempotency_key) BETWEEN 8 AND 255),
  request_hash        text NOT NULL,
  request             jsonb NOT NULL,
  resolved            jsonb NOT NULL,
  config_hash         text NOT NULL,
  lifecycle_status    text NOT NULL CHECK (lifecycle_status IN (
                        'created','validating','queued','provisioning','running','paused',
                        'registering','publish_guard','evaluating','policy_check',
                        'pending_approval','approved','canary','deployed','superseded',
                        'retired','rejected','failed','cancelled')),
  risk_tier           text NOT NULL CHECK (risk_tier IN ('low','medium','high')),
  output_adapter      text NOT NULL,
  adapter_version_id  text,
  submitted_by        text NOT NULL,
  trained_tokens      bigint,
  gpu_seconds         double precision,
  cost_usd            double precision,
  flop_estimate       double precision,
  error               jsonb,
  stage_state         jsonb NOT NULL DEFAULT '{}'::jsonb,
  cancel_requested    boolean NOT NULL DEFAULT false,
  pause_requested     boolean NOT NULL DEFAULT false,
  created_at          timestamptz NOT NULL DEFAULT now(),
  updated_at          timestamptz NOT NULL DEFAULT now(),
  finished_at         timestamptz,
  UNIQUE (tenant_id, idempotency_key),
  FOREIGN KEY (tenant_id, project) REFERENCES projects(tenant_id, name)
);
CREATE INDEX jobs_tenant_status_idx ON jobs (tenant_id, lifecycle_status, id DESC);

-- Durable work queue (one row per job that a worker should drive now or later).
CREATE TABLE job_queue (
  job_id            text PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
  tenant_id         uuid NOT NULL REFERENCES tenants(id),
  run_after         timestamptz NOT NULL DEFAULT now(),
  lease_owner       text,
  lease_expires_at  timestamptz,
  attempts          integer NOT NULL DEFAULT 0,
  last_error        text
);
CREATE INDEX job_queue_ready_idx ON job_queue (run_after) WHERE lease_owner IS NULL;

CREATE TABLE job_metrics (
  tenant_id  uuid NOT NULL REFERENCES tenants(id),
  job_id     text NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  step       integer NOT NULL,
  metrics    jsonb NOT NULL,
  ts         timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (job_id, step)
);

CREATE TABLE job_checkpoints (
  tenant_id  uuid NOT NULL REFERENCES tenants(id),
  job_id     text NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  step       integer NOT NULL,
  path       text NOT NULL,
  ts         timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (job_id, step)
);

CREATE TABLE adapter_versions (
  id                text PRIMARY KEY,
  tenant_id         uuid NOT NULL REFERENCES tenants(id),
  project           text NOT NULL,
  name              text NOT NULL,
  version           integer NOT NULL CHECK (version >= 1),
  serving_name      text NOT NULL,
  artifact_uri      text NOT NULL,
  artifact_sha256   text NOT NULL,
  signature_uri     text NOT NULL,
  card              jsonb NOT NULL,
  status            text NOT NULL CHECK (status IN ('pending','candidate','challenger','champion','superseded','retired','rejected')),
  job_id            text NOT NULL REFERENCES jobs(id),
  created_at        timestamptz NOT NULL DEFAULT now(),
  updated_at        timestamptz NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, project, name, version),
  UNIQUE (serving_name)
);

CREATE TABLE adapter_aliases (
  tenant_id   uuid NOT NULL REFERENCES tenants(id),
  project     text NOT NULL,
  name        text NOT NULL,
  alias       text NOT NULL CHECK (alias IN ('candidate','challenger','champion')),
  version_id  text NOT NULL REFERENCES adapter_versions(id),
  updated_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, project, name, alias)
);

CREATE TABLE eval_reports (
  id                  text PRIMARY KEY,
  tenant_id           uuid NOT NULL REFERENCES tenants(id),
  adapter_version_id  text NOT NULL REFERENCES adapter_versions(id),
  suite_ref           text NOT NULL,
  report              jsonb NOT NULL,
  report_sha256       text NOT NULL,
  created_at          timestamptz NOT NULL DEFAULT now()
);

-- One immutable row per policy decision. `sha256` is the digest of the exact
-- OPA input (bundle + target + approvals); `bundle_sha256` identifies the evidence.
CREATE TABLE evidence_bundles (
  sha256         text NOT NULL,
  bundle_sha256  text NOT NULL,
  tenant_id      uuid NOT NULL REFERENCES tenants(id),
  job_id         text NOT NULL REFERENCES jobs(id),
  target_env     text NOT NULL CHECK (target_env IN ('staging','production')),
  body           jsonb NOT NULL,
  decision       jsonb NOT NULL,
  created_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, sha256)
);
CREATE INDEX evidence_job_idx ON evidence_bundles (tenant_id, job_id, created_at DESC);

CREATE TABLE approvals (
  id                  text PRIMARY KEY,
  tenant_id           uuid NOT NULL REFERENCES tenants(id),
  job_id              text NOT NULL REFERENCES jobs(id),
  adapter_version_id  text NOT NULL REFERENCES adapter_versions(id),
  approver            text NOT NULL,
  decision            text NOT NULL CHECK (decision IN ('approve','reject')),
  justification       text NOT NULL CHECK (length(justification) BETWEEN 3 AND 4000),
  evidence_sha256     text NOT NULL,
  created_at          timestamptz NOT NULL DEFAULT now(),
  UNIQUE (job_id, approver)
);

CREATE TABLE deployments (
  tenant_id               uuid NOT NULL REFERENCES tenants(id),
  project                 text NOT NULL,
  logical_name            text NOT NULL,
  champion_version_id     text REFERENCES adapter_versions(id),
  challenger_version_id   text REFERENCES adapter_versions(id),
  challenger_weight       integer NOT NULL DEFAULT 0 CHECK (challenger_weight BETWEEN 0 AND 100),
  previous_champion_id    text REFERENCES adapter_versions(id),
  updated_at              timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, project, logical_name)
);

CREATE TABLE feedback (
  id                  text PRIMARY KEY,
  tenant_id           uuid NOT NULL REFERENCES tenants(id),
  adapter_serving_name text NOT NULL,
  logical_name        text,
  request_id          text,
  rating              smallint NOT NULL CHECK (rating IN (-1, 1)),
  comment             text CHECK (comment IS NULL OR length(comment) <= 4000),
  payload             jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_by          text NOT NULL,
  created_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX feedback_adapter_idx ON feedback (adapter_serving_name, created_at);

-- Append-only, hash-chained audit log (NFR-3).
CREATE TABLE events (
  tenant_id       uuid NOT NULL REFERENCES tenants(id),
  seq             bigint NOT NULL CHECK (seq >= 1),
  ts              timestamptz NOT NULL,
  actor           text NOT NULL,
  action          text NOT NULL,
  resource_type   text NOT NULL,
  resource_id     text NOT NULL,
  data            jsonb NOT NULL,
  prev_event_sha  text NOT NULL,
  event_sha       text NOT NULL,
  PRIMARY KEY (tenant_id, seq)
);
CREATE INDEX events_resource_idx ON events (tenant_id, resource_type, resource_id, seq);

CREATE OR REPLACE FUNCTION mccp_events_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'events are append-only';
END
$$;
CREATE TRIGGER events_no_update BEFORE UPDATE OR DELETE ON events
  FOR EACH ROW EXECUTE FUNCTION mccp_events_immutable();

-- ---------------------------------------------------------------- RLS
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['projects','dataset_versions','eval_suites','jobs','job_queue',
                           'job_metrics','job_checkpoints','adapter_versions','adapter_aliases',
                           'eval_reports','evidence_bundles','approvals','deployments','feedback','events']
  LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format(
      'CREATE POLICY tenant_isolation ON %I USING (tenant_id = mccp_current_tenant()) '
      'WITH CHECK (tenant_id = mccp_current_tenant())', t);
  END LOOP;
END
$$;
ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_self ON tenants USING (id = mccp_current_tenant());

-- Cross-tenant entry points, owned by mccp_owner (table owner => not subject
-- to RLS). They expose only what the caller needs.

-- Map an IdP tenant claim to the internal tenant id.
CREATE OR REPLACE FUNCTION mccp_resolve_tenant(p_idp_tenant text) RETURNS uuid
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
  SELECT id FROM tenants WHERE idp_tenant_id = p_idp_tenant
$$;

-- Lease the next ready job across all tenants (FOR UPDATE SKIP LOCKED).
CREATE OR REPLACE FUNCTION mccp_lease_job(p_owner text, p_lease_seconds integer)
RETURNS TABLE (job_id text, tenant_id uuid, attempts integer)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
  RETURN QUERY
  UPDATE job_queue q
     SET lease_owner = p_owner,
         lease_expires_at = now() + make_interval(secs => p_lease_seconds),
         attempts = q.attempts + 1
   WHERE q.job_id = (
          SELECT q2.job_id FROM job_queue q2
           WHERE q2.run_after <= now()
             AND (q2.lease_owner IS NULL OR q2.lease_expires_at < now())
           ORDER BY q2.run_after, q2.job_id
           FOR UPDATE SKIP LOCKED
           LIMIT 1)
  RETURNING q.job_id, q.tenant_id, q.attempts;
END
$$;

-- Supersede sweep needs to see aged versions across tenants.
CREATE OR REPLACE FUNCTION mccp_due_retirements(p_older_than_s integer)
RETURNS TABLE (job_id text, tenant_id uuid)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
  SELECT j.id, j.tenant_id FROM jobs j
   WHERE j.lifecycle_status = 'superseded'
     AND j.updated_at < now() - make_interval(secs => p_older_than_s)
$$;

REVOKE ALL ON FUNCTION mccp_resolve_tenant(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION mccp_lease_job(text, integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION mccp_due_retirements(integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION mccp_resolve_tenant(text) TO mccp_app;
GRANT EXECUTE ON FUNCTION mccp_lease_job(text, integer) TO mccp_app;
GRANT EXECUTE ON FUNCTION mccp_due_retirements(integer) TO mccp_app;
GRANT EXECUTE ON FUNCTION mccp_current_tenant() TO mccp_app;

-- Least privilege for the app role.
GRANT SELECT, INSERT ON tenants, projects, recipes, dataset_versions, eval_suites,
  job_metrics, job_checkpoints, eval_reports, evidence_bundles, approvals, feedback TO mccp_app;
GRANT SELECT, INSERT, UPDATE ON jobs, adapter_versions, adapter_aliases, deployments TO mccp_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON job_queue TO mccp_app;
GRANT SELECT, INSERT ON events TO mccp_app;
GRANT UPDATE (body, body_sha256) ON recipes TO mccp_app;
