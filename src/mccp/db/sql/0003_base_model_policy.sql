-- Model profiles: per-tenant base-model governance.
-- `base_model_policy.allowed_origins` (ISO 3166-1 alpha-2 list) restricts which
-- publisher countries a tenant's adapters may be built on; enforced by the
-- promotion policy. Empty object = no restriction (previous behaviour).
ALTER TABLE tenants ADD COLUMN base_model_policy jsonb NOT NULL DEFAULT '{}'::jsonb;
GRANT UPDATE (base_model_policy) ON tenants TO mccp_app;
