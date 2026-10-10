-- Run ONCE as a Postgres superuser (or cloud admin) before migrations.
-- Creates the owner role (owns tables, runs migrations) and the app role
-- (used by API and worker; NOT owner, NOBYPASSRLS, so row-level security applies).
-- Passwords here are placeholders: set real ones from your secret store.

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mccp_owner') THEN
    CREATE ROLE mccp_owner LOGIN PASSWORD 'mccp_owner' NOSUPERUSER NOCREATEROLE;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mccp_app') THEN
    CREATE ROLE mccp_app LOGIN PASSWORD 'mccp_app' NOSUPERUSER NOCREATEROLE NOBYPASSRLS;
  END IF;
END
$$;
