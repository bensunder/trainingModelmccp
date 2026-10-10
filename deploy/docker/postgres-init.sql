-- Runs once on first container start (docker-entrypoint-initdb.d).
CREATE ROLE mccp_owner LOGIN PASSWORD 'mccp_owner';
CREATE ROLE mccp_app LOGIN PASSWORD 'mccp_app' NOBYPASSRLS;
CREATE DATABASE mccp OWNER mccp_owner;
GRANT CONNECT ON DATABASE mccp TO mccp_app;
CREATE ROLE litellm LOGIN PASSWORD 'litellm';
CREATE DATABASE litellm OWNER litellm;
CREATE ROLE mlflow LOGIN PASSWORD 'mlflow';
CREATE DATABASE mlflow OWNER mlflow;
