// Contract mirrored from the gateway's admin API (/admin/v1). Keep in sync with mccp.route.app.

export type Iso = string;

export interface Org {
  id: string;
  name: string;
  created_at: Iso;
  guardrails: Partial<GuardrailPolicy>;
  default_profile: string | null;
  rpm_per_key: number;
}

export interface Team {
  id: string;
  org_id: string;
  name: string;
  created_at: Iso;
  approved_models: string[] | null; // null: every catalog model
  default_profile: string | null;
}

export interface ApiKey {
  id: string;
  org_id: string;
  team_id: string;
  label: string;
  prefix: string;
  created_at: Iso;
  revoked_at: Iso | null;
  default_profile: string | null;
}

export interface CreatedKey extends ApiKey {
  key: string; // plaintext, shown once
}

export type Scope = "org" | "team" | "key";
export type Period = "day" | "month";

export interface Budget {
  id: string;
  org_id: string;
  scope: Scope;
  scope_id: string;
  period: Period;
  limit_usd: number;
  hard: boolean;
}

export type GuardAction = "off" | "flag" | "redact" | "block";

export interface GuardrailPolicy {
  secrets: GuardAction;
  pii: GuardAction;
  injection: GuardAction;
}

export interface Usage {
  since: Iso;
  until?: Iso;
  requests: number;
  errors: number;
  responses_without_usage: number;
  spend_usd: number;
  baseline_usd: number;
  saved_usd: number;
  saved_pct: number | null;
  tokens: { input: number; output: number; cache_read: number; cache_write: number };
  cache_hit_rate: number | null;
  by_model: Record<string, { requests: number; cost_usd: number; tokens: number }>;
  by_day: Record<string, { cost_usd: number; requests: number }>;
  by_day_model: Record<string, Record<string, number>>;
  by_team: Record<string, number>;
  by_task: Record<string, number>;
}

export interface CatalogModel {
  id: string;
  provider: string;
  tier: "frontier" | "strong" | "fast";
  wires: string[];
  price: { input_per_m: number; output_per_m: number; cache_read_per_m: number | null; cache_write_per_m: number | null };
  price_source: string;
  context_window: number;
  provider_ready: boolean;
}

export interface CatalogProfile {
  id: string;
  name: string;
  strategy: "cost" | "balanced" | "intelligence";
  candidates: string[];
  fallback: string;
  background_model: string | null;
}

export interface Catalog {
  version: string;
  baseline_model: string;
  default_profile: string;
  models: CatalogModel[];
  profiles: CatalogProfile[];
}

export interface Routing {
  catalog_version: string;
  evidence: { suite: string; generated_at: string };
  profiles: Record<string, { strategy: string; routes: Record<string, { model: string | null; reason: string }> }>;
}

export interface Problem {
  type: string;
  title: string;
  status: number;
  detail: string;
  [extra: string]: unknown;
}
