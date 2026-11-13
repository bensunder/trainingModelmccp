import type {
  AdminRole, ApiKey, AuditPage, Budget, Catalog, CreatedAdmin, CreatedKey, GuardrailPolicy, Me, Org, OrgAdmin, Period,
  Problem, Routing, Scope, Team, Usage,
} from "./types";

/** An error the admin API reported (RFC 9457 problem+json), or a transport failure. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly title: string,
    readonly detail: string,
    readonly problem?: Problem,
  ) {
    super(`${title}: ${detail}`);
    this.name = "ApiError";
  }
}

export type Fetch = typeof fetch;
const TIMEOUT_MS = 15_000;

export function createClient(token: string, fetchImpl: Fetch = fetch.bind(globalThis), base = "") {
  async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(new DOMException("timeout", "TimeoutError")), TIMEOUT_MS);
    const onAbort = () => ctrl.abort(signal?.reason);
    signal?.addEventListener("abort", onAbort, { once: true });
    let res: Response;
    try {
      res = await fetchImpl(`${base}${path}`, {
        method,
        signal: ctrl.signal,
        headers: { authorization: `Bearer ${token}`, ...(body !== undefined ? { "content-type": "application/json" } : {}) },
        body: body !== undefined ? JSON.stringify(body) : undefined,
      });
    } catch (e) {
      if (signal?.aborted) throw e; // caller cancelled: not an error to show
      const timeout = e instanceof DOMException && (e.name === "TimeoutError" || ctrl.signal.reason?.name === "TimeoutError");
      throw new ApiError(0, timeout ? "Timed out" : "Network error", timeout ? "the gateway did not answer in time" : "could not reach the gateway");
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    }
    if (res.status === 204) return undefined as T;
    const text = await res.text();
    const data: unknown = text ? safeJson(text) : undefined;
    if (!res.ok) {
      const p = isProblem(data) ? data : undefined;
      throw new ApiError(res.status, p?.title ?? res.statusText ?? "Error", p?.detail ?? `HTTP ${res.status}`, p);
    }
    return data as T;
  }

  const org = (id: string) => `/admin/v1/orgs/${encodeURIComponent(id)}`;
  return {
    me: (s?: AbortSignal) => request<Me>("GET", "/admin/v1/me", undefined, s),
    admins: (id: string, s?: AbortSignal) => request<OrgAdmin[]>("GET", `${org(id)}/admins`, undefined, s),
    createAdmin: (id: string, label: string, role: AdminRole) => request<CreatedAdmin>("POST", `${org(id)}/admins`, { label, role }),
    revokeAdmin: (id: string, adminId: string) => request<OrgAdmin>("DELETE", `${org(id)}/admins/${encodeURIComponent(adminId)}`),
    audit: (id: string, before?: string | null, s?: AbortSignal) =>
      request<AuditPage>("GET", `${org(id)}/audit?limit=100${before ? `&before=${encodeURIComponent(before)}` : ""}`, undefined, s),
    listOrgs: (s?: AbortSignal) => request<Org[]>("GET", "/admin/v1/orgs", undefined, s),
    createOrg: (name: string) => request<Org>("POST", "/admin/v1/orgs", { name }),
    patchOrg: (id: string, patch: Partial<Pick<Org, "name" | "default_profile" | "rpm_per_key">>) => request<Org>("PATCH", org(id), patch),
    putGuardrails: (id: string, p: GuardrailPolicy) => request<Org>("PUT", `${org(id)}/guardrails`, p),
    usage: (id: string, since?: string, s?: AbortSignal) =>
      request<Usage>("GET", `${org(id)}/usage${since ? `?since=${encodeURIComponent(since)}` : ""}`, undefined, s),
    teams: (id: string, s?: AbortSignal) => request<Team[]>("GET", `${org(id)}/teams`, undefined, s),
    createTeam: (id: string, name: string, approved_models: string[] | null) => request<Team>("POST", `${org(id)}/teams`, { name, approved_models }),
    patchTeam: (id: string, teamId: string, patch: Partial<Pick<Team, "approved_models" | "default_profile">>) =>
      request<Team>("PATCH", `${org(id)}/teams/${encodeURIComponent(teamId)}`, patch),
    keys: (id: string, s?: AbortSignal) => request<ApiKey[]>("GET", `${org(id)}/keys`, undefined, s),
    createKey: (id: string, teamId: string, label: string) => request<CreatedKey>("POST", `${org(id)}/teams/${encodeURIComponent(teamId)}/keys`, { label }),
    revokeKey: (id: string, keyId: string) => request<ApiKey>("DELETE", `${org(id)}/keys/${encodeURIComponent(keyId)}`),
    budgets: (id: string, s?: AbortSignal) => request<Budget[]>("GET", `${org(id)}/budgets`, undefined, s),
    putBudget: (id: string, b: { scope: Scope; scope_id: string; period: Period; limit_usd: number; hard: boolean }) =>
      request<Budget>("PUT", `${org(id)}/budgets`, b),
    deleteBudget: (id: string, budgetId: string) => request<void>("DELETE", `${org(id)}/budgets/${encodeURIComponent(budgetId)}`),
    catalog: (s?: AbortSignal) => request<Catalog>("GET", "/admin/v1/catalog", undefined, s),
    routing: (s?: AbortSignal) => request<Routing>("GET", "/admin/v1/routing", undefined, s),
  };
}

export type Client = ReturnType<typeof createClient>;

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function isProblem(v: unknown): v is Problem {
  return typeof v === "object" && v !== null && "title" in v && "status" in v;
}
