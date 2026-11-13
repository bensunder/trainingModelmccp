import { lazy, Suspense, useEffect, useRef, useState, type FormEvent } from "react";
import { ApiError } from "./api/client";
import type { Org } from "./api/types";
import { AuthProvider, useAuth, useCan, useClient } from "./auth/AuthContext";
import { Button, ErrorNote, Field, Loading } from "./components/ui";
import { ROUTES, useHashRoute, type Route } from "./hooks/useHashRoute";
import { clearResourceCache, useResource } from "./hooks/useResource";

// One chunk per screen: the first paint ships only the shell and the overview.
const Overview = lazy(() => import("./features/overview/Overview"));
const Teams = lazy(() => import("./features/teams/Teams"));
const Budgets = lazy(() => import("./features/budgets/Budgets"));
const Guardrails = lazy(() => import("./features/guardrails/Guardrails"));
const RoutingScreen = lazy(() => import("./features/routing/Routing"));
const Admins = lazy(() => import("./features/admins/Admins"));
const Audit = lazy(() => import("./features/audit/Audit"));

const LABELS: Record<Route, string> = {
  overview: "Overview", teams: "Teams & keys", budgets: "Budgets", guardrails: "Guardrails", routing: "Routing",
  admins: "Administrators", audit: "Audit log",
};

export default function App() {
  return (
    <AuthProvider>
      <Gate />
    </AuthProvider>
  );
}

function Gate() {
  const { client } = useAuth();
  return client ? <Shell /> : <SignIn />;
}

export function SignIn() {
  const { signIn } = useAuth();
  const [token, setToken] = useState("");
  const [err, setErr] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      await signIn(token);
    } catch (x) {
      setErr(x instanceof ApiError ? (x.status === 401 ? new ApiError(401, "Not accepted", "that token is not valid or was revoked") : x) : new ApiError(0, "Error", String(x)));
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="signin">
      <form className="card" onSubmit={submit} aria-labelledby="signin-title">
        <h1 id="signin-title">Route control</h1>
        <p className="muted">Sign in with your administrator token (<code>ra_…</code>) or the platform operator token. It stays in this tab's memory only.</p>
        <ErrorNote error={err} />
        <Field label="Administrator token">
          {(id) => <input id={id} type="password" autoComplete="off" spellCheck={false} value={token} onChange={(e) => setToken(e.target.value)} required autoFocus />}
        </Field>
        <Button type="submit" busy={busy}>Sign in</Button>
      </form>
    </main>
  );
}

function Shell() {
  const client = useClient();
  const { signOut, me } = useAuth();
  const { operator, write } = useCan();
  const [route, go] = useHashRoute();
  const nav = useRef<HTMLElement>(null);
  useEffect(() => {
    // on narrow screens the section bar scrolls sideways: keep the current section in view
    nav.current?.querySelector<HTMLElement>('a[aria-current="page"]')?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
  }, [route]);
  const orgs = useResource<Org[]>("orgs", (s) => client.listOrgs(s));
  const [orgId, setOrgId] = useState<string | null>(null);
  useEffect(() => {
    if (!orgId && orgs.data?.length) setOrgId(orgs.data[0]!.id);
  }, [orgs.data, orgId]);
  const org = orgs.data?.find((o) => o.id === orgId) ?? null;
  return (
    <div className="shell">
      <header className="topbar">
        <span className="brand" aria-label="Route">route<span className="brand-dot">.</span></span>
        <nav aria-label="Sections" ref={nav}>
          <ul>
            {ROUTES.map((r) => (
              <li key={r}>
                <a href={`#/${r}`} aria-current={route === r ? "page" : undefined} onClick={(e) => { e.preventDefault(); go(r); }}>
                  {LABELS[r]}
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <div className="topbar-end">
          {me && <span className="who" title={me.kind === "operator" ? "Platform operator" : `Administrator of this organisation (${me.role})`}>
            {me.kind === "operator" ? "operator" : `${me.label} · ${me.role}`}
          </span>}
          {operator && orgs.data && orgs.data.length > 0 && (
            <label className="org-pick">
              <span className="sr-only">Organisation</span>
              <select value={orgId ?? ""} onChange={(e) => setOrgId(e.target.value)}>
                {orgs.data.map((o) => <option key={o.id} value={o.id}>{o.name}</option>)}
              </select>
            </label>
          )}
          <Button type="button" variant="ghost" onClick={() => { clearResourceCache(); signOut(); }}>Sign out</Button>
        </div>
      </header>
      <main className="content" id="main">
        <h1 className="page-title">{LABELS[route]}</h1>
        {!write && <p className="note" role="note">Read-only access: you can view everything in this organisation but not change it.</p>}
        <ErrorNote error={orgs.error} onRetry={orgs.refresh} />
        {orgs.loading && <Loading label="Loading organisations" />}
        {operator && orgs.data?.length === 0 && <NoOrgs onCreated={(o) => { orgs.mutate((p) => [...(p ?? []), o]); setOrgId(o.id); }} />}
        <Suspense fallback={<Loading />}>
          {route === "routing" && <RoutingScreen />}
          {org && route === "overview" && <Overview orgId={org.id} />}
          {org && route === "teams" && <Teams orgId={org.id} />}
          {org && route === "budgets" && <Budgets orgId={org.id} />}
          {org && route === "admins" && <Admins orgId={org.id} />}
          {org && route === "audit" && <Audit orgId={org.id} />}
          {org && route === "guardrails" && <Guardrails org={org} onSaved={(o) => orgs.mutate((p) => p?.map((x) => (x.id === o.id ? o : x)))} />}
        </Suspense>
      </main>
    </div>
  );
}

function NoOrgs({ onCreated }: { onCreated: (o: Org) => void }) {
  const client = useClient();
  const [name, setName] = useState("");
  const [err, setErr] = useState<ApiError | null>(null);
  return (
    <form className="card row" onSubmit={async (e) => {
      e.preventDefault();
      try { onCreated(await client.createOrg(name.trim())); } catch (x) { setErr(x instanceof ApiError ? x : new ApiError(0, "Error", String(x))); }
    }}>
      <ErrorNote error={err} />
      <Field label="Create your first organisation">{(id) => <input id={id} value={name} onChange={(e) => setName(e.target.value)} required maxLength={200} />}</Field>
      <Button type="submit">Create</Button>
    </form>
  );
}
