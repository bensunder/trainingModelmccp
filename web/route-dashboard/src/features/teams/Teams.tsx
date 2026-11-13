import { useState, type FormEvent } from "react";
import { ApiError } from "../../api/client";
import type { ApiKey, Catalog, CreatedKey, Team } from "../../api/types";
import { useCan, useClient } from "../../auth/AuthContext";
import { Button, Card, DataTable, Dialog, Empty, ErrorNote, Field, Loading } from "../../components/ui";
import { useResource } from "../../hooks/useResource";
import { date } from "../../utils/format";

export default function TeamsContainer({ orgId }: { orgId: string }) {
  const client = useClient();
  const { write } = useCan();
  const teams = useResource<Team[]>(`teams:${orgId}`, (s) => client.teams(orgId, s));
  const keys = useResource<ApiKey[]>(`keys:${orgId}`, (s) => client.keys(orgId, s));
  const catalog = useResource<Catalog>("catalog", (s) => client.catalog(s));
  const [actionError, setActionError] = useState<ApiError | null>(null);
  const [created, setCreated] = useState<CreatedKey | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  async function act<T>(label: string, fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(label);
    setActionError(null);
    try {
      return await fn();
    } catch (e) {
      setActionError(e instanceof ApiError ? e : new ApiError(0, "Error", String(e)));
      return undefined;
    } finally {
      setBusy(null);
    }
  }

  return (
    <TeamsView
      teams={teams.data}
      keys={keys.data}
      models={catalog.data?.models.map((m) => m.id) ?? []}
      profiles={catalog.data?.profiles.map((p) => p.id) ?? []}
      loading={teams.loading || keys.loading}
      canWrite={write}
      error={teams.error ?? keys.error ?? actionError}
      busy={busy}
      created={created}
      onDismissCreated={() => setCreated(null)}
      onCreateTeam={async (name, approved) => {
        const t = await act("team", () => client.createTeam(orgId, name, approved));
        if (t) teams.mutate((prev) => [...(prev ?? []), t].sort((a, b) => a.name.localeCompare(b.name)));
      }}
      onSaveTeam={async (team, approved, profile) => {
        const prev = teams.data;
        teams.mutate((list) => list?.map((t) => (t.id === team.id ? { ...t, approved_models: approved, default_profile: profile } : t))); // optimistic
        const t = await act(`save:${team.id}`, () => client.patchTeam(orgId, team.id, { approved_models: approved, default_profile: profile }));
        if (!t) teams.mutate(() => prev); // roll back
      }}
      onCreateKey={async (teamId, label) => {
        const k = await act(`key:${teamId}`, () => client.createKey(orgId, teamId, label));
        if (k) {
          setCreated(k);
          const { key: _secret, ...stored } = k;
          keys.mutate((prev) => [...(prev ?? []), stored]);
        }
      }}
      onRevoke={async (k) => {
        if (!window.confirm(`Revoke ${k.label} (${k.prefix}…)? Requests with it stop immediately.`)) return;
        const r = await act(`revoke:${k.id}`, () => client.revokeKey(orgId, k.id));
        if (r) keys.mutate((prev) => prev?.map((x) => (x.id === r.id ? r : x)));
      }}
    />
  );
}

interface ViewProps {
  teams: Team[] | undefined;
  keys: ApiKey[] | undefined;
  models: string[];
  profiles: string[];
  loading: boolean;
  canWrite: boolean;
  error: ApiError | null;
  busy: string | null;
  created: CreatedKey | null;
  onDismissCreated: () => void;
  onCreateTeam: (name: string, approved: string[] | null) => Promise<void>;
  onSaveTeam: (team: Team, approved: string[] | null, profile: string | null) => Promise<void>;
  onCreateKey: (teamId: string, label: string) => Promise<void>;
  onRevoke: (k: ApiKey) => Promise<void>;
}

export function TeamsView(p: ViewProps) {
  return (
    <>
      <ErrorNote error={p.error} />
      {p.loading && <Loading label="Loading teams" />}
      {p.canWrite && <NewTeamForm busy={p.busy === "team"} onCreate={p.onCreateTeam} />}
      {p.teams?.length === 0 && <Empty>No teams yet. Create one to issue API keys.</Empty>}
      {p.teams?.map((t) => (
        <TeamCard key={t.id} {...p} team={t} keys={(p.keys ?? []).filter((k) => k.team_id === t.id)} />
      ))}
      <Dialog open={p.created !== null} title="New API key" onClose={p.onDismissCreated}>
        <p>Copy it now. It is shown once and cannot be recovered; a lost key is revoked and replaced.</p>
        <output className="secret" aria-label="API key">{p.created?.key}</output>
        <div className="row">
          <Button type="button" variant="ghost" onClick={() => p.created && navigator.clipboard?.writeText(p.created.key)}>
            Copy
          </Button>
          <Button type="button" onClick={p.onDismissCreated}>I stored it</Button>
        </div>
      </Dialog>
    </>
  );
}

function NewTeamForm({ busy, onCreate }: { busy: boolean; onCreate: (name: string, approved: string[] | null) => Promise<void> }) {
  const [name, setName] = useState("");
  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    await onCreate(name.trim(), null);
    setName("");
  }
  return (
    <Card title="New team">
      <form className="row" onSubmit={submit}>
        <Field label="Team name">{(id) => <input id={id} value={name} onChange={(e) => setName(e.target.value)} required maxLength={200} />}</Field>
        <Button type="submit" busy={busy}>Create team</Button>
      </form>
    </Card>
  );
}

function TeamCard({ team, keys, models, profiles, busy, canWrite, onSaveTeam, onCreateKey, onRevoke }: Omit<ViewProps, "keys"> & { team: Team; keys: ApiKey[] }) {
  const [all, setAll] = useState(team.approved_models === null);
  const [approved, setApproved] = useState<Set<string>>(new Set(team.approved_models ?? []));
  const [profile, setProfile] = useState(team.default_profile ?? "");
  const [label, setLabel] = useState("");
  const toggle = (m: string) => setApproved((s) => { const n = new Set(s); if (n.has(m)) n.delete(m); else n.add(m); return n; });
  return (
    <Card title={team.name}>
      <form
        className="stack"
        onSubmit={(e) => {
          e.preventDefault();
          void onSaveTeam(team, all ? null : [...approved].sort(), profile || null);
        }}
      >
        <fieldset disabled={!canWrite}>
          <legend>Approved models</legend>
          <label className="check">
            <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} /> Every catalog model
          </label>
          {!all && (
            <div className="checks">
              {models.map((m) => (
                <label key={m} className="check">
                  <input type="checkbox" checked={approved.has(m)} onChange={() => toggle(m)} /> {m}
                </label>
              ))}
            </div>
          )}
        </fieldset>
        <Field label="Default route profile" hint="Used when a tool asks for a model the gateway does not know">
          {(id) => (
            <select id={id} value={profile} onChange={(e) => setProfile(e.target.value)} disabled={!canWrite}>
              <option value="">Organisation default</option>
              {profiles.map((pr) => (
                <option key={pr} value={pr}>{pr}</option>
              ))}
            </select>
          )}
        </Field>
        {canWrite && <div><Button type="submit" variant="ghost" busy={busy === `save:${team.id}`}>Save team</Button></div>}
      </form>
      <h3>API keys</h3>
      {keys.length ? (
        <DataTable
          caption={`API keys of ${team.name}`}
          rows={keys}
          rowKey={(k) => k.id}
          columns={[
            { key: "l", header: "Used by", render: (k) => k.label },
            { key: "p", header: "Key", render: (k) => <code>{k.prefix}…</code> },
            { key: "c", header: "Created", render: (k) => date(k.created_at) },
            { key: "s", header: "Status", render: (k) => (k.revoked_at ? <span className="tag tag-off">revoked</span> : <span className="tag tag-on">active</span>) },
            {
              key: "a", header: "", render: (k) =>
                k.revoked_at || !canWrite ? null : (
                  <Button type="button" variant="danger" busy={busy === `revoke:${k.id}`} onClick={() => void onRevoke(k)} aria-label={`Revoke ${k.label}`}>
                    Revoke
                  </Button>
                ),
            },
          ]}
        />
      ) : (
        <Empty>No keys yet.</Empty>
      )}
      {canWrite && <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          if (label.trim()) void onCreateKey(team.id, label.trim()).then(() => setLabel(""));
        }}
      >
        <Field label="New key for" hint="A developer, CI job or service">
          {(id) => <input id={id} value={label} onChange={(e) => setLabel(e.target.value)} maxLength={200} placeholder="e.g. alice-laptop" />}
        </Field>
        <Button type="submit" busy={busy === `key:${team.id}`}>Issue key</Button>
      </form>}
    </Card>
  );
}
