import { useEffect, useState } from "react";
import { ApiError } from "../../api/client";
import type { ApiKey, AuditEvent, AuditPage, OrgAdmin, Team } from "../../api/types";
import { useClient } from "../../auth/AuthContext";
import { Button, Card, DataTable, Empty, ErrorNote, Loading } from "../../components/ui";
import { useResource } from "../../hooks/useResource";
import { dateTime } from "../../utils/format";

const ACTIONS: Record<string, string> = {
  "org.create": "Organisation created",
  "org.update": "Organisation settings changed",
  "guardrails.update": "Guardrail policy changed",
  "team.create": "Team created",
  "team.update": "Team changed",
  "key.create": "API key issued",
  "key.revoke": "API key revoked",
  "budget.put": "Budget set",
  "budget.delete": "Budget removed",
  "admin.create": "Administrator added",
  "admin.revoke": "Administrator revoked",
};

export default function AuditContainer({ orgId }: { orgId: string }) {
  const client = useClient();
  const first = useResource<AuditPage>(`audit:${orgId}`, (s) => client.audit(orgId, null, s));
  const admins = useResource<OrgAdmin[]>(`admins:${orgId}`, (s) => client.admins(orgId, s));
  const teams = useResource<Team[]>(`teams:${orgId}`, (s) => client.teams(orgId, s));
  const keys = useResource<ApiKey[]>(`keys:${orgId}`, (s) => client.keys(orgId, s));
  const [older, setOlder] = useState<AuditEvent[]>([]);
  const [next, setNext] = useState<string | null>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setOlder([]);
    setNext(first.data?.next_before ?? null);
  }, [first.data]);
  async function loadOlder() {
    if (!next) return;
    setBusy(true);
    setErr(null);
    try {
      const page = await client.audit(orgId, next);
      setOlder((o) => [...o, ...page.events]);
      setNext(page.next_before);
    } catch (e) {
      setErr(e instanceof ApiError ? e : new ApiError(0, "Error", String(e)));
    } finally {
      setBusy(false);
    }
  }
  const names = new Map((admins.data ?? []).map((a) => [`admin:${a.id}`, a.label]));
  // ids in details are shown by name when we know them (a deleted object keeps its id)
  const objects = new Map<string, string>([[orgId, "whole organisation"]]);
  for (const t of teams.data ?? []) objects.set(t.id, `team ${t.name}`);
  for (const k of keys.data ?? []) objects.set(k.id, `key ${k.label}`);
  return (
    <AuditView
      events={first.data ? [...first.data.events, ...older] : undefined}
      actorName={(actor) => (actor === "operator" ? "Platform operator" : names.get(actor) ?? actor)}
      objectName={(id) => objects.get(id)}
      loading={first.loading}
      error={first.error ?? err}
      hasMore={!!next}
      busy={busy}
      onLoadOlder={loadOlder}
      onRetry={first.refresh}
    />
  );
}

export function describe(detail: Record<string, unknown>, objectName: (id: string) => string | undefined = () => undefined): string {
  return Object.entries(detail)
    .map(([k, v]) => {
      if (k.endsWith("_id") && typeof v === "string") {
        const name = objectName(v);
        return `${k.slice(0, -3).replace(/_/g, " ")}: ${name ?? v.slice(0, 8)}`;
      }
      return `${k.replace(/_/g, " ")}: ${Array.isArray(v) ? v.join(", ") : v === null ? "all" : String(v)}`;
    })
    .join(" · ");
}

export function AuditView(p: {
  events: AuditEvent[] | undefined;
  actorName: (actor: string) => string;
  objectName?: (id: string) => string | undefined;
  loading: boolean;
  error: ApiError | null;
  hasMore: boolean;
  busy: boolean;
  onLoadOlder: () => void;
  onRetry?: () => void;
}) {
  return (
    <Card title="Audit log">
      <ErrorNote error={p.error} onRetry={p.onRetry} />
      <p className="muted">Every change made by an administrator or the platform operator, newest first. Keys and tokens are recorded by prefix only.</p>
      {p.loading && <Loading label="Loading audit log" />}
      {p.events?.length ? (
        <DataTable
          caption="Audit log"
          stackOnNarrow
          rows={p.events}
          rowKey={(e) => e.id}
          columns={[
            { key: "t", header: "When", render: (e) => <time dateTime={e.created_at}>{dateTime(e.created_at)}</time> },
            { key: "a", header: "Who", render: (e) => p.actorName(e.actor) },
            { key: "w", header: "What", render: (e) => ACTIONS[e.action] ?? e.action },
            { key: "d", header: "Details", render: (e) => <span className="muted">{describe(e.detail, p.objectName)}</span> },
          ]}
        />
      ) : (
        !p.loading && <Empty>No changes recorded yet.</Empty>
      )}
      {p.hasMore && (
        <div className="row">
          <Button type="button" variant="ghost" busy={p.busy} onClick={p.onLoadOlder}>Load older</Button>
        </div>
      )}
    </Card>
  );
}
