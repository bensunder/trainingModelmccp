import { useState, type FormEvent } from "react";
import { ApiError } from "../../api/client";
import type { AdminRole, CreatedAdmin, OrgAdmin } from "../../api/types";
import { useCan, useClient } from "../../auth/AuthContext";
import { Button, Card, DataTable, Dialog, Empty, ErrorNote, Field, Loading } from "../../components/ui";
import { useResource } from "../../hooks/useResource";
import { date } from "../../utils/format";

export default function AdminsContainer({ orgId }: { orgId: string }) {
  const client = useClient();
  const { write } = useCan();
  const admins = useResource<OrgAdmin[]>(`admins:${orgId}`, (s) => client.admins(orgId, s));
  const [err, setErr] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [created, setCreated] = useState<CreatedAdmin | null>(null);
  async function act<T>(label: string, fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(label);
    setErr(null);
    try {
      return await fn();
    } catch (e) {
      setErr(e instanceof ApiError ? e : new ApiError(0, "Error", String(e)));
      return undefined;
    } finally {
      setBusy(null);
    }
  }
  return (
    <AdminsView
      admins={admins.data}
      loading={admins.loading}
      error={admins.error ?? err}
      canWrite={write}
      busy={busy}
      created={created}
      onDismissCreated={() => setCreated(null)}
      onCreate={async (label, role) => {
        const a = await act("create", () => client.createAdmin(orgId, label, role));
        if (a) {
          setCreated(a);
          const { token: _secret, ...stored } = a;
          admins.mutate((prev) => [...(prev ?? []), stored]);
        }
        return !!a;
      }}
      onRevoke={async (a) => {
        if (!window.confirm(`Revoke ${a.label} (${a.prefix}…)? Their dashboard and API access stop immediately.`)) return;
        const r = await act(`revoke:${a.id}`, () => client.revokeAdmin(orgId, a.id));
        if (r) admins.mutate((prev) => prev?.map((x) => (x.id === r.id ? r : x)));
      }}
    />
  );
}

export interface AdminsViewProps {
  admins: OrgAdmin[] | undefined;
  loading: boolean;
  error: ApiError | null;
  canWrite: boolean;
  busy: string | null;
  created: CreatedAdmin | null;
  onDismissCreated: () => void;
  onCreate: (label: string, role: AdminRole) => Promise<boolean>;
  onRevoke: (a: OrgAdmin) => Promise<void>;
}

export function AdminsView(p: AdminsViewProps) {
  return (
    <>
      <ErrorNote error={p.error} />
      {p.loading && <Loading label="Loading administrators" />}
      {p.canWrite && <NewAdminForm busy={p.busy === "create"} onCreate={p.onCreate} />}
      <Card title="Administrators">
        <p className="muted">Owners manage teams, keys, budgets, guardrails and other administrators. Viewers can only read.</p>
        {p.admins?.length ? (
          <DataTable
            caption="Organisation administrators"
            rows={p.admins}
            rowKey={(a) => a.id}
            columns={[
              { key: "l", header: "Who", render: (a) => a.label },
              { key: "r", header: "Role", render: (a) => (a.role === "owner" ? "Owner" : "Viewer") },
              { key: "p", header: "Token", render: (a) => <code>{a.prefix}…</code> },
              { key: "c", header: "Created", render: (a) => date(a.created_at) },
              { key: "s", header: "Status", render: (a) => (a.revoked_at ? <span className="tag tag-off">revoked</span> : <span className="tag tag-on">active</span>) },
              {
                key: "x", header: "", render: (a) =>
                  p.canWrite && !a.revoked_at ? (
                    <Button type="button" variant="danger" busy={p.busy === `revoke:${a.id}`} onClick={() => void p.onRevoke(a)} aria-label={`Revoke ${a.label}`}>
                      Revoke
                    </Button>
                  ) : null,
              },
            ]}
          />
        ) : (
          !p.loading && <Empty>No organisation administrators yet. The platform operator can always manage this organisation.</Empty>
        )}
      </Card>
      <Dialog open={p.created !== null} title="New administrator token" onClose={p.onDismissCreated}>
        <p>Give this token to {p.created?.label} through a private channel. It is shown once and cannot be recovered.</p>
        <output className="secret" aria-label="Administrator token">{p.created?.token}</output>
        <div className="row">
          <Button type="button" variant="ghost" onClick={() => p.created && navigator.clipboard?.writeText(p.created.token)}>Copy</Button>
          <Button type="button" onClick={p.onDismissCreated}>I stored it</Button>
        </div>
      </Dialog>
    </>
  );
}

function NewAdminForm({ busy, onCreate }: { busy: boolean; onCreate: (label: string, role: AdminRole) => Promise<boolean> }) {
  const [label, setLabel] = useState("");
  const [role, setRole] = useState<AdminRole>("viewer");
  async function submit(e: FormEvent) {
    e.preventDefault();
    if (label.trim() && (await onCreate(label.trim(), role))) setLabel("");
  }
  return (
    <Card title="Add an administrator">
      <form className="row wrap" onSubmit={submit}>
        <Field label="Who" hint="Name or e-mail of the person holding the token">
          {(id) => <input id={id} value={label} onChange={(e) => setLabel(e.target.value)} required maxLength={200} />}
        </Field>
        <Field label="Role">
          {(id) => (
            <select id={id} value={role} onChange={(e) => setRole(e.target.value as AdminRole)}>
              <option value="viewer">Viewer (read only)</option>
              <option value="owner">Owner (can change everything)</option>
            </select>
          )}
        </Field>
        <Button type="submit" busy={busy}>Create token</Button>
      </form>
    </Card>
  );
}
