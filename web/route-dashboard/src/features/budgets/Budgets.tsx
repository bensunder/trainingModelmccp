import { useState, type FormEvent } from "react";
import { ApiError } from "../../api/client";
import type { ApiKey, Budget, Period, Scope, Team } from "../../api/types";
import { useCan, useClient } from "../../auth/AuthContext";
import { Button, Card, DataTable, Empty, ErrorNote, Field, Loading } from "../../components/ui";
import { useResource } from "../../hooks/useResource";
import { usd } from "../../utils/format";

export default function BudgetsContainer({ orgId }: { orgId: string }) {
  const client = useClient();
  const { write } = useCan();
  const budgets = useResource<Budget[]>(`budgets:${orgId}`, (s) => client.budgets(orgId, s));
  const teams = useResource<Team[]>(`teams:${orgId}`, (s) => client.teams(orgId, s));
  const keys = useResource<ApiKey[]>(`keys:${orgId}`, (s) => client.keys(orgId, s));
  const [err, setErr] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const names = new Map<string, string>([[orgId, "Whole organisation"]]);
  for (const t of teams.data ?? []) names.set(t.id, `Team ${t.name}`);
  for (const k of keys.data ?? []) names.set(k.id, `Key ${k.label} (${k.prefix}…)`);
  const targets: { scope: Scope; id: string; name: string }[] = [
    { scope: "org", id: orgId, name: "Whole organisation" },
    ...(teams.data ?? []).map((t) => ({ scope: "team" as const, id: t.id, name: `Team ${t.name}` })),
    ...(keys.data ?? []).filter((k) => !k.revoked_at).map((k) => ({ scope: "key" as const, id: k.id, name: `Key ${k.label}` })),
  ];
  async function run(fn: () => Promise<void>) {
    setBusy(true);
    setErr(null);
    try {
      await fn();
    } catch (e) {
      setErr(e instanceof ApiError ? e : new ApiError(0, "Error", String(e)));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <ErrorNote error={budgets.error ?? err} onRetry={budgets.refresh} />
      {budgets.loading && <Loading label="Loading budgets" />}
      {write && <BudgetForm
        targets={targets}
        busy={busy}
        onSave={(b) =>
          run(async () => {
            const saved = await client.putBudget(orgId, b);
            budgets.mutate((prev) => [...(prev ?? []).filter((x) => x.id !== saved.id), saved]);
          })
        }
      />}
      <Card title="Budgets">
        {budgets.data?.length ? (
          <DataTable
            caption="Budgets"
            rows={budgets.data}
            rowKey={(b) => b.id}
            columns={[
              { key: "s", header: "Applies to", render: (b) => names.get(b.scope_id) ?? `${b.scope} ${b.scope_id.slice(0, 8)}` },
              { key: "p", header: "Period", render: (b) => (b.period === "month" ? "Monthly" : "Daily") },
              { key: "l", header: "Limit", numeric: true, render: (b) => usd(b.limit_usd) },
              { key: "h", header: "When reached", render: (b) => (b.hard ? "Refuse requests" : "Warn only") },
              {
                key: "x", header: "", render: (b) => !write ? null : (
                  <Button type="button" variant="danger" aria-label="Delete budget" disabled={busy}
                    onClick={() => run(async () => { await client.deleteBudget(orgId, b.id); budgets.mutate((prev) => prev?.filter((x) => x.id !== b.id)); })}>
                    Delete
                  </Button>
                ),
              },
            ]}
          />
        ) : (
          !budgets.loading && <Empty>No budgets. Spend is unlimited until you set one.</Empty>
        )}
      </Card>
    </>
  );
}

export function BudgetForm({ targets, busy, onSave }: {
  targets: { scope: Scope; id: string; name: string }[];
  busy: boolean;
  onSave: (b: { scope: Scope; scope_id: string; period: Period; limit_usd: number; hard: boolean }) => void;
}) {
  const [target, setTarget] = useState(0);
  const [period, setPeriod] = useState<Period>("month");
  const [limit, setLimit] = useState("");
  const [hard, setHard] = useState(true);
  function submit(e: FormEvent) {
    e.preventDefault();
    const t = targets[target];
    const v = Number(limit);
    if (!t || !Number.isFinite(v) || v < 0) return;
    onSave({ scope: t.scope, scope_id: t.id, period, limit_usd: v, hard });
  }
  return (
    <Card title="Set a budget">
      <form className="row wrap" onSubmit={submit}>
        <Field label="Applies to">
          {(id) => (
            <select id={id} value={target} onChange={(e) => setTarget(Number(e.target.value))}>
              {targets.map((t, i) => <option key={`${t.scope}:${t.id}`} value={i}>{t.name}</option>)}
            </select>
          )}
        </Field>
        <Field label="Period">
          {(id) => (
            <select id={id} value={period} onChange={(e) => setPeriod(e.target.value as Period)}>
              <option value="month">Monthly</option>
              <option value="day">Daily</option>
            </select>
          )}
        </Field>
        <Field label="Limit (USD)">
          {(id) => <input id={id} type="number" min={0} step="0.01" inputMode="decimal" value={limit} onChange={(e) => setLimit(e.target.value)} required />}
        </Field>
        <label className="check"><input type="checkbox" checked={hard} onChange={(e) => setHard(e.target.checked)} /> Refuse requests when reached</label>
        <Button type="submit" busy={busy}>Save budget</Button>
      </form>
    </Card>
  );
}
