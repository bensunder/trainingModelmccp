import { useMemo, useState } from "react";
import { useClient } from "../../auth/AuthContext";
import type { Team, Usage } from "../../api/types";
import { Card, DataTable, ErrorNote, Kpi, Loading } from "../../components/ui";
import { StackedBars } from "../../components/StackedBars";
import { useResource } from "../../hooks/useResource";
import { compact, daysAgoIso, monthStartIso, pct, usd } from "../../utils/format";

const RANGES = { month: "This month", "7d": "Last 7 days", "30d": "Last 30 days" } as const;
type Range = keyof typeof RANGES;

function sinceFor(r: Range): string {
  return r === "month" ? monthStartIso() : daysAgoIso(r === "7d" ? 7 : 30);
}

export default function OverviewContainer({ orgId }: { orgId: string }) {
  const client = useClient();
  const [range, setRange] = useState<Range>("month");
  const since = useMemo(() => sinceFor(range), [range]);
  const usage = useResource<Usage>(`usage:${orgId}:${range}`, (s) => client.usage(orgId, since, s));
  const teams = useResource<Team[]>(`teams:${orgId}`, (s) => client.teams(orgId, s));
  const teamName = useMemo(() => new Map((teams.data ?? []).map((t) => [t.id, t.name])), [teams.data]);
  return (
    <OverviewView
      range={range}
      onRange={setRange}
      usage={usage.data}
      loading={usage.loading}
      refreshing={usage.refreshing}
      error={usage.error}
      onRetry={usage.refresh}
      teamName={(id) => teamName.get(id) ?? id.slice(0, 8)}
    />
  );
}

export function OverviewView(p: {
  range: Range;
  onRange: (r: Range) => void;
  usage: Usage | undefined;
  loading: boolean;
  refreshing: boolean;
  error: Parameters<typeof ErrorNote>[0]["error"];
  onRetry: () => void;
  teamName: (id: string) => string;
}) {
  const u = p.usage;
  return (
    <>
      <div className="toolbar">
        <div role="group" aria-label="Period" className="segmented">
          {(Object.keys(RANGES) as Range[]).map((r) => (
            <button key={r} type="button" aria-pressed={p.range === r} onClick={() => p.onRange(r)}>
              {RANGES[r]}
            </button>
          ))}
        </div>
        {p.refreshing && <span className="muted" role="status">Updating…</span>}
      </div>
      <ErrorNote error={p.error} onRetry={p.onRetry} />
      {p.loading && <Loading label="Loading usage" />}
      {u && (
        <>
          <dl className="kpis">
            <Kpi label="Spend" value={usd(u.spend_usd)} hint={`${compact(u.requests)} requests`} />
            <Kpi label="Saved vs baseline" value={usd(u.saved_usd)} hint={u.saved_pct === null ? "no baseline yet" : `${pct(u.saved_pct)} less`} tone="good" />
            <Kpi label="Tokens" value={compact(u.tokens.input + u.tokens.output + u.tokens.cache_read + u.tokens.cache_write)} hint={`${compact(u.tokens.output)} output`} />
            <Kpi label="Cache hit rate" value={pct(u.cache_hit_rate)} />
            <Kpi label="Errors" value={compact(u.errors)} tone={u.errors > 0 ? "warn" : undefined} hint={u.responses_without_usage ? `${u.responses_without_usage} unmetered` : undefined} />
          </dl>
          <Card title="Spend by model">
            <StackedBars data={u.by_day_model} title="Spend per day by model" />
          </Card>
          <div className="grid-2">
            <Card title="Models">
              <DataTable
                caption="Spend by model"
                rows={Object.entries(u.by_model).sort((a, b) => b[1].cost_usd - a[1].cost_usd)}
                rowKey={([m]) => m}
                columns={[
                  { key: "m", header: "Model", render: ([m]) => m },
                  { key: "r", header: "Requests", numeric: true, render: ([, v]) => compact(v.requests) },
                  { key: "t", header: "Tokens", numeric: true, render: ([, v]) => compact(v.tokens) },
                  { key: "c", header: "Spend", numeric: true, render: ([, v]) => usd(v.cost_usd) },
                ]}
              />
            </Card>
            <Card title="Teams and task types">
              <DataTable
                caption="Spend by team"
                rows={Object.entries(u.by_team).sort((a, b) => b[1] - a[1])}
                rowKey={([t]) => t}
                columns={[
                  { key: "t", header: "Team", render: ([t]) => p.teamName(t) },
                  { key: "c", header: "Spend", numeric: true, render: ([, v]) => usd(v) },
                ]}
              />
              <DataTable
                caption="Requests by task type"
                rows={Object.entries(u.by_task).sort((a, b) => b[1] - a[1])}
                rowKey={([t]) => t}
                columns={[
                  { key: "t", header: "Task type", render: ([t]) => t.replace("_", " ") },
                  { key: "n", header: "Requests", numeric: true, render: ([, n]) => compact(n) },
                ]}
              />
            </Card>
          </div>
        </>
      )}
    </>
  );
}
