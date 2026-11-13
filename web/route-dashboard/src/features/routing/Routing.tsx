import type { Catalog, Routing } from "../../api/types";
import { useClient } from "../../auth/AuthContext";
import { Card, DataTable, ErrorNote, Loading } from "../../components/ui";
import { useResource } from "../../hooks/useResource";

export default function RoutingContainer() {
  const client = useClient();
  const routing = useResource<Routing>("routing", (s) => client.routing(s));
  const catalog = useResource<Catalog>("catalog", (s) => client.catalog(s));
  return <RoutingView routing={routing.data} catalog={catalog.data} loading={routing.loading} error={routing.error ?? catalog.error} />;
}

export function RoutingView({ routing, catalog, loading, error }: { routing?: Routing; catalog?: Catalog; loading: boolean; error: Parameters<typeof ErrorNote>[0]["error"] }) {
  const noEvidence = routing?.evidence.suite === "none";
  return (
    <>
      <ErrorNote error={error} />
      {loading && <Loading label="Loading routing" />}
      {routing && (
        <p className={noEvidence ? "note note-warn" : "note"}>
          {noEvidence
            ? "No evaluation evidence loaded: every task type except background requests stays on each profile's strong model (no evidence, no downgrade)."
            : `Evidence: ${routing.evidence.suite}, generated ${routing.evidence.generated_at}. Catalog ${routing.catalog_version}.`}
        </p>
      )}
      {routing && Object.entries(routing.profiles).map(([pid, p]) => (
        <Card key={pid} title={`${pid} · ${p.strategy}`}>
          <DataTable
            caption={`Routing of ${pid}`}
            rows={Object.entries(p.routes)}
            rowKey={([t]) => t}
            columns={[
              { key: "t", header: "Task type", render: ([t]) => t.replace("_", " ") },
              { key: "m", header: "Model today", render: ([, r]) => <strong>{r.model ?? "—"}</strong> },
              { key: "r", header: "Why", render: ([, r]) => <span className="muted">{r.reason}</span> },
            ]}
          />
        </Card>
      ))}
      {catalog && (
        <Card title="Model catalog">
          <DataTable
            caption="Models and prices"
            rows={catalog.models}
            rowKey={(m) => m.id}
            columns={[
              { key: "i", header: "Model", render: (m) => m.id },
              { key: "p", header: "Provider", render: (m) => <>{m.provider} {m.provider_ready ? null : <span className="tag tag-off">no key</span>}</> },
              { key: "t", header: "Tier", render: (m) => m.tier },
              { key: "in", header: "$ / 1M in", numeric: true, render: (m) => m.price.input_per_m.toFixed(2) },
              { key: "out", header: "$ / 1M out", numeric: true, render: (m) => m.price.output_per_m.toFixed(2) },
              { key: "s", header: "Price source", render: (m) => (m.price_source.toUpperCase().startsWith("UNVERIFIED") ? <span className="tag tag-warn">unverified</span> : <span className="muted">{m.price_source}</span>) },
            ]}
          />
        </Card>
      )}
    </>
  );
}
