import { memo, useMemo } from "react";
import { date, usd } from "../utils/format";

const PALETTE = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)", "var(--c7)", "var(--c8)"];

export interface StackedBarsProps {
  data: Record<string, Record<string, number>>; // day -> series -> value
  title: string;
  height?: number;
}

export function seriesOf(data: StackedBarsProps["data"]): string[] {
  const totals = new Map<string, number>();
  for (const day of Object.values(data)) for (const [k, v] of Object.entries(day)) totals.set(k, (totals.get(k) ?? 0) + v);
  return [...totals.entries()].sort((a, b) => b[1] - a[1]).map(([k]) => k);
}

/** Spend per day, stacked by model. SVG for sighted users; the same numbers as a
 * table for screen readers (the SVG is aria-hidden). */
export const StackedBars = memo(function StackedBars({ data, title, height = 220 }: StackedBarsProps) {
  const days = useMemo(() => Object.keys(data).sort(), [data]);
  const series = useMemo(() => seriesOf(data), [data]);
  const max = useMemo(() => Math.max(0, ...days.map((d) => Object.values(data[d] ?? {}).reduce((a, b) => a + b, 0))), [days, data]);
  if (!days.length) return <p className="empty">No spend in this period yet.</p>;
  const w = 640;
  const pad = { l: 8, r: 8, t: 8, b: 22 };
  const bw = (w - pad.l - pad.r) / days.length;
  const scale = (v: number) => (max > 0 ? (v / max) * (height - pad.t - pad.b) : 0);
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${w} ${height}`} role="presentation" aria-hidden="true" preserveAspectRatio="none">
        {days.map((d, i) => {
          let y = height - pad.b;
          return (
            <g key={d} data-day={d}>
              {series.map((s, si) => {
                const v = data[d]?.[s] ?? 0;
                const h = scale(v);
                y -= h;
                return h > 0 ? <rect key={s} x={pad.l + i * bw + bw * 0.15} y={y} width={bw * 0.7} height={h} fill={PALETTE[si % PALETTE.length]} rx={2} /> : null;
              })}
              {(days.length <= 10 || i % Math.ceil(days.length / 8) === 0) && (
                <text x={pad.l + i * bw + bw / 2} y={height - 6} textAnchor="middle" className="axis">
                  {date(d)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <figcaption>
        <ul className="legend">
          {series.map((s, si) => (
            <li key={s}>
              <span className="swatch" style={{ background: PALETTE[si % PALETTE.length] }} aria-hidden="true" />
              {s}
            </li>
          ))}
        </ul>
      </figcaption>
      <table className="sr-only">
        <caption>{title}</caption>
        <thead>
          <tr>
            <th scope="col">Day</th>
            {series.map((s) => (
              <th key={s} scope="col">{s}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {days.map((d) => (
            <tr key={d}>
              <th scope="row">{d}</th>
              {series.map((s) => (
                <td key={s}>{usd(data[d]?.[s] ?? 0)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
});
