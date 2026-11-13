const usdFmt = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
const smallUsd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumSignificantDigits: 3 });
const compactFmt = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

export const usd = (v: number) => (Math.abs(v) > 0 && Math.abs(v) < 1 ? smallUsd.format(v) : usdFmt.format(v));
export const compact = (v: number) => compactFmt.format(v);
export const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);
export const date = (iso: string) => new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric" });
export const dateTime = (iso: string) =>
  new Date(iso).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

/** First day of the current UTC month, as the API expects. */
export function monthStartIso(now = new Date()): string {
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1)).toISOString();
}

export function daysAgoIso(days: number, now = new Date()): string {
  return new Date(now.getTime() - days * 86_400_000).toISOString();
}
