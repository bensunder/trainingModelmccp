import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";

interface Entry<T> {
  data: T;
  at: number;
}
const cache = new Map<string, Entry<unknown>>();
const FRESH_MS = 15_000;

export function clearResourceCache(): void {
  cache.clear();
}

export interface Resource<T> {
  data: T | undefined;
  error: ApiError | null;
  loading: boolean; // no data yet
  refreshing: boolean; // showing cached data while fetching
  refresh: () => void;
  mutate: (update: (prev: T | undefined) => T | undefined) => void; // optimistic local change
}

/** Fetch keyed data with stale-while-revalidate: cached data shows instantly, a
 * background request refreshes it; in-flight requests are aborted on unmount or
 * key change, so stale answers never overwrite newer ones. key=null: idle. */
export function useResource<T>(key: string | null, load: (signal: AbortSignal) => Promise<T>): Resource<T> {
  const hit = key ? (cache.get(key) as Entry<T> | undefined) : undefined;
  const [data, setData] = useState<T | undefined>(hit?.data);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const [tick, setTick] = useState(0);
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    if (!key) return;
    const cached = cache.get(key) as Entry<T> | undefined;
    setData(cached?.data);
    if (cached && Date.now() - cached.at < FRESH_MS && tick === 0) return;
    const ctrl = new AbortController();
    setBusy(true);
    setError(null);
    loadRef
      .current(ctrl.signal)
      .then((d) => {
        if (ctrl.signal.aborted) return;
        cache.set(key, { data: d, at: Date.now() });
        setData(d);
      })
      .catch((e: unknown) => {
        if (ctrl.signal.aborted) return;
        setError(e instanceof ApiError ? e : new ApiError(0, "Error", String(e)));
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setBusy(false);
      });
    return () => ctrl.abort();
  }, [key, tick]);

  const refresh = useCallback(() => setTick((t) => t + 1), []);
  const mutate = useCallback(
    (update: (prev: T | undefined) => T | undefined) => {
      setData((prev) => {
        const next = update(prev);
        if (key && next !== undefined) cache.set(key, { data: next, at: Date.now() });
        return next;
      });
    },
    [key],
  );
  return { data, error, loading: busy && data === undefined, refreshing: busy && data !== undefined, refresh, mutate };
}
