import { useEffect, useState } from "react";

export const ROUTES = ["overview", "teams", "budgets", "guardrails", "routing", "admins", "audit"] as const;
export type Route = (typeof ROUTES)[number];

function read(): Route {
  const h = window.location.hash.replace(/^#\/?/, "");
  return (ROUTES as readonly string[]).includes(h) ? (h as Route) : "overview";
}

/** Tiny hash router: deep links (#/budgets) and back/forward work, no dependency. */
export function useHashRoute(): [Route, (r: Route) => void] {
  const [route, setRoute] = useState<Route>(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return [route, (r) => { window.location.hash = `/${r}`; }];
}
