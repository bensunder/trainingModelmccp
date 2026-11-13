import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { createClient, type Client } from "../api/client";
import type { Me } from "../api/types";

interface Auth {
  client: Client | null;
  me: Me | null;
  signIn: (token: string) => Promise<void>;
  signOut: () => void;
}

const Ctx = createContext<Auth | null>(null);

/** The admin token lives in memory only: never in localStorage or cookies, so a
 * reload or a closed tab signs out and nothing persists on the machine. The token
 * is either the platform operator's or an org admin's (`ra_...`); /me says which. */
export function AuthProvider({ children, makeClient = createClient }: { children: ReactNode; makeClient?: typeof createClient }) {
  const [session, setSession] = useState<{ client: Client; me: Me } | null>(null);
  const signIn = useCallback(
    async (token: string) => {
      const c = makeClient(token.trim());
      const me = await c.me(); // validates the token; throws ApiError(401) when wrong
      setSession({ client: c, me });
    },
    [makeClient],
  );
  const signOut = useCallback(() => setSession(null), []);
  const value = useMemo(() => ({ client: session?.client ?? null, me: session?.me ?? null, signIn, signOut }), [session, signIn, signOut]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAuth(): Auth {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAuth outside AuthProvider");
  return v;
}

export function useClient(): Client {
  const { client } = useAuth();
  if (!client) throw new Error("not signed in");
  return client;
}

/** What the signed-in principal may do. The server enforces the same rules; this
 * only hides controls that would be refused. */
export function useCan(): { write: boolean; operator: boolean } {
  const { me } = useAuth();
  return { write: !!me && me.role !== "viewer", operator: me?.kind === "operator" };
}
