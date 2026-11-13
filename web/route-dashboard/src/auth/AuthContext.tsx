import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { createClient, type Client } from "../api/client";

interface Auth {
  client: Client | null;
  signIn: (token: string) => Promise<void>;
  signOut: () => void;
}

const Ctx = createContext<Auth | null>(null);

/** The admin token lives in memory only: never in localStorage or cookies, so a
 * reload or a closed tab signs out and nothing persists on the machine. */
export function AuthProvider({ children, makeClient = createClient }: { children: ReactNode; makeClient?: typeof createClient }) {
  const [client, setClient] = useState<Client | null>(null);
  const signIn = useCallback(
    async (token: string) => {
      const c = makeClient(token.trim());
      await c.listOrgs(); // validates the token; throws ApiError(401) when wrong
      setClient(c);
    },
    [makeClient],
  );
  const signOut = useCallback(() => setClient(null), []);
  const value = useMemo(() => ({ client, signIn, signOut }), [client, signIn, signOut]);
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
