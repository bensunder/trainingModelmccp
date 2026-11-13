import { ApiError, createClient } from "./client";

function fakeFetch(status: number, body: unknown, ct = "application/problem+json") {
  return vi.fn(async (_url: RequestInfo | URL, _init?: RequestInit) =>
    new Response(body === undefined ? null : JSON.stringify(body), { status, headers: { "content-type": ct } }),
  );
}

test("sends the bearer token and parses JSON", async () => {
  const f = fakeFetch(200, [{ id: "o1", name: "Acme" }], "application/json");
  const orgs = await createClient("tok", f as unknown as typeof fetch).listOrgs();
  expect(orgs[0]?.name).toBe("Acme");
  const init = f.mock.calls[0]?.[1];
  expect((init?.headers as Record<string, string>).authorization).toBe("Bearer tok");
  expect(f.mock.calls[0]?.[0]).toBe("/admin/v1/orgs");
});

test("turns problem+json into ApiError with the server's title and detail", async () => {
  const f = fakeFetch(429, { type: "x/budget-exceeded", title: "Budget exceeded", status: 429, detail: "team month budget $5.00 reached" });
  const err = await createClient("t", f as unknown as typeof fetch).usage("o1").catch((e: unknown) => e);
  expect(err).toBeInstanceOf(ApiError);
  expect((err as ApiError).status).toBe(429);
  expect((err as ApiError).detail).toContain("$5.00");
});

test("network failure is a readable ApiError; caller cancellation is not", async () => {
  const down = vi.fn(async () => { throw new TypeError("fetch failed"); });
  const err = await createClient("t", down as unknown as typeof fetch).catalog().catch((e: unknown) => e);
  expect((err as ApiError).title).toBe("Network error");
  const ctrl = new AbortController();
  const slow = vi.fn((_u: RequestInfo | URL, init?: RequestInit) => new Promise<Response>((_, rej) => init?.signal?.addEventListener("abort", () => rej(new DOMException("aborted", "AbortError")))));
  const p = createClient("t", slow as unknown as typeof fetch).catalog(ctrl.signal);
  ctrl.abort();
  await expect(p).rejects.not.toBeInstanceOf(ApiError);
});

test("encodes path parameters and sends JSON bodies", async () => {
  const f = fakeFetch(200, { id: "b" }, "application/json");
  await createClient("t", f as unknown as typeof fetch).putBudget("o/1", { scope: "team", scope_id: "t1", period: "month", limit_usd: 5, hard: true });
  expect(f.mock.calls[0]?.[0]).toBe("/admin/v1/orgs/o%2F1/budgets");
  expect(JSON.parse(String(f.mock.calls[0]?.[1]?.body))).toMatchObject({ limit_usd: 5, hard: true });
});
