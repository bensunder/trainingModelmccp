import { monthStartIso, pct, usd } from "./format";

test("formats money, percentages and month start", () => {
  expect(usd(1234.5)).toBe("$1,234.50");
  expect(usd(0.00123)).toBe("$0.00123");
  expect(pct(0.4216)).toBe("42.2%");
  expect(pct(null)).toBe("—");
  expect(monthStartIso(new Date("2026-11-12T20:00:00Z"))).toBe("2026-11-01T00:00:00.000Z");
});
