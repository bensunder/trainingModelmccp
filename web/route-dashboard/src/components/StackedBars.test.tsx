import { render, screen, within } from "@testing-library/react";
import { seriesOf, StackedBars } from "./StackedBars";

const data = { "2026-11-02": { glm: 1, "claude-sonnet": 4 }, "2026-11-01": { glm: 2 } };

test("series are ordered by total spend", () => {
  expect(seriesOf(data)).toEqual(["claude-sonnet", "glm"]);
});

test("the same numbers are available as a table for screen readers", () => {
  render(<StackedBars data={data} title="Spend per day" />);
  const table = screen.getByRole("table", { name: "Spend per day" });
  const rows = within(table).getAllByRole("row");
  expect(rows).toHaveLength(3); // header + 2 days, sorted
  expect(within(rows[1]!).getByRole("rowheader").textContent).toBe("2026-11-01");
  expect(screen.getByText("claude-sonnet", { selector: "li" })).toBeTruthy();
});

test("empty data shows a message instead of an empty chart", () => {
  render(<StackedBars data={{}} title="x" />);
  expect(screen.getByText(/No spend/)).toBeTruthy();
});
