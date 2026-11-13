import { render, screen, within } from "@testing-library/react";
import type { AuditEvent, OrgAdmin, Team } from "../api/types";
import { AdminsView } from "./admins/Admins";
import { AuditView, describe } from "./audit/Audit";
import { TeamsView } from "./teams/Teams";

const team: Team = { id: "t1", org_id: "o", name: "eng", created_at: "2026-11-01T00:00:00Z", approved_models: null, default_profile: null };
const key = { id: "k1", org_id: "o", team_id: "t1", label: "alice", prefix: "rk_abcdefgh", created_at: "2026-11-01T00:00:00Z", revoked_at: null, default_profile: null };
const noop = async () => {};

function teams(canWrite: boolean) {
  return render(
    <TeamsView teams={[team]} keys={[key]} models={["glm"]} profiles={["auto-cost"]} loading={false} canWrite={canWrite}
      error={null} busy={null} created={null} onDismissCreated={() => {}} onCreateTeam={noop} onSaveTeam={noop}
      onCreateKey={noop} onRevoke={noop} />,
  );
}

test("a viewer sees teams and keys but no control that would change them", () => {
  teams(false);
  expect(screen.getByText("alice")).toBeTruthy();
  for (const name of ["Create team", "Save team", "Issue key", "Revoke alice"]) expect(screen.queryByRole("button", { name })).toBeNull();
  expect(screen.getByLabelText(/Every catalog model/).matches(":disabled")).toBe(true); // via the disabled fieldset
});

test("an owner gets the same screen with its controls", () => {
  teams(true);
  for (const name of ["Create team", "Save team", "Issue key", "Revoke alice"]) expect(screen.getByRole("button", { name })).toBeTruthy();
});

test("administrators list: revoked tokens have no revoke button; viewers get no form", () => {
  const admins: OrgAdmin[] = [
    { id: "a1", org_id: "o", label: "alice", role: "owner", prefix: "ra_aaaaaaaa", created_at: "2026-11-01T00:00:00Z", revoked_at: null },
    { id: "a2", org_id: "o", label: "bob", role: "viewer", prefix: "ra_bbbbbbbb", created_at: "2026-11-02T00:00:00Z", revoked_at: "2026-11-03T00:00:00Z" },
  ];
  const props = { admins, loading: false, error: null, busy: null, created: null, onDismissCreated: () => {}, onCreate: async () => true, onRevoke: noop };
  const { unmount } = render(<AdminsView {...props} canWrite />);
  expect(screen.getByRole("button", { name: "Revoke alice" })).toBeTruthy();
  expect(screen.queryByRole("button", { name: "Revoke bob" })).toBeNull();
  expect(screen.getByRole("button", { name: "Create token" })).toBeTruthy();
  unmount();
  render(<AdminsView {...props} canWrite={false} />);
  expect(screen.queryByRole("button", { name: "Create token" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Revoke alice" })).toBeNull();
});

test("audit log names the actor, describes the change and offers older pages", () => {
  const events: AuditEvent[] = [
    { id: "e1", created_at: "2026-11-12T10:00:00Z", org_id: "o", actor: "admin:a1", action: "key.revoke", target: "k1", detail: { prefix: "rk_abcdefgh", label: "alice" } },
    { id: "e2", created_at: "2026-11-11T10:00:00Z", org_id: "o", actor: "operator", action: "org.create", target: "o", detail: { name: "Acme" } },
  ];
  render(<AuditView events={events} actorName={(a) => (a === "operator" ? "Platform operator" : "alice")} loading={false} error={null} hasMore busy={false} onLoadOlder={() => {}} />);
  const rows = within(screen.getByRole("table")).getAllByRole("row");
  expect(rows[1]!.textContent).toContain("API key revoked");
  expect(rows[1]!.textContent).toContain("alice");
  expect(rows[2]!.textContent).toContain("Platform operator");
  expect(screen.getByRole("button", { name: "Load older" })).toBeTruthy();
  expect(describe({ approved_models: null, team_id: "t1" }, (id) => (id === "t1" ? "team eng" : undefined))).toBe("approved models: all · team: team eng");
  expect(describe({ scope_id: "0123456789abcdef" })).toBe("scope: 01234567"); // unknown id: short form
});
