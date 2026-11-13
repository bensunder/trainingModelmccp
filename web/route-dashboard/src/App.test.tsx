import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { ApiError, type Client } from "./api/client";
import { SignIn } from "./App";
import { AuthProvider } from "./auth/AuthContext";

test("a rejected token shows a clear message and keeps the user on the sign-in form", async () => {
  const makeClient = () => ({ listOrgs: async () => { throw new ApiError(401, "Unauthorized", "admin token required"); } }) as unknown as Client;
  render(<AuthProvider makeClient={makeClient as never}><SignIn /></AuthProvider>);
  fireEvent.change(screen.getByLabelText("Admin token"), { target: { value: "wrong" } });
  fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
  await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("not valid"));
  expect(screen.getByLabelText("Admin token")).toBeTruthy();
});
