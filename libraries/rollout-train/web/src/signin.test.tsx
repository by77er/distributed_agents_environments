import { QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { setSignedOut } from "./api/access";
import { readJson } from "./api/client";
import { asked, newQueryClient } from "./api/queries";
import { App } from "./App";
import { SignIn } from "./components/signin";

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("signing in", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    act(() => setSignedOut(false));
  });

  it("shows the sign-in form in place of the page once the monitor refuses a request as not signed in", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ error: "sign in" }, 401)));
    vi.stubGlobal("EventSource", class { addEventListener() {} close() {} });
    HTMLElement.prototype.scrollTo = () => {};  // (jsdom draws no page to scroll)
    render(<QueryClientProvider client={newQueryClient()}><MemoryRouter><App /></MemoryRouter></QueryClientProvider>);
    expect(await screen.findByRole("button", { name: "Sign in" })).toBeTruthy();
    expect(screen.queryByRole("navigation", { name: "pages" })).toBeNull();
  });

  it("gives the monitor the token as JSON, and goes on once it takes it", async () => {
    const fetched = vi.fn(async (_path: string, init?: RequestInit) =>
      JSON.parse(String(init?.body)).token === "right" ? json({ signed_in: true }) : json({ error: "no" }, 401));
    vi.stubGlobal("fetch", fetched);
    const signedIn = vi.fn();
    render(<SignIn onSignedIn={signedIn} />);
    const input = screen.getByLabelText("Token");
    fireEvent.change(input, { target: { value: "wrong" } });
    act(() => { screen.getByRole("button", { name: "Sign in" }).click(); });
    expect((await screen.findByRole("alert")).textContent).toBe("That is not this monitor's token.");
    fireEvent.change(input, { target: { value: " right " } });
    act(() => { screen.getByRole("button", { name: "Sign in" }).click(); });
    await waitFor(() => expect(signedIn).toHaveBeenCalledOnce());
    const [path, init] = fetched.mock.calls.at(-1)!;
    expect([path, init?.method, new Headers(init?.headers).get("Content-Type"), init?.body]).toEqual(["login", "POST", "application/json", '{"token":"right"}']);
  });
});

describe("what the page asks of the monitor", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    setSignedOut(false);
  });

  it("says JSON for every change, with a body or without", async () => {
    const fetched = vi.fn<(path: string, init?: RequestInit) => Promise<Response>>(async () => json({ launch: {} }));
    vi.stubGlobal("fetch", fetched);
    await asked("api/launches/launch_1/stop", "POST");
    await asked("api/bookmarks/best", "DELETE");
    await asked("api/rename", "POST", { id: "r", name: "n" });
    expect(fetched.mock.calls.map(([, init]) => new Headers(init?.headers).get("Content-Type"))).toEqual(["application/json", "application/json", "application/json"]);
  });

  it("is told it is signed out by a read or a change the monitor refuses", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ error: "sign in" }, 401)));
    await expect(readJson("api/system")).rejects.toHaveProperty("name", "SignedOut");
    await expect(asked("api/rename", "POST", {})).rejects.toHaveProperty("name", "SignedOut");
  });
});
