// Signing in to the monitor. Every request the page makes is to the monitor that served it, so the browser sends the
// cookie signing in set (HttpOnly: the page never reads it). A request the monitor refuses as not signed in (401) says
// so here, and the page shows its sign-in form until a token is given.

import { useSyncExternalStore } from "react";

/** The monitor refused a request: the page is not signed in. */
export class SignedOut extends Error {
  constructor(path: string) {
    super(`${path}: not signed in`);
    this.name = "SignedOut";
  }
}

let signedOut = false;
const listeners = new Set<() => void>();

/** Say that the monitor refused a request as not signed in (or that it now takes them). */
export function setSignedOut(next: boolean): void {
  if (next === signedOut) return;
  signedOut = next;
  for (const listener of listeners) listener();
}

/** Whether the monitor refuses the page's requests as not signed in. */
export const useSignedOut = (): boolean =>
  useSyncExternalStore(
    listener => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => signedOut,
  );

/** Sign in with the monitor's token: it sets the cookie every request then carries. Whether it took the token. */
export async function signIn(token: string): Promise<boolean> {
  const answer = await fetch("login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token }) });
  if (answer.ok) setSignedOut(false);
  return answer.ok;
}
