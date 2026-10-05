// The sign-in form, shown in place of the page while the monitor refuses its requests as not signed in: the monitor's
// token, given once, sets the cookie every request then carries. (`/login?token=TOKEN`, opened, does the same.)

import { useState } from "react";
import { signIn } from "../api/access";

const reload = () => window.location.reload();

export function SignIn({ onSignedIn = reload }: { onSignedIn?: () => void }) {
  const [token, setToken] = useState("");
  const [refused, setRefused] = useState(false);
  const [asking, setAsking] = useState(false);
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setAsking(true);
    const taken = await signIn(token.trim()).catch(() => false);
    setAsking(false);
    setRefused(!taken);
    if (taken) onSignedIn();
  };
  return (
    <main className="signin">
      <form className="card fields" onSubmit={event => void submit(event)}>
        <span className="wordmark" aria-label="Rollout"><span className="speed" aria-hidden="true"><i /><i /><i /></span>ROLLOUT</span>
        <label className="field">
          <span>Token</span>
          <input type="password" autoFocus autoComplete="current-password" value={token} onChange={event => setToken(event.target.value)} />
        </label>
        <button type="submit" className="action" disabled={asking || !token.trim()}>Sign in</button>
        {refused ? <span className="error-text" role="alert">That is not this monitor's token.</span> : null}
      </form>
    </main>
  );
}
