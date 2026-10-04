// A run's name, and a control to call it by another: its id stays as it is, and so does everything kept under it.

import { useState } from "react";
import { useRename } from "../api/queries";

export function Rename({ id, name }: { id: string; name: string }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(name);
  const rename = useRename();
  if (!editing) {
    return (
      <span className="named">
        <span title={`id: ${id}`}>{name}</span>
        <button type="button" className="linkish rename-open" onClick={() => { setDraft(name); rename.reset(); setEditing(true); }} aria-label="rename this run">rename</button>
      </span>
    );
  }
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (draft.trim() === name) { setEditing(false); return; }
    rename.mutate({ id, name: draft }, { onSuccess: () => setEditing(false) });
  };
  return (
    <form className="rename" onSubmit={submit}>
      <input
        autoFocus
        value={draft}
        onChange={event => setDraft(event.target.value)}
        onKeyDown={event => { if (event.key === "Escape") setEditing(false); }}
        aria-label="the run's new name"
        spellCheck={false}
      />
      <button type="submit" disabled={rename.isPending || !draft.trim()}>{rename.isPending ? "saving…" : "save"}</button>
      <button type="button" onClick={() => setEditing(false)}>cancel</button>
      <span className="faint small">id {id}</span>
      {rename.isError ? <span className="error-text small">{rename.error.message}</span> : null}
    </form>
  );
}
