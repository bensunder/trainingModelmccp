import { useEffect, useState } from "react";
import { ApiError } from "../../api/client";
import type { GuardAction, GuardrailPolicy, Org } from "../../api/types";
import { useCan, useClient } from "../../auth/AuthContext";
import { Button, Card, ErrorNote, Field } from "../../components/ui";

const DEFAULTS: GuardrailPolicy = { secrets: "redact", pii: "flag", injection: "flag" };
const ACTIONS: { v: GuardAction; label: string }[] = [
  { v: "off", label: "Off" },
  { v: "flag", label: "Flag (record, allow)" },
  { v: "redact", label: "Redact before sending" },
  { v: "block", label: "Block the request" },
];
const ROWS: { key: keyof GuardrailPolicy; title: string; help: string; actions: GuardAction[] }[] = [
  { key: "secrets", title: "Secrets", help: "Cloud keys, tokens, private keys, passwords in code and in tool arguments.", actions: ["off", "flag", "redact", "block"] },
  { key: "pii", title: "Personal data", help: "E-mail addresses, phone numbers, card and national-ID numbers.", actions: ["off", "flag", "redact", "block"] },
  { key: "injection", title: "Prompt injection", help: "Text that tries to override instructions (from pasted pages or tool output).", actions: ["off", "flag", "block"] },
];

export default function GuardrailsContainer({ org, onSaved }: { org: Org; onSaved: (o: Org) => void }) {
  const client = useClient();
  const { write } = useCan();
  const [policy, setPolicy] = useState<GuardrailPolicy>({ ...DEFAULTS, ...org.guardrails });
  const [err, setErr] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  useEffect(() => setPolicy({ ...DEFAULTS, ...org.guardrails }), [org]);
  async function save() {
    setBusy(true);
    setErr(null);
    setSaved(false);
    try {
      onSaved(await client.putGuardrails(org.id, policy));
      setSaved(true);
    } catch (e) {
      setErr(e instanceof ApiError ? e : new ApiError(0, "Error", String(e)));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Card title="Guardrails" actions={write ? <Button type="button" onClick={save} busy={busy}>Save policy</Button> : undefined}>
      <ErrorNote error={err} />
      {saved && <p className="note note-ok" role="status">Saved. New requests use this policy.</p>}
      <p className="muted">Findings are recorded as the rule and a count; the matched text itself is never stored.</p>
      <div className="stack">
        {ROWS.map((r) => (
          <Field key={r.key} label={r.title} hint={r.help}>
            {(id) => (
              <select id={id} value={policy[r.key]} disabled={!write} onChange={(e) => setPolicy((p) => ({ ...p, [r.key]: e.target.value as GuardAction }))}>
                {ACTIONS.filter((a) => r.actions.includes(a.v)).map((a) => <option key={a.v} value={a.v}>{a.label}</option>)}
              </select>
            )}
          </Field>
        ))}
      </div>
    </Card>
  );
}
