// Words and numbers, as the page writes them.

export const span = (seconds: number | null | undefined): string =>
  seconds == null ? "" : seconds < 90 ? `${Math.round(seconds)} s` : seconds < 5400 ? `${Math.round(seconds / 60)} min` : `${(seconds / 3600).toFixed(1)} h`;

export const clock = (at: number | null | undefined): string =>
  at ? new Date(at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";

export const day = (at: number): string => new Date(at * 1000).toLocaleDateString([], { month: "short", day: "numeric" });

export const bytes = (count: number | null | undefined): string =>
  (count ?? 0) >= 2 ** 30 ? `${((count ?? 0) / 2 ** 30).toFixed(1)} GiB` : (count ?? 0) >= 2 ** 20 ? `${Math.round((count ?? 0) / 2 ** 20)} MiB`
    : `${Math.round((count ?? 0) / 1024)} KiB`;

export const figure = (value: unknown): string =>
  value == null ? "–" : typeof value === "boolean" ? (value ? "yes" : "no")
    : typeof value === "number" ? (Number.isInteger(value) ? value.toLocaleString() : value.toFixed(Math.abs(value) < 1 ? 4 : 1))
      : String(value);

export const tokens = (count: number | null | undefined): string =>
  (count ?? 0) >= 1e6 ? `${((count ?? 0) / 1e6).toFixed(1)} M` : (count ?? 0) >= 1e3 ? `${Math.round((count ?? 0) / 1e3)} k` : String(count ?? 0);


export const mean = (values: number[]): number | null => (values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null);

export const percent = (value: number): string => `${Math.round(100 * value)}%`;

export const tick = (value: number): string =>
  Math.abs(value) >= 1e6 ? `${+(value / 1e6).toPrecision(3)}M` : Math.abs(value) >= 1e4 ? `${+(value / 1e3).toPrecision(3)}k` : String(+value.toPrecision(3));

export const tickSpan = (seconds: number): string =>
  seconds < 120 ? `${Math.round(seconds)} s` : seconds < 7200 ? `${+(seconds / 60).toPrecision(2)} min` : `${+(seconds / 3600).toPrecision(2)} h`;

/** A slot's hue, from its whole name: names that differ only in their number (`agent-1`, `agent-2`) are a wide step
 * apart, and the rest of the name moves them all (`red1` and `blue1` differ). */
export function slotHue(name: string): number {
  const [, stem, number] = name.match(/^(.*?)(\d*)$/)!;
  let hash = 2166136261;
  for (const letter of stem) hash = Math.imul(hash ^ letter.charCodeAt(0), 16777619);
  hash = Math.imul(hash ^ (hash >>> 16), 0x85ebca6b);
  hash = Math.imul(hash ^ (hash >>> 13), 0xc2b2ae35);
  return (((hash ^ (hash >>> 16)) >>> 0) + Number(number || 0) * 97) % 360;
}

/** A share of booleans as a percentage, or a dash for none. */
export const shareOf = (outcomes: boolean[]): string =>
  outcomes.length ? `${Math.round((100 * outcomes.filter(Boolean).length) / outcomes.length)}%` : "–";

export const plural = (count: number, word: string): string => `${count} ${word}${count === 1 ? "" : "s"}`;

export const byNumber = (a: string, b: string): number => a.localeCompare(b, undefined, { numeric: true });
