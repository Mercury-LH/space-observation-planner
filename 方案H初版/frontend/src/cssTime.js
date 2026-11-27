// CSS minifiers may rewrite 900ms as .9s; Web Animations always takes milliseconds.
export function cssTimeToMilliseconds(value, fallback) {
  const match = value.trim().match(/^(\d+(?:\.\d+)?|\.\d+)(ms|s)$/);
  if (!match) return fallback;
  const duration = Number(match[1]) * (match[2] === "s" ? 1000 : 1);
  return Number.isFinite(duration) ? duration : fallback;
}
