import { useCallback, useEffect, useRef, useState } from "react";

export async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    credentials: "same-origin", ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(typeof detail === "string" ? detail : Array.isArray(detail)
      ? detail.map(item => item.msg).join("；") : `请求失败（${response.status}），请重试。`);
  }
  return payload;
}

function stored(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function remember(key, value) {
  try { if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value); } catch { /* session still works */ }
}

export function usePlanner() {
  const [run, setRun] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const busy = useRef(false);
  const initialization = useRef(null);
  const refreshHistory = useCallback(async () => {
    const { runs } = await api("/runs");
    setHistory(runs);
    return runs;
  }, []);
  const select = useCallback(value => {
    setRun(value);
    remember("obsplan-run", value.id);
    setError("");
  }, []);

  useEffect(() => {
    let active = true;
    initialization.current ??= api("/runs");
    initialization.current.then(({ runs }) => {
      const last = runs.find(item => item.id === stored("obsplan-run")) ?? runs[0];
      if (active) { setHistory(runs); if (last) select(last); }
    }).catch(error => { if (active) setError(error.message); });
    return () => { active = false; };
  }, [refreshHistory, select]);

  const running = run && (["queued", "running"].includes(run.state)
    || ["queued", "running"].includes(run.validation_state));
  useEffect(() => {
    if (!running) return;
    const controller = new AbortController();
    let timer;
    const poll = async () => {
      try {
        const value = await api(`/runs/${run.id}`, { signal: controller.signal });
        if (!controller.signal.aborted) {
          setRun(value);
          setError("");
          if (value.state !== "running" && value.state !== "queued"
            && !["queued", "running"].includes(value.validation_state)) await refreshHistory();
        }
      } catch (error) { if (!controller.signal.aborted) setError(`状态读取失败，正在重试。${error.message}`); }
      if (!controller.signal.aborted) timer = setTimeout(poll, 1500);
    };
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [run?.id, running, refreshHistory]);

  async function submit(configuration) {
    if (busy.current || running) return;
    busy.current = true;
    setSubmitting(true);
    setError("");
    const request = { date: configuration.date, days: Number(configuration.days),
      stations: configuration.stations.map(Number).sort((a, b) => a - b), algo: configuration.algo };
    const encoded = JSON.stringify(request);
    let pending;
    try { pending = JSON.parse(stored("obsplan-pending")); } catch { /* ignore corrupt local state */ }
    const key = pending?.request === encoded ? pending.key
      : (globalThis.crypto?.randomUUID?.() ?? `request-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    remember("obsplan-pending", JSON.stringify({ request: encoded, key }));
    try {
      await initialization.current?.catch(() => {});
      // Establish the session before POST even if the initial history request failed.
      await api("/config");
      const value = await api("/runs", { method: "POST", body: encoded, headers: { "Idempotency-Key": key } });
      select(value);
      remember("obsplan-pending", null);
      await refreshHistory();
    } catch (error) { setError(error.message); }
    finally { busy.current = false; setSubmitting(false); }
  }

  async function validate() {
    if (!run || busy.current) return;
    busy.current = true;
    setError("");
    try { setRun(await api(`/runs/${run.id}/validate`, { method: "POST" })); }
    catch (error) { setError(error.message); }
    finally { busy.current = false; }
  }
  return { run, history, error, submitting, running, submit, validate, select, refreshHistory };
}
