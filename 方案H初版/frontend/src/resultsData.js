export const resultFiles = {
  plan: "/samples/observation-plan.csv",
  summary: "/samples/result-summary.json",
};

// This archive stores its outage only as prose in summary.notes.replan.event.
// Keep the clock annotation attached to that archive's ID.
const exampleOutages = {
  "20261113-083604-HR": [{
    stationId: "2",
    startUtc: "2026-08-01T12:00:00Z",
    endUtc: "2026-08-01T18:00:00Z",
  }],
};

function readTasks(csv) {
  const text = csv.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index++) {
    const char = text[index];
    if (char === '"') {
      if (quoted && text[index + 1] === '"') {
        field += '"';
        index++;
      } else {
        quoted = !quoted;
      }
    } else if (!quoted && char === ",") {
      row.push(field);
      field = "";
    } else if (!quoted && (char === "\r" || char === "\n")) {
      if (row.length || field) rows.push([...row, field]);
      row = [];
      field = "";
      if (char === "\r" && text[index + 1] === "\n") index++;
    } else {
      field += char;
    }
  }
  if (quoted) throw new Error("Incomplete quoted CSV field");
  if (row.length || field) rows.push([...row, field]);

  const [headers = [], ...records] = rows;
  const required = ["task_id", "station_id", "target_id", "observation_start_utc", "observation_end_utc", "duration_seconds"];
  if (!required.every((name) => headers.includes(name))) throw new Error("Invalid observation plan headers");
  return records.map((values) => {
    if (values.length !== headers.length) throw new Error("Incomplete observation task");
    return Object.fromEntries(headers.map((name, index) => [name, values[index]]));
  });
}

async function fetchResult(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
  return response;
}

export async function loadObservationResults(signal) {
  const [csv, summary] = await Promise.all([
    fetchResult(resultFiles.plan, signal).then((response) => response.text()),
    fetchResult(resultFiles.summary, signal).then((response) => response.json()),
  ]);
  const tasks = readTasks(csv);
  const covered = new Set(tasks.map((task) => task.target_id)).size;
  if (tasks.length !== summary.metrics.tasks || covered !== summary.metrics.covered) {
    throw new Error("The schedule and result summary do not match");
  }

  // The solver stores zero-based station indices; the CSV uses one-based station IDs.
  const stations = new Set((summary.subset?.stations ?? []).map((index) => String(index + 1)));
  for (const task of tasks) stations.add(task.station_id);
  const stationIds = Array.from(stations).sort((a, b) => Number(a) - Number(b));
  const start = summary.config.start_utc;
  const startUtc = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(start) ? start : `${start}Z`;
  return { tasks, summary, stationIds, outages: exampleOutages[summary.run_id] ?? [], startUtc };
}
