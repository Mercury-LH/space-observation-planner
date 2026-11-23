import { useId } from "react";
import "./ObservationClock.css";

const DAY_MS = 86_400_000;
const CLOCK_EDGE = 163;
const BEZEL_RADIUS = 175;
const TRACK_WIDTH = 10;
const TICKS = Array.from({ length: 96 }, (_, index) => index);

function utcTime(value) {
  if (typeof value !== "string") return NaN;
  const localTimestamp = value.includes("T") && !/(?:Z|[+-]\d{2}:?\d{2})$/i.test(value);
  return Date.parse(localTimestamp ? value + "Z" : value);
}

function stationKey(value) {
  return String(value).replace(/^0+(?=\d)/, "");
}

function point(radius, turn) {
  const angle = 2 * Math.PI * turn;
  return [radius * Math.sin(angle), -radius * Math.cos(angle)];
}

function arc(radius, start, end) {
  const from = point(radius, start);
  const to = point(radius, end);
  // A complete circle needs two SVG arc commands.
  if (end - start >= 1) {
    const opposite = point(radius, start + .5);
    return `M${from} A${radius} ${radius} 0 1 1 ${opposite} A${radius} ${radius} 0 1 1 ${from}`;
  }
  return `M${from} A${radius} ${radius} 0 ${end - start > .5 ? 1 : 0} 1 ${to}`;
}

function interval(radius, startUtc, endUtc, dayStart) {
  const start = Math.max(0, (utcTime(startUtc) - dayStart) / DAY_MS);
  const end = Math.min(1, (utcTime(endUtc) - dayStart) / DAY_MS);
  return end > start ? arc(radius, start, end) : "";
}

export default function ObservationClock({ tasks, startUtc, stationIds, outages = [] }) {
  const id = useId();
  const dayStart = Math.floor(utcTime(startUtc) / DAY_MS) * DAY_MS;
  const tracks = stationIds.map((station, index) => {
    const key = stationKey(station);
    const radius = 149 - 18 * index;
    const segments = tasks
      .filter(row => stationKey(row.station_id) === key)
      .map(row => interval(radius, row.observation_start_utc, row.observation_end_utc, dayStart))
      .filter(Boolean);
    const offline = outages
      .filter(outage => stationKey(outage.stationId) === key)
      .map(outage => interval(radius, outage.startUtc, outage.endUtc, dayStart))
      .join(" ");
    return { key, radius, segments, offline };
  });
  const count = tracks.reduce((sum, track) => sum + track.segments.length, 0);
  const trackDescription = tracks.map(({ key, segments }, index) =>
    `${index === 0 ? "外" : "内"}圈站点 ${key.padStart(2, "0")} 有 ${segments.length} 项任务`
  ).join("，");

  return (
    <svg className="observation-clock" viewBox="-211 -208 422 416"
      role="img" aria-labelledby={id + "-title"} aria-describedby={id + "-desc"} focusable="false">
      <title id={id + "-title"}>{`24 小时观测时序，共 ${count} 项任务`}</title>
      <desc id={id + "-desc"}>上方为 UTC 零时，时间顺时针递增。{trackDescription}。色段位置与长度对应已安排任务的起止时间，扫光沿时间轴循环回放。</desc>
      <defs>
        <linearGradient id={id + "-steel"} gradientUnits="userSpaceOnUse"
          x1="-136" y1="-122" x2="143" y2="111">
          <stop stopColor="#e1edf4" />
          <stop offset=".34" stopColor="#9eafbc" />
          <stop offset=".7" stopColor="#596e7d" />
          <stop offset="1" stopColor="#9aafbc" />
        </linearGradient>
        <pattern id={id + "-hatch"} width="5" height="5" patternUnits="userSpaceOnUse">
          <path d="M-1 1l2 -2 M0 5l5 -5 M4 6l2 -2" fill="none" stroke="currentColor" strokeWidth=".85" />
        </pattern>
        <clipPath id={id + "-sweep"} clipPathUnits="userSpaceOnUse">
          <path className="observation-clock__motion" d="M0 0 L-79 -173 L0 -190 Z" />
        </clipPath>
        {tracks.map(({ key, segments }) =>
          <path key={key} id={id + "-station-" + key} d={segments.join(" ")} />)}
      </defs>
      <g fill="none" aria-hidden="true">
        <g className="observation-clock__bezel" strokeLinecap="round">
          <circle r={BEZEL_RADIUS} stroke="#263946" strokeWidth="1" />
          <path d={[arc(BEZEL_RADIUS, -.29, -.055), arc(BEZEL_RADIUS, .02, .30),
            arc(BEZEL_RADIUS, .36, .62), arc(BEZEL_RADIUS, .645, .673)].join(" ")}
            stroke={`url(#${id}-steel)`} strokeWidth="3" />
          <path d={arc(BEZEL_RADIUS, -.055, -.025)} stroke="#d4e3ed" strokeWidth="4.5" />
          <path d={arc(BEZEL_RADIUS + 3, -.16, -.025)} stroke="#a2b6c4" strokeWidth="1" />
        </g>
        <circle className="observation-clock__rim" r={CLOCK_EDGE} />
        {TICKS.map(tick => {
          const [x1, y1] = point(CLOCK_EDGE, tick / TICKS.length);
          const [x2, y2] = point(CLOCK_EDGE + (tick % 4 === 0 ? 6 : 2.5), tick / TICKS.length);
          return <line className="observation-clock__tick" key={tick} x1={x1} y1={y1} x2={x2} y2={y2} />;
        })}
        {tracks.map(({ key, radius, offline }, index) => (
          <g key={key} strokeWidth={TRACK_WIDTH}>
            <circle className="observation-clock__bed" r={radius} />
            {offline && <path className="observation-clock__outage" d={offline} stroke={`url(#${id}-hatch)`} />}
            <use className="observation-clock__tasks" data-track={index} href={"#" + id + "-station-" + key} />
          </g>
        ))}
        <g className="observation-clock__highlight" clipPath={`url(#${id}-sweep)`} strokeWidth={TRACK_WIDTH}>
          {tracks.map(({ key }, index) =>
            <use key={key} data-track={index} href={"#" + id + "-station-" + key} />)}
        </g>
        <g className="observation-clock__motion observation-clock__tracer">
          <path d={arc(158, -.038, 0)} stroke="#b2d1e5" strokeWidth="1.5" opacity=".38" />
          <path d={arc(158, -.014, 0)} stroke="#dcecf4" strokeWidth="1.8" opacity=".86" />
          <path d="M0 -124 V-158" stroke="#eef6fa" strokeWidth=".9" opacity=".64" />
          <circle cy="-158" r="4.3" fill="#a5d6f3" opacity=".17" />
          <circle cy="-158" r="2.1" fill="#f1f6f7" />
        </g>
      </g>
      <g className="observation-clock__hours" aria-hidden="true">
        <text y="-192">00</text><text x="197" y="1">06</text>
        <text y="193">12</text><text x="-197" y="1">18</text>
      </g>
      <g className="observation-clock__total" aria-hidden="true">
        <text className="observation-clock__count" y="9">{count}</text>
        <text className="observation-clock__caption" y="39">已安排任务</text>
      </g>
    </svg>
  );
}
