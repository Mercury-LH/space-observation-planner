import { useRef, useState } from "react";
import "./NetworkSignals.css";

// Coordinates use the same proportions as the dotted land artwork.
const stations = {
  northAmerica: [150, 194],
  southAmerica: [313, 396],
  northAtlantic: [403, 108],
  africa: [560, 291],
  northAsia: [844, 136],
  eastAsia: [874, 242],
};

const routes = [
  ["northAmerica", [245, 41], "northAtlantic"],
  ["northAmerica", [289, 172], "southAmerica"],
  ["southAmerica", [430, 241], "africa"],
  ["northAtlantic", [518, 95], "africa"],
  ["northAtlantic", [626, 2], "northAsia"],
  ["africa", [698, 98], "northAsia"],
  ["africa", [711, 176], "eastAsia"],
];

const paths = routes.map(([from, control, to]) =>
  `M ${stations[from]} Q ${control} ${stations[to]}`
);
const connections = paths.join(" ");

function nextSignal(excluded, delay = 140 + Math.random() * 460) {
  const available = paths.filter((path) => !excluded.includes(path));
  return {
    path: available[Math.floor(Math.random() * available.length)],
    duration: 3300 + Math.random() * 2000,
    delay,
    direction: Math.random() < 0.5 ? "normal" : "reverse",
  };
}

function NetworkSignals({ onStationArrival }) {
  const arrivalsRemaining = useRef(3);
  const lastPulse = useRef(-Infinity);
  const [signals, setSignals] = useState(() => {
    const initial = [];
    for (const delay of [-950, -2100, -1550, -2600]) {
      initial.push(nextSignal(initial.map((signal) => signal.path), delay));
    }
    return initial;
  });

  function continueFlight(event, index) {
    if (event.target !== event.currentTarget || event.animationName !== "network-signal-flight") return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const visible = !document.hidden && (!reduced || index < signals.length - 1);
    if (visible && onStationArrival) {
      const now = performance.now();
      arrivalsRemaining.current--;
      if (arrivalsRemaining.current <= 0 && now - lastPulse.current >= 4000) {
        lastPulse.current = now;
        arrivalsRemaining.current = 3 + Math.floor(Math.random() * 3);
        onStationArrival();
      }
    }
    setSignals((current) => {
      const occupied = current
        .filter((_, i) => !reduced || i < current.length - 1 || i === index)
        .map((signal) => signal.path);
      const next = nextSignal(occupied);
      return current.map((signal, i) => {
        if (i === index) return next;
        // A hidden signal can lend its route and take the one just vacated.
        return signal.path === next.path ? current[index] : signal;
      });
    });
  }

  return (
    <g className="network-signals" fill="none" strokeLinecap="round">
      {signals.map((signal, index) => (
        <g
          key={`${index}-${signal.path}`}
          className="network-signal"
          style={{
            "--flight-duration": `${signal.duration}ms`,
            "--flight-delay": `${signal.delay}ms`,
            animationDirection: signal.direction,
          }}
          onAnimationEnd={(event) => continueFlight(event, index)}
        >
          <path d={signal.path} pathLength="100" stroke="#94dcff" strokeOpacity=".32" strokeWidth="var(--signal-halo, 23)" />
          <path d={signal.path} pathLength="100" stroke="#f2fbff" strokeWidth="var(--signal-core, 10)" />
        </g>
      ))}
    </g>
  );
}

export function ObservatoryNetwork({ onStationArrival }) {
  return (
    <div className="observatory-network" aria-hidden="true">
      <svg viewBox="0 0 1080 515" preserveAspectRatio="xMaxYMax meet" focusable="false">
        <g fill="none" strokeLinecap="round">
          <path className="network-route-edge" d={connections} vectorEffect="non-scaling-stroke" />
          <path className="network-route" d={connections} vectorEffect="non-scaling-stroke" />
        </g>
        <NetworkSignals onStationArrival={onStationArrival} />
        {Object.entries(stations).map(([name, [x, y]]) => (
          <g key={name}>
            <circle cx={x} cy={y} r="6.5" fill="#203a50" stroke="#a2bed0" strokeWidth="1" vectorEffect="non-scaling-stroke" />
            <circle cx={x} cy={y} r="2.4" fill="#afccdc" />
          </g>
        ))}
      </svg>
    </div>
  );
}
