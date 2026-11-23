import { useLayoutEffect, useRef } from "react";
import "./RunOrbitMark.css";

const orbit = "M100.787 34.180 A48.5 36.4 -26 1 0 19.213 85.820 A48.5 36.4 -26 1 0 100.787 34.180 Z";
const triangle = "M47.5 43 Q47.5 39.2 50.75 41.25 L77.65 57.9 Q81 60 77.65 62.1 L50.75 78.75 Q47.5 80.8 47.5 77 Z";
const fileArrowDown = "M46 40h17l11 11v27a2 2 0 0 1-2 2H46a2 2 0 0 1-2-2V42a2 2 0 0 1 2-2Z M63 40v11h11 M59 55v15m-6-6 6 6 6-6";
const tail = [
  { length: 10, width: 2, opacity: .20 },
  { length: 7, width: 2.2, opacity: .36 },
  { length: 4, width: 2.4, opacity: .62 },
  { length: 1.9, width: 2.6, opacity: .87 },
];

export function RunOrbitMark({ page = "run" }) {
  const tracer = useRef(null);

  useLayoutEffect(() => {
    const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
    const duration = preference.matches ? 6000 : 4200;
    const loop = tracer.current.animate(
      [{ strokeDashoffset: "0px" }, { strokeDashoffset: "100px" }],
      { duration, iterations: Infinity, easing: "linear" },
    );
    // The flying and docked tracers use the same document clock, even while hidden.
    loop.startTime = 0;
    function changeSpeed({ matches }) {
      loop.updatePlaybackRate(duration / (matches ? 6000 : 4200));
    }
    preference.addEventListener("change", changeSpeed);
    return () => {
      preference.removeEventListener("change", changeSpeed);
      loop.cancel();
    };
  }, []);

  return (
    <svg className="run-orbit-mark" viewBox="0 0 120 120"
      fill="none" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      <g className="run-orbit-mark__track">
        <path d={orbit} stroke="#aac9db" strokeWidth="2.1" strokeOpacity=".7" />
        <g className="run-orbit-mark__tracer" ref={tracer} stroke="#ebf7ff">
          {tail.map(({ length, width, opacity }) => (
            <path key={length} d={orbit} pathLength="100" strokeWidth={width} opacity={opacity}
              strokeDasharray={`${length} ${100 - length}`} />
          ))}
          <g strokeDasharray="0 100">
            <path d={orbit} pathLength="100" stroke="#b4deff" strokeWidth="10.6" opacity=".15" />
            <path d={orbit} pathLength="100" stroke="#ecf7ff" strokeWidth="6.5" />
          </g>
        </g>
      </g>
      <g className="run-orbit-mark__glyph" stroke="#f1f7fa" strokeWidth={page === "run" ? 3.65 : 2.8}>
        {page === "configure" ? (
          <>
            <path d="M40 46h21m9 0h10 M40 60h7m9 0h24 M40 74h21m9 0h10" />
            <circle cx="65.5" cy="46" r="4.5" />
            <circle cx="51.5" cy="60" r="4.5" />
            <circle cx="65.5" cy="74" r="4.5" />
          </>
        ) : <path d={page === "results" ? fileArrowDown : triangle} />}
      </g>
    </svg>
  );
}
