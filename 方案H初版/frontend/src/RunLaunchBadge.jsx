import { useImperativeHandle, useLayoutEffect, useRef } from "react";
import { RunOrbitMark } from "./RunOrbitMark";
import "./RunLaunchBadge.css";

export function captureWorkspacePill(button) {
  const label = button?.querySelector(".elevation-label");
  if (!label) return null;
  const box = label.getBoundingClientRect();
  if (!box.width || !box.height) return null;
  const css = getComputedStyle(label);
  const scale = box.height / label.offsetHeight;
  return {
    button, label, box, background: css.backgroundColor, shadow: css.boxShadow,
    textColor: css.color, fontFamily: css.fontFamily, fontWeight: css.fontWeight,
    fontSize: parseFloat(css.fontSize) * scale, spacing: parseFloat(css.letterSpacing) * scale || 0,
  };
}

function place(x, y) { return `translate3d(${x}px, ${y}px, 0)`; }

function flightPath(source, target, reduced) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  // Sample a shallow quadratic arc; the animation timing sets the travel speed.
  const lift = reduced ? 0 : Math.min(55, Math.abs(dy) * .17);
  return Array.from({ length: 33 }, (_, index) => {
    const p = index / 32;
    return { transform: place(source.x + dx * p, source.y + dy * p - 4 * lift * p * (1 - p)) };
  });
}

export function RunLaunchBadge({ ref, origin, page, title, href }) {
  const slotRef = useRef(null);
  const layerRef = useRef(null);
  const active = useRef(null);

  function reset() {
    const flight = active.current;
    if (!flight) return;
    active.current = null;
    for (const animation of flight.animations) animation.cancel();
    flight.source.label.removeAttribute("data-in-flight");
    slotRef.current?.removeAttribute("data-in-flight");
    if (layerRef.current?.matches(":popover-open")) layerRef.current.hidePopover();
  }

  function prepare(source) {
    const layer = layerRef.current;
    const slot = slotRef.current;
    const dialog = slot?.closest("dialog");
    if (!dialog?.open || !source || !layer?.showPopover) return null;
    const target = slot.getBoundingClientRect();
    // The sky dock is positioned relative to the sliding drawer. Measure its resting viewport position.
    const offsetX = dialog.getBoundingClientRect().right - document.documentElement.clientWidth;
    const to = { x: target.left + target.width / 2 - offsetX, y: target.top + target.height / 2 };
    const box = source.box;
    const from = { x: box.left + box.width / 2, y: box.top + box.height / 2 };
    const actor = layer.querySelector(".run-flight__actor");
    const shell = layer.querySelector(".run-flight__shell");
    const word = layer.querySelector(".run-flight__word");
    const size = target.width;
    const css = getComputedStyle(dialog);
    const duration = parseFloat(css.getPropertyValue("--drawer-enter-duration"));
    const exitDuration = parseFloat(css.getPropertyValue("--drawer-exit-duration"));
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    layer.style.setProperty("--flight-size", `${size}px`);
    Object.assign(word.style, {
      fontFamily: source.fontFamily, fontWeight: source.fontWeight, fontSize: `${source.fontSize}px`,
      letterSpacing: `${source.spacing}px`, color: source.textColor,
    });
    layer.showPopover();
    source.label.dataset.inFlight = "";
    slot.dataset.inFlight = "";

    const travel = actor.animate(flightPath(from, to, reduced), {
      duration, fill: "both", easing: reduced ? "ease-in-out" : css.getPropertyValue("--drawer-settle").trim(),
    });
    const formTiming = { duration, easing: reduced ? "ease-in-out" : "cubic-bezier(.32,.08,.18,1)", fill: "both" };
    // The shell itself tightens into an ellipse, matching the tilt and radii of the final orbit.
    const orbitWidth = size * 97 / 120;
    const orbitHeight = size * 72.8 / 120;
    const shellShape = (progress, angle, bg, stroke, shadow = "0 0 0 transparent") => ({
      width: `${box.width + (orbitWidth - box.width) * progress}px`,
      height: `${box.height + (orbitHeight - box.height) * progress}px`,
      transform: `translate(-50%, -50%) rotate(${angle}deg)`,
      backgroundColor: bg, borderColor: stroke, boxShadow: shadow,
    });
    const animations = [travel,
      shell.animate([
        { ...shellShape(0, 0, source.background, "transparent", source.shadow), offset: 0 },
        { ...shellShape(.38, -2, source.background, "rgba(180,210,232,.26)"), offset: .31 },
        { ...shellShape(.93, -21, "rgba(30,58,79,.27)", "rgba(180,210,232,.8)"), offset: .73 },
        { ...shellShape(1, -26, "transparent", "transparent"), offset: 1 },
      ], formTiming),
      word.animate([
        { transform: "translate(-50%, -50%) scale(1)", opacity: 1 },
        { transform: "translate(-50%, -50%) scale(.74, .91)", opacity: .85, offset: .22 },
        { transform: "translate(-50%, -50%) scale(.20, .65)", opacity: 0, offset: .52 },
        { transform: "translate(-50%, -50%) scale(.20, .65)", opacity: 0 },
      ], formTiming),
      layer.querySelector(".run-orbit-mark__glyph").animate([
        { opacity: 0, transform: "scale(.45)" }, { opacity: 0, transform: "scale(.5)", offset: .25 },
        { opacity: 1, transform: "scale(1)", offset: .70 }, { opacity: 1, transform: "scale(1)" },
      ], formTiming),
      layer.querySelector(".run-orbit-mark__track").animate([
        { opacity: 0 }, { opacity: 0, offset: .42 }, { opacity: 1 },
      ], formTiming),
    ];
    active.current = { source, animations, travel, duration, exitDuration, returning: false };
    return active.current;
  }

  useImperativeHandle(ref, () => ({
    enter() {
      reset();
      const flight = prepare(origin);
      if (flight) flight.travel.finished.then(() => {
        if (active.current === flight && !flight.returning) reset();
      }, () => {});
    },
    leave() {
      let flight = active.current;
      if (!flight) {
        flight = prepare(captureWorkspacePill(origin?.button));
        if (!flight) return;
        // From a settled dock, ease back along the same path without lingering in the sky.
        flight.travel.effect.updateTiming({ easing: "cubic-bezier(.45,0,.6,1)" });
        for (const animation of flight.animations) animation.currentTime = flight.duration;
      }
      flight.returning = true;
      for (const animation of flight.animations) {
        animation.playbackRate = -flight.duration / flight.exitDuration;
        animation.play();
      }
      return flight.travel.finished.catch(() => {});
    },
  }));
  useLayoutEffect(() => {
    // A layout change establishes a new dock position. Never leave a stale flight over the interface.
    window.addEventListener("resize", reset);
    return () => { window.removeEventListener("resize", reset); reset(); };
  }, []);

  return (
    <>
      {page === "run" ? (
        <a className="run-launch-slot" ref={slotRef} href={href}
          aria-label="查看运行状态" title="查看运行状态">
          <RunOrbitMark />
        </a>
      ) : (
        <span className="run-launch-slot" ref={slotRef} aria-hidden="true">
          <RunOrbitMark page={page} />
        </span>
      )}
      <div className="run-flight" popover="manual" inert aria-hidden="true" ref={layerRef}>
        <div className="run-flight__actor">
          <span className="run-flight__shell" />
          <span className="run-flight__word">{origin?.label.textContent || title}</span>
          <span className="run-flight__symbol"><RunOrbitMark page={page} /></span>
        </div>
      </div>
    </>
  );
}
