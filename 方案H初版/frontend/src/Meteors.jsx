import { useEffect, useRef, useState } from "react";
import "./Meteors.css";

const between = (minimum, maximum) =>
  minimum + Math.random() * (maximum - minimum);

/** Decorative meteors, positioned in the free sky above the actual controls. */
export function Meteors() {
  const layerRef = useRef(null);
  const nextId = useRef(0);
  const [meteors, setMeteors] = useState([]);

  useEffect(() => {
    const timers = new Set();
    let disposed = false;

    function later(callback, delay) {
      const timer = window.setTimeout(() => {
        timers.delete(timer);
        if (!disposed && !document.hidden) callback();
      }, delay);
      timers.add(timer);
    }

    function clearTimers() {
      for (const timer of timers) window.clearTimeout(timer);
      timers.clear();
    }

    function skyBounds() {
      const layer = layerRef.current;
      if (!layer) return null;
      const bounds = layer.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return null;

      const homepage = layer.closest(".homepage");
      const header = homepage?.querySelector(".topbar")?.getBoundingClientRect();
      const entries = homepage?.querySelector(".entries")?.getBoundingClientRect();
      const firstCard = homepage
        ?.querySelector(".entries > button")
        ?.getBoundingClientRect();
      const landscape = homepage
        ?.querySelector(".scene-plane")
        ?.getBoundingClientRect();
      const mobile = bounds.width <= 800;
      const compact = mobile && header && entries && entries.top - header.bottom < 90;
      const clearance = compact ? 4 : 14;
      const top = header ? header.bottom - bounds.top + clearance : bounds.height * 0.14;
      // Even the last pixels of the trail remain above the UI and dome.
      const skyBottom = Math.min(
        bounds.height * 0.48,
        landscape
          ? landscape.top - bounds.top + landscape.height * 0.52 - 14
          : bounds.height * 0.48,
      );
      let bottom = Math.min(
        skyBottom,
        entries ? entries.top - bounds.top - clearance : bounds.height * 0.43,
      );
      let width = bounds.width;
      // Tall desktop cards can leave no sky above them. Use the open column
      // to their left, still ending above the observatory's fixed dome.
      if (!mobile && firstCard && bottom - top < 60) {
        const leftSkyWidth = Math.min(
          bounds.width,
          firstCard.left - bounds.left - clearance,
        );
        if (leftSkyWidth >= 160) {
          width = leftSkyWidth;
          bottom = skyBottom;
        }
      }
      if (bottom - top < 12) return null;
      return { width, top, height: bottom - top, mobile };
    }

    function releaseMeteor(region, companion = false) {
      const { width, top, height, mobile } = region;
      const startX = width * between(companion ? 0.32 : 0.16, companion ? 0.46 : 0.32);
      const startY = top + height * between(companion ? 0.51 : 0.3, companion ? 0.64 : 0.42);
      const dx = Math.min(
        width * between(mobile ? 0.48 : 0.32, mobile ? 0.64 : 0.48),
        width * 0.95 - startX,
      );
      const length =
        Math.min(width * 0.23, between(mobile ? 76 : 120, mobile ? 104 : 190)) *
        (companion ? 0.77 : 1);
      const dy = Math.min(
        dx * between(0.16, 0.28),
        top + height * 0.9 - startY,
        ((startY - top) * dx * 0.86) / length,
      );
      const duration = between(mobile ? 1250 : 1150, mobile ? 1550 : 1450);
      const id = ++nextId.current;
      const meteor = {
        id,
        x: startX,
        y: startY - top,
        skyTop: top,
        skyHeight: height,
        skyWidth: width,
        dx,
        dy,
        duration,
        direction: (Math.atan2(dy, dx) * 180) / Math.PI,
        length,
        brightness: companion ? 0.76 : between(0.85, 1),
      };
      setMeteors((current) => [...current, meteor]);
      later(() => {
        setMeteors((current) => current.filter((item) => item.id !== id));
      }, duration + 100);
    }

    function group() {
      const region = skyBounds();
      if (region) {
        releaseMeteor(region);
        // Keep pairs occasional and separated vertically; small short screens
        // retain a single clean streak rather than two overlapping flashes.
        if (region.height >= 60 && Math.random() < 0.24) {
          later(() => {
            const currentRegion = skyBounds();
            if (currentRegion?.height >= 60) releaseMeteor(currentRegion, true);
          }, between(230, 420));
        }
      }
      // A pass lasts about 1.4 s, leaving several quiet seconds between groups.
      later(group, between(4200, 6500));
    }

    function restart() {
      clearTimers();
      setMeteors([]);
      if (!document.hidden) later(group, 1050);
    }

    // Hidden tabs have neither live particles nor scheduled timers. On return,
    // start a fresh pass instead of replaying a backlog of missed meteors.
    document.addEventListener("visibilitychange", restart);
    window.addEventListener("resize", restart);
    restart();
    return () => {
      disposed = true;
      clearTimers();
      document.removeEventListener("visibilitychange", restart);
      window.removeEventListener("resize", restart);
    };
  }, []);

  return (
    <div ref={layerRef} className="meteors-layer" aria-hidden="true">
      {meteors.map((meteor) => (
        <span
          key={meteor.id}
          className="meteors-lane"
          style={{ top: meteor.skyTop, height: meteor.skyHeight, width: meteor.skyWidth }}
        >
          <span
            className="meteors-flight"
            style={{
              left: meteor.x,
              top: meteor.y,
              "--meteor-dx": `${meteor.dx}px`,
              "--meteor-dy": `${meteor.dy}px`,
              "--meteor-duration": `${meteor.duration}ms`,
              "--meteor-direction": `${meteor.direction}deg`,
              "--meteor-length": `${meteor.length}px`,
              "--meteor-brightness": meteor.brightness,
            }}
          >
            <span className="meteors-streak" />
          </span>
        </span>
      ))}
    </div>
  );
}
