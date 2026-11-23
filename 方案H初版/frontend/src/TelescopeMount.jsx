import { useEffect, useImperativeHandle, useRef, useState } from "react";
import "./TelescopeMount.css";

export function TelescopeMount({ ref }) {
  const mount = useRef(null);
  const receiver = useRef(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let mounted = true;
    // Reveal all three layers together. A failed decode leaves the original
    // photograph visible, including its complete stationary telescope.
    const layers = Array.from(mount.current.querySelectorAll("img"));
    Promise.all(layers.map((image) => image.decode())).then(
      () => { if (mounted) setReady(true); },
      () => {},
    );
    return () => { mounted = false; };
  }, []);

  useImperativeHandle(ref, () => ({
    receive() {
      if (!ready || document.hidden) return;
      receiver.current.animate(
        [
          { opacity: 0 },
          { opacity: 1, offset: 0.16 },
          { opacity: 0.85, offset: 0.34 },
          { opacity: 0.38, offset: 0.62 },
          { opacity: 0 },
        ],
        { duration: 1250, easing: "ease-in-out" },
      );
    },
  }), [ready]);

  return (
    <div ref={mount} className="telescope-mount" data-ready={ready} aria-hidden="true">
      <img src="/assets/dish-sky-patch.webp" alt="" draggable={false} />
      <div className="telescope-mount__rotor">
        <img src="/assets/dish-reflector.webp" alt="" draggable={false} />
        <span className="telescope-mount__receiver">
          <span ref={receiver} className="telescope-mount__pulse" />
        </span>
      </div>
      <img src="/assets/dish-bearing.webp" alt="" draggable={false} />
    </div>
  );
}
