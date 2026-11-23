import { useEffect, useRef } from "react";
import { createLandscapeRenderer } from "./landscapeRenderer";
import { createLandscapeFallback } from "./landscapeFallback";

// Both crops use the same frame. Match the mobile sky to its CSS background.
export function LandscapeMotion({ imageRef, mobileSkyRef }) {
  const mount = useRef(null);

  useEffect(() => {
    const host = mount.current;
    const image = imageRef.current;
    const mobileSky = mobileSkyRef.current;
    const compact = matchMedia("(max-width: 800px)");
    let skyContext, renderer, canvas, frameId;
    let seconds = 0;
    let previousFrame = null;
    let failed = false;
    let skyCrop;

    function resize() {
      if (!renderer) return;
      const width = Math.min(image.naturalWidth,
        Math.max(1, Math.round(host.clientWidth * Math.min(window.devicePixelRatio, 1.5))));
      const height = Math.round(width * image.naturalHeight / image.naturalWidth);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      skyCrop = null;
      if (compact.matches) {
        const scene = mobileSky.parentElement;
        const skyScale = Number(getComputedStyle(scene).getPropertyValue("--mobile-sky-scale"));
        const photoHeight = scene.clientHeight * skyScale;
        const scale = photoHeight / image.naturalHeight;
        const photoWidth = image.naturalWidth * scale;
        // The lower scene is fully opaque from 32% down its photograph.
        const until = Math.min(scene.clientHeight, Math.ceil(
          host.getBoundingClientRect().top - scene.getBoundingClientRect().top + host.clientHeight * 0.32));
        mobileSky.width = scene.clientWidth;
        mobileSky.height = Math.max(1, until);
        mobileSky.style.height = mobileSky.height + "px";
        skyContext ??= mobileSky.getContext("2d", { alpha: false });
        skyCrop = {
          left: (photoWidth - scene.clientWidth) * 0.25 / photoWidth * width,
          width: scene.clientWidth / photoWidth * width,
          height: mobileSky.height / photoHeight * height,
        };
      }
      paint();
    }

    function paint() {
      renderer.draw(seconds);
      if (skyCrop && skyContext) {
        skyContext.drawImage(canvas, skyCrop.left, 0, skyCrop.width, skyCrop.height,
          0, 0, mobileSky.width, mobileSky.height);
        mobileSky.hidden = false;
      }
    }

    function initialize(useFallback = false) {
      if (renderer || failed || !image.complete || !image.naturalWidth) return;
      canvas = document.createElement("canvas");
      if (!useFallback) renderer = createLandscapeRenderer(canvas, image);
      if (!renderer) {
        canvas = document.createElement("canvas");
        try {
          renderer = createLandscapeFallback(canvas, image);
        } catch {
          // Keep the photograph visible when the device cannot allocate a canvas.
          renderer = null;
        }
      } else {
        canvas.addEventListener("webglcontextlost", onContextLoss);
      }
      if (!renderer) { failed = true; return; }
      host.replaceChildren(canvas);
      resize();
    }

    function stopClock() {
      cancelAnimationFrame(frameId);
      frameId = undefined;
      previousFrame = null;
    }

    function update() {
      stopClock();
      if (document.hidden) return;
      initialize();
      if (renderer) frameId = requestAnimationFrame(tick);
    }

    function tick(now) {
      // About 30 frames per second; no React state changes on animation frames.
      if (previousFrame === null || now - previousFrame >= 31) {
        if (previousFrame !== null) seconds += (now - previousFrame) / 1000;
        previousFrame = now;
        paint();
      }
      frameId = requestAnimationFrame(tick);
    }

    function onContextLoss(event) {
      event.preventDefault();
      stopClock();
      canvas.removeEventListener("webglcontextlost", onContextLoss);
      renderer.destroy();
      renderer = null;
      host.replaceChildren();
      mobileSky.hidden = true;
      initialize(true);
      update();
    }

    const observer = new ResizeObserver(resize);
    observer.observe(host);
    observer.observe(mobileSky.parentElement);
    image.addEventListener("load", update);
    document.addEventListener("visibilitychange", update);
    update();
    return () => {
      stopClock();
      observer.disconnect();
      image.removeEventListener("load", update);
      document.removeEventListener("visibilitychange", update);
      canvas?.removeEventListener("webglcontextlost", onContextLoss);
      renderer?.destroy();
      host.replaceChildren();
      mobileSky.hidden = true;
    };
  }, [imageRef, mobileSkyRef]);

  return <div ref={mount} className="landscape-motion" />;
}
