import "./WorkspaceDrawer.css";
import { RunLaunchBadge } from "./RunLaunchBadge";
import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { XIcon } from "@phosphor-icons/react";

export function WorkspaceDrawer({ runOrigin, workbenchUrl, workspacePage, title, subtitle, onClose, children, returnFocusRef }) {
  const headingId = useId();
  const subtitleId = useId();
  const bodyId = useId();
  const dialogRef = useRef(null);
  const badgeRef = useRef(null);
  const returnFlight = useRef(null);
  const headingRef = useRef(null);
  const bodyRef = useRef(null);
  const triggerRef = useRef(null);
  const openingPageRef = useRef(workspacePage);
  const previousPageRef = useRef(workspacePage);
  const interactionRef = useRef({ resize: null, backdropDown: false });
  const [preferredWidth, setPreferredWidth] = useState(null);
  const [viewport, setViewport] = useState(() => window.innerWidth);
  const [closing, setClosing] = useState(false);
  const mobile = viewport <= 700;
  const maximum = mobile ? viewport : Math.floor(viewport * 0.9);
  const minimum = mobile ? viewport : Math.min(480, maximum);
  const initialWidth = viewport * (openingPageRef.current === "results" ? 0.79 : 0.72);
  const width = mobile ? viewport : Math.round(Math.max(
    minimum,
    Math.min(maximum, preferredWidth ?? initialWidth),
  ));

  function stopResize() {
    const drag = interactionRef.current.resize;
    interactionRef.current.resize = null;
    if (drag?.element.hasPointerCapture(drag.pointerId)) {
      drag.element.releasePointerCapture(drag.pointerId);
    }
  }

  function requestClose() {
    if (closing) return;
    stopResize();
    returnFlight.current = badgeRef.current?.leave();
    setClosing(true);
  }

  useLayoutEffect(() => {
    const dialog = dialogRef.current;
    triggerRef.current = document.activeElement;
    if (!dialog.open) dialog.showModal();
    badgeRef.current?.enter();
    return () => {
      stopResize();
      if (dialog.open) dialog.close();
      const trigger = returnFocusRef?.current || triggerRef.current;
      if (trigger?.isConnected) trigger.focus({ preventScroll: true });
    };
  }, []);

  useLayoutEffect(() => {
    const body = bodyRef.current;
    body.scrollTop = 0;
    headingRef.current?.focus({ preventScroll: true });
    if (previousPageRef.current === workspacePage) return;
    previousPageRef.current = workspacePage;
    // The sheet stays in place when changing between configuration and review.
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const reveal = body.animate([
      { opacity: 0, transform: reducedMotion ? "none" : "translateY(6px)" },
      { opacity: 1, transform: "none" },
    ], { duration: reducedMotion ? 200 : 240, easing: "cubic-bezier(.2,.75,.3,1)" });
    return () => reveal.cancel();
  }, [workspacePage]);

  useEffect(() => {
    let frame = 0;
    function onResize() {
      if (!frame) frame = requestAnimationFrame(() => {
        frame = 0;
        setViewport(window.innerWidth);
      });
    }
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      cancelAnimationFrame(frame);
    };
  }, []);

  useEffect(() => {
    if (mobile) stopResize();
  }, [mobile]);

  useEffect(() => {
    if (!closing) return;
    let active = true;
    const dialog = dialogRef.current;
    const exits = dialog.getAnimations().filter(
      (animation) => ["transform", "clip-path"].includes(animation.transitionProperty),
    );
    // Native transitions can reverse an unfinished entrance at its current position.
    Promise.allSettled([...exits.map((animation) => animation.finished), returnFlight.current]).then(() => {
      if (active) onClose();
    });
    return () => { active = false; };
  }, [closing, onClose]);

  function isOutside(event) {
    if (event.target !== event.currentTarget) return false;
    const rect = event.currentTarget.getBoundingClientRect();
    return event.clientX < rect.left || event.clientX > rect.right ||
      event.clientY < rect.top || event.clientY > rect.bottom;
  }

  function beginResize(event) {
    if (mobile || closing || event.button !== 0 || !event.isPrimary) return;
    event.preventDefault();
    event.currentTarget.focus({ preventScroll: true });
    interactionRef.current.resize = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startWidth: width,
      element: event.currentTarget,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function moveResize(event) {
    const drag = interactionRef.current.resize;
    if (!drag || drag.pointerId !== event.pointerId || mobile || closing) return;
    setPreferredWidth(Math.max(minimum, Math.min(
      maximum,
      drag.startWidth + drag.startX - event.clientX,
    )));
  }

  function resizeWithKeyboard(event) {
    if (mobile || closing) return;
    const step = event.shiftKey ? 64 : 24;
    let next = width;
    if (event.key === "ArrowLeft") next += step;
    else if (event.key === "ArrowRight") next -= step;
    else if (event.key === "Home") next = minimum;
    else if (event.key === "End") next = maximum;
    else return;
    event.preventDefault();
    setPreferredWidth(Math.max(minimum, Math.min(maximum, next)));
  }

  return (
    <dialog
      ref={dialogRef}
      className="dialog workspace-drawer"
      data-workspace-page={workspacePage}
      data-night-workspace=""
      style={{ "--drawer-width": `${width}px` }}
      data-closing={closing ? "true" : undefined}
      aria-labelledby={headingId}
      aria-describedby={subtitle ? subtitleId : undefined}
      onCancel={(event) => { event.preventDefault(); requestClose(); }}
      onPointerDown={(event) => {
        interactionRef.current.backdropDown = isOutside(event);
      }}
      onClick={(event) => {
        if (interactionRef.current.backdropDown && isOutside(event)) requestClose();
        interactionRef.current.backdropDown = false;
      }}
    >
      <div
        className="workspace-drawer__resize"
        role="separator"
        aria-label="调整面板宽度"
        aria-orientation="vertical"
        aria-controls={bodyId}
        aria-valuemin={minimum}
        aria-valuemax={maximum}
        aria-valuenow={width}
        aria-valuetext={`${width} 像素`}
        hidden={mobile}
        tabIndex={mobile ? -1 : 0}
        onPointerDown={beginResize}
        onPointerMove={moveResize}
        onPointerUp={stopResize}
        onPointerCancel={stopResize}
        onLostPointerCapture={stopResize}
        onKeyDown={resizeWithKeyboard}
      />
      <RunLaunchBadge
        key={workspacePage}
        ref={badgeRef}
        origin={runOrigin?.button.dataset.action === workspacePage ? runOrigin : null}
        page={workspacePage}
        title={title}
        href={workbenchUrl}
      />
      <div className="dialog-head workspace-drawer__head">
        <div>
          <h2 ref={headingRef} id={headingId} tabIndex={-1}>{title}</h2>
          {subtitle && <p id={subtitleId}>{subtitle}</p>}
        </div>
        <button type="button" className="icon-button" aria-label="关闭面板" onClick={requestClose}>
          <XIcon size={26} weight="light" aria-hidden="true" focusable="false" />
        </button>
      </div>
      <div ref={bodyRef} id={bodyId} className="dialog-body workspace-drawer__body">
        {children}
      </div>
    </dialog>
  );
}
