import "./WorkspaceDrawer.css";
import "./WorkspaceActions.css";
import { RunLaunchBadge } from "./RunLaunchBadge";
import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { XIcon } from "@phosphor-icons/react";

export function WorkspaceDrawer({ runOrigin, workbenchUrl, workspacePage, pageTitles, renderPage, onClose, returnFocusRef }) {
  const headingId = useId();
  const bodyId = useId();
  const dialogRef = useRef(null);
  const badgeRef = useRef(null);
  const returnFlight = useRef(null);
  const headingRef = useRef(null);
  const trackRef = useRef(null);
  const triggerRef = useRef(null);
  const openingPageRef = useRef(workspacePage);
  const [route, setRoute] = useState({ active: workspacePage, leaving: null });
  if (route.active !== workspacePage) {
    setRoute({ active: workspacePage, leaving: route.active });
  }
  // The entry chosen on the homepage is the root of this navigation stack.
  const rootPage = openingPageRef.current;
  const pages = rootPage === "results" ? [rootPage] : [rootPage, rootPage === "run" ? "configure" : "run"];
  const depth = pages.indexOf(workspacePage);
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
    headingRef.current?.focus({ preventScroll: true });
  }, [workspacePage]);

  useLayoutEffect(() => {
    if (!route.leaving) return;
    let current = true;
    function settle() {
      if (current) setRoute({ active: workspacePage, leaving: null });
    }
    // Keep both React trees during travel. CSS can then reverse from the current
    // position on a second click, including a partially completed first hop.
    const slide = trackRef.current.getAnimations().find((a) => a.transitionProperty === "transform");
    if (slide) slide.finished.then(settle, settle);
    else settle();
    return () => { current = false; };
  }, [workspacePage, route.leaving]);

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
      style={{ "--drawer-width": `${width}px`, "--workspace-page-offset": `${-100 * depth}%` }}
      data-closing={closing ? "true" : undefined}
      aria-labelledby={headingId}
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
        origin={runOrigin}
        page={workspacePage}
        title={pageTitles[workspacePage]}
        href={workbenchUrl}
      />
      <div className="dialog-head workspace-drawer__head">
        <div className="workspace-drawer__titles">
          <div className="workspace-drawer__strip">
            {pages.map(page => (
              <div className="workspace-drawer__frame" key={page} inert={page !== workspacePage}>
                {page === workspacePage || page === route.leaving ? (
                  <h2 id={page === workspacePage ? headingId : undefined}
                    ref={page === workspacePage ? headingRef : undefined} tabIndex={-1}>
                    {pageTitles[page]}
                  </h2>
                ) : null}
              </div>
            ))}
          </div>
        </div>
        <button type="button" className="icon-button workspace-action workspace-close" aria-label="关闭面板" onClick={requestClose}>
          <XIcon size={26} weight="light" aria-hidden="true" focusable="false" />
        </button>
      </div>
      <div className="workspace-drawer__pages" id={bodyId}>
        <div className="workspace-drawer__strip" ref={trackRef}>
          {pages.map(page => (
            <div className="workspace-drawer__frame" key={page} data-page={page}
              inert={page !== workspacePage || closing} aria-hidden={page !== workspacePage || undefined}>
              {(page === workspacePage || page === route.leaving) && (
                <div className="dialog-body workspace-drawer__body">{renderPage(page)}</div>
              )}
            </div>
          ))}
        </div>
      </div>
    </dialog>
  );
}
