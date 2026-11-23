import { useId, useLayoutEffect, useRef, useState } from "react";
import { XIcon } from "@phosphor-icons/react";
import "./AuxiliaryDialog.css";

function onBackdrop(event) {
  const dialog = event.currentTarget;
  if (event.target !== dialog) return false;
  const { left, right, top, bottom } = dialog.getBoundingClientRect();
  return event.clientX < left || event.clientX > right || event.clientY < top || event.clientY > bottom;
}

export function AuxiliaryDialog({ title, subtitle, onClose, children, returnFocusRef }) {
  const dialogRef = useRef(null);
  const backdropDown = useRef(false);
  const [closing, setClosing] = useState(false);
  const titleId = useId();
  const descriptionId = useId();

  useLayoutEffect(() => {
    const dialog = dialogRef.current;
    const trigger = returnFocusRef?.current ?? document.activeElement;
    dialog.showModal();
    return () => {
      dialog.close();
      if (trigger?.isConnected) trigger.focus({ preventScroll: true });
    };
  }, [returnFocusRef]);

  useLayoutEffect(() => {
    if (!closing) return;
    let mounted = true;
    const slide = dialogRef.current.getAnimations().find(animation => animation.transitionProperty === "transform");
    // An interrupted entrance reverses from the position currently on screen.
    const finished = slide?.finished ?? Promise.resolve();
    finished.catch(() => {}).then(() => { if (mounted) onClose(); });
    return () => { mounted = false; };
  }, [closing, onClose]);

  return (
    <dialog ref={dialogRef} className="dialog auxiliary-dialog" data-closing={closing || undefined}
      aria-labelledby={titleId} aria-describedby={subtitle ? descriptionId : undefined}
      onCancel={event => { event.preventDefault(); setClosing(true); }}
      onPointerDown={event => { backdropDown.current = onBackdrop(event); }}
      onClick={event => {
        if (backdropDown.current && onBackdrop(event)) setClosing(true);
        backdropDown.current = false;
      }}>
      <div className="dialog-head">
        <div>
          <h2 id={titleId}>{title}</h2>
          {subtitle && <p id={descriptionId}>{subtitle}</p>}
        </div>
        <button type="button" className="icon-button" aria-label="关闭面板" onClick={() => setClosing(true)}>
          <XIcon size={21} aria-hidden="true" />
        </button>
      </div>
      <div className="dialog-body">{children}</div>
    </dialog>
  );
}
