/*
 * From Uiverse.io by barisdogansutcu - https://uiverse.io/barisdogansutcu/fuzzy-bullfrog-72
 * MIT License, Copyright (c) barisdogansutcu (Barış Doğan Sütçü)
 */
/*
 * From Uiverse.io by Spacious74 - https://uiverse.io/Spacious74/moody-owl-18
 * MIT License, Copyright (c) Spacious74 (VishalS)
 */
/*
 * From Uiverse.io by Gaurav-WebDev - https://uiverse.io/Gaurav-WebDev/tall-mayfly-60
 * MIT License, Copyright (c) Gaurav-WebDev (Gaurav Sharma)
 */
import { useState } from "react";
import "./ElevationMenu.css";

export function ElevationMenu({ entries, onOpen }) {
  const [highlight, setHighlight] = useState(null);

  function followPointer(event) {
    if (event.pointerType === "touch") return;
    const action = event.target.closest("button")?.dataset.action ?? null;
    if (highlight !== action) setHighlight(action);
  }

  function returnToFocus(event) {
    const focused = event.currentTarget.querySelector("button:focus-visible");
    setHighlight(focused?.dataset.action ?? null);
  }

  return (
    <div
      id="main-entries"
      className="entries elevation-menu"
      onPointerOver={followPointer}
      onPointerMove={followPointer}
      onPointerLeave={returnToFocus}
      onPointerCancel={returnToFocus}
      onFocus={(event) => {
        if (event.target.matches(":focus-visible")) setHighlight(event.target.dataset.action);
      }}
      onBlur={(event) => {
        const action = event.target.dataset.action;
        setHighlight((current) => current === action ? null : current);
      }}
    >
      {entries.map(({ id, title, icon: Icon }) => (
        <button
          key={id}
          type="button"
          className="elevation-entry"
          style={{ "--characters": Array.from(title).length }}
          data-action={id}
          data-highlighted={highlight === id}
          aria-haspopup="dialog"
          onPointerDown={(event) => {
            if (event.pointerType === "touch") setHighlight(id);
          }}
          onPointerUp={(event) => {
            if (event.pointerType === "touch") setHighlight(null);
          }}
          onClick={(event) => onOpen(id, event.currentTarget)}
        >
          <span className="elevation-marker" aria-hidden="true">
            <span className="elevation-marker__active">
              <Icon weight={id === "run" ? "fill" : "bold"} focusable="false" />
            </span>
          </span>
          <span className="elevation-label">{title}</span>
        </button>
      ))}
    </div>
  );
}
