export function WorkspaceDownloadIcon() {
  return (
    <svg className="workspace-download-icon" width="20" height="20" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round"
      strokeLinejoin="round" aria-hidden="true" focusable="false">
      <path className="workspace-download-icon__arrow" d="M12 3.5v11m-4.1-4.1 4.1 4.1 4.1-4.1" />
      <path d="M3.8 15v4.2a1.2 1.2 0 0 0 1.2 1.2h14a1.2 1.2 0 0 0 1.2-1.2V15" />
    </svg>
  );
}

// Animate the request itself. Native links and the JSON button still download immediately.
export function animateWorkspaceDownload(event) {
  const arrow = event.currentTarget.querySelector(".workspace-download-icon__arrow");
  if (!arrow?.animate || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const from = getComputedStyle(arrow).transform;
  for (const animation of arrow.getAnimations()) {
    if (animation.id === "workspace-download") animation.cancel();
  }
  arrow.animate([
    { transform: from },
    { transform: "translateY(-2.5px)", offset: .18 },
    { transform: "translateY(4px)", offset: .6 },
    { offset: 1 },
  ], { id: "workspace-download", duration: 440, easing: "cubic-bezier(.22,.72,.3,1)" });
}
