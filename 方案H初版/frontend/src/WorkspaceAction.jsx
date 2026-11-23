/*
 * From Uiverse.io by Shakil-Babu - https://uiverse.io/Shakil-Babu/moody-mule-8
 * MIT License, Copyright (c) Shakil-Babu (Babu)
 */
export function WorkspaceAction({ as: Element = "button", className = "", children, ...props }) {
  return (
    <Element {...props} className={`workspace-action ${className}`.trimEnd()}>
      {children}
    </Element>
  );
}
