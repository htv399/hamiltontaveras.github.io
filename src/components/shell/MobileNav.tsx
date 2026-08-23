// CMP-MOBILE-NAV. Interactive reason: open state, focus trap and escape
// handling for the mobile menu disclosure. Ships only client:idle.
import { useEffect, useId, useRef, useState } from "react";
import type { NavItem } from "../../config/navigation";
import type { Language } from "../../config/site";

interface Props {
  items: NavItem[];
  language: Language;
  currentPath: string;
  menuLabel: string;
  closeLabel: string;
}

export default function MobileNav({ items, language, currentPath, menuLabel, closeLabel }: Props) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const openButtonRef = useRef<HTMLButtonElement>(null);
  const firstLinkRef = useRef<HTMLAnchorElement>(null);

  useEffect(() => {
    if (!open) return;
    firstLinkRef.current?.focus();

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setOpen(false);
        openButtonRef.current?.focus();
        return;
      }
      if (e.key !== "Tab" || !dialogRef.current) return;
      const focusable = dialogRef.current.querySelectorAll<HTMLElement>("a[href], button:not([disabled])");
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open]);

  return (
    <div className="mobile-nav">
      <button
        ref={openButtonRef}
        type="button"
        className="mobile-nav__trigger"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((v) => !v)}
      >
        {menuLabel}
      </button>
      {open && (
        <div
          id={panelId}
          ref={dialogRef}
          role="dialog"
          aria-modal="true"
          aria-label={menuLabel}
          className="mobile-nav__panel"
        >
          <button type="button" className="mobile-nav__close" onClick={() => setOpen(false)}>
            {closeLabel}
          </button>
          <ul className="mobile-nav__list">
            {items.map((item, i) => {
              const href = item.route[language];
              const active = currentPath === href;
              return (
                <li key={item.key}>
                  <a
                    href={href}
                    ref={i === 0 ? firstLinkRef : undefined}
                    aria-current={active ? "page" : undefined}
                    onClick={() => setOpen(false)}
                  >
                    {item.label[language]}
                  </a>
                </li>
              );
            })}
          </ul>
        </div>
      )}
      <style
        // dangerouslySetInnerHTML, not {`...`}: a plain <style> JSX text
        // child gets HTML-entity-escaped during SSR (e.g. `"` -> `&quot;`)
        // but not identically on the client re-render, which is a
        // guaranteed hydration mismatch for any CSS containing quotes.
        dangerouslySetInnerHTML={{
          __html: `
        .mobile-nav__trigger { font-weight: 600; font-size: var(--size-small); color: var(--header-fg, currentColor); }
        .mobile-nav__panel {
          position: fixed; inset: 0; z-index: 900;
          background: var(--color-navy-900); color: var(--color-white);
          padding: var(--space-6) var(--space-5);
          display: flex; flex-direction: column; gap: var(--space-6);
        }
        .mobile-nav__close { align-self: flex-end; color: var(--color-white); font-weight: 600; }
        .mobile-nav__list { display: flex; flex-direction: column; gap: var(--space-5); }
        .mobile-nav__list a { font-size: var(--size-h3); font-family: var(--font-ui); color: var(--color-white); }
        .mobile-nav__list a[aria-current="page"] { color: var(--color-focus); font-weight: 600; }
      `
        }}
      />
    </div>
  );
}
