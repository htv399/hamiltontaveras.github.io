// CMP-FILTER-DRAWER. Mobile-only disclosure around a filter dimension set:
// manages open state and focus without changing desktop layout, where
// filters stay inline.
import { useId, useRef, useState, type ReactNode } from "react";
import type { FilterDimension } from "./ContentFilters";

interface Props {
  dimensions: FilterDimension[];
  triggerLabel: string;
  closeLabel: string;
  children: ReactNode;
}

export default function FilterDrawer({ dimensions, triggerLabel, closeLabel, children }: Props) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const panelRef = useRef<HTMLDivElement>(null);

  if (dimensions.every((d) => d.options.length < 2)) return <>{children}</>;

  return (
    <div className="filter-drawer">
      <button
        type="button"
        className="filter-drawer__trigger"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? closeLabel : triggerLabel}
      </button>
      <div id={panelId} ref={panelRef} className="filter-drawer__panel" hidden={!open}>
        {children}
      </div>
      <style
        dangerouslySetInnerHTML={{
          __html: `
        .filter-drawer__trigger { display: none; font-weight: 600; }
        @media (max-width: 768px) {
          .filter-drawer__trigger { display: inline-flex; }
          .filter-drawer__panel[hidden] { display: none; }
        }
      `
        }}
      />
    </div>
  );
}
