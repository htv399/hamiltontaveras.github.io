// CMP-FILTERS. Client-side filtering of a static index. URL query reflects
// state so a filtered view is shareable and survives reload; "clear all" is
// a real <button> reachable and operable by keyboard.
import { useEffect, useMemo, useState } from "react";

export interface FilterDimension {
  key: string;
  label: string;
  options: { value: string; label: string }[];
}
export interface SearchRecord {
  id: string;
  title: string;
  href: string;
  summary: string;
  publishedAt?: string;
  tags: Record<string, string[]>;
}
interface Props {
  items: SearchRecord[];
  dimensions: FilterDimension[];
  clearLabel: string;
  emptyHeading: string;
}

function readInitialState(dimensions: FilterDimension[]): Record<string, string[]> {
  if (typeof window === "undefined") return {};
  const params = new URLSearchParams(window.location.search);
  const state: Record<string, string[]> = {};
  for (const dim of dimensions) {
    const value = params.get(dim.key);
    if (value) state[dim.key] = value.split(",");
  }
  return state;
}

export default function ContentFilters({ items, dimensions, clearLabel, emptyHeading }: Props) {
  const [active, setActive] = useState<Record<string, string[]>>(() => readInitialState(dimensions));

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    for (const dim of dimensions) {
      if (active[dim.key]?.length) params.set(dim.key, active[dim.key].join(","));
      else params.delete(dim.key);
    }
    const query = params.toString();
    const url = `${window.location.pathname}${query ? `?${query}` : ""}`;
    window.history.replaceState({}, "", url);
  }, [active, dimensions]);

  function toggle(dimKey: string, value: string) {
    setActive((prev) => {
      const current = prev[dimKey] ?? [];
      const next = current.includes(value) ? current.filter((v) => v !== value) : [...current, value];
      return { ...prev, [dimKey]: next };
    });
  }

  function clearAll() {
    setActive({});
  }

  const filtered = useMemo(() => {
    return items.filter((item) =>
      dimensions.every((dim) => {
        const selected = active[dim.key];
        if (!selected || selected.length === 0) return true;
        return selected.some((v) => item.tags[dim.key]?.includes(v));
      })
    );
  }, [items, dimensions, active]);

  const hasActiveFilters = Object.values(active).some((v) => v.length > 0);
  if (dimensions.every((d) => d.options.length < 2)) {
    // CMP-FILTERS rule: no filters when fewer than two meaningful values.
    return null;
  }

  return (
    <div className="content-filters">
      <div className="content-filters__controls">
        {dimensions.map((dim) => (
          <fieldset key={dim.key} className="content-filters__group">
            <legend className="u-tag">{dim.label}</legend>
            {dim.options.map((opt) => {
              const checked = active[dim.key]?.includes(opt.value) ?? false;
              return (
                <label key={opt.value} className="content-filters__option">
                  <input type="checkbox" checked={checked} onChange={() => toggle(dim.key, opt.value)} />
                  {opt.label}
                </label>
              );
            })}
          </fieldset>
        ))}
        {hasActiveFilters && (
          <button type="button" onClick={clearAll} className="content-filters__clear">
            {clearLabel}
          </button>
        )}
      </div>

      <p className="text-small" role="status" aria-live="polite">
        {filtered.length}/{items.length}
      </p>

      {filtered.length === 0 ? (
        <p className="content-filters__empty">{emptyHeading}</p>
      ) : (
        <ul className="content-filters__results">
          {filtered.map((item) => (
            <li key={item.id}>
              <a href={item.href}>
                <span className="text-body-l">{item.title}</span>
                <span className="text-small">{item.summary}</span>
              </a>
            </li>
          ))}
        </ul>
      )}

      <style
        dangerouslySetInnerHTML={{
          __html: `
        .content-filters__controls { display: flex; flex-wrap: wrap; gap: var(--space-5); align-items: flex-start; border-bottom: var(--border-hairline); padding-bottom: var(--space-4); }
        .content-filters__group { border: 0; display: flex; flex-direction: column; gap: var(--space-1); }
        .content-filters__option { display: flex; align-items: center; gap: var(--space-1); font-size: var(--size-small); }
        .content-filters__clear { align-self: flex-end; font-size: var(--size-small); font-weight: 600; text-decoration: underline; }
        .content-filters__results { display: flex; flex-direction: column; gap: var(--space-3); margin-top: var(--space-4); }
        .content-filters__results a { display: flex; flex-direction: column; gap: var(--space-1); border-top: var(--border-hairline); padding-top: var(--space-3); }
      `
        }}
      />
    </div>
  );
}
