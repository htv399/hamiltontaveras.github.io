// CMP-CHART. Interactive reason: series selection and tooltip on evidence
// that a static Figure cannot express. No charting dependency is approved
// by TECH-001 (required_capabilities does not list one), so this draws a
// minimal accessible SVG chart by hand instead of adding a library. Always
// hydrated with client:visible from the caller so it never blocks the
// initial paint budget.
import { useId, useState } from "react";

export interface ChartSeries {
  label: string;
  color: string;
  points: { x: string; y: number }[];
}
export interface ChartSpec {
  title: string;
  unit: string;
  series: ChartSeries[];
  dataVintage: string;
  source: string;
  accessibleSummary: string;
}
export interface DataTableSpec {
  caption: string;
  columns: string[];
  rows: (string | number)[][];
}
interface Props {
  spec: ChartSpec;
  fallbackTable: DataTableSpec;
}

const WIDTH = 640;
const HEIGHT = 280;
const PADDING = 32;

export default function InteractiveChart({ spec, fallbackTable }: Props) {
  const [activeSeries, setActiveSeries] = useState<string | null>(null);
  const titleId = useId();
  const descId = useId();

  const allY = spec.series.flatMap((s) => s.points.map((p) => p.y));
  const min = Math.min(...allY, 0);
  const max = Math.max(...allY, 1);
  const xLabels = spec.series[0]?.points.map((p) => p.x) ?? [];

  function toXY(i: number, y: number, n: number) {
    const x = PADDING + (i / Math.max(n - 1, 1)) * (WIDTH - PADDING * 2);
    const yy = HEIGHT - PADDING - ((y - min) / (max - min || 1)) * (HEIGHT - PADDING * 2);
    return [x, yy];
  }

  return (
    <div className="interactive-chart">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-labelledby={titleId}
        aria-describedby={descId}
        className="interactive-chart__svg"
      >
        <title id={titleId}>{spec.title}</title>
        <desc id={descId}>{spec.accessibleSummary}</desc>
        <line x1={PADDING} y1={HEIGHT - PADDING} x2={WIDTH - PADDING} y2={HEIGHT - PADDING} stroke="var(--color-slate-500)" strokeWidth="1" />
        {spec.series.map((series) => {
          const dimmed = activeSeries && activeSeries !== series.label;
          const points = series.points.map((p, i) => toXY(i, p.y, series.points.length));
          const path = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ");
          return (
            <path
              key={series.label}
              d={path}
              fill="none"
              stroke={series.color}
              strokeWidth={dimmed ? 1 : 2}
              opacity={dimmed ? 0.3 : 1}
              className="interactive-chart__line"
            />
          );
        })}
      </svg>
      <div className="interactive-chart__ticks" aria-hidden="true">
        {xLabels.map((label) => (
          <span key={label}>{label}</span>
        ))}
      </div>
      <div className="interactive-chart__legend" role="group" aria-label="Series">
        {spec.series.map((series) => (
          <button
            key={series.label}
            type="button"
            aria-pressed={activeSeries === series.label}
            onClick={() => setActiveSeries((cur) => (cur === series.label ? null : series.label))}
            className="interactive-chart__legend-item"
          >
            <span className="interactive-chart__swatch" style={{ background: series.color }} aria-hidden="true" />
            {series.label}
          </button>
        ))}
      </div>
      <p className="text-micro">{spec.unit} · {spec.dataVintage} · {spec.source}</p>

      <details className="interactive-chart__table">
        <summary className="text-small">{fallbackTable.caption}</summary>
        <table className="data-table">
          <caption className="visually-hidden">{fallbackTable.caption}</caption>
          <thead>
            <tr>{fallbackTable.columns.map((c) => <th key={c} scope="col">{c}</th>)}</tr>
          </thead>
          <tbody>
            {fallbackTable.rows.map((row, i) => (
              <tr key={i}>{row.map((cell, j) => (j === 0 ? <th key={j} scope="row">{cell}</th> : <td key={j} data-numeric="true">{cell}</td>))}</tr>
            ))}
          </tbody>
        </table>
      </details>

      <style
        dangerouslySetInnerHTML={{
          __html: `
        .interactive-chart__svg { width: 100%; height: auto; min-height: 280px; }
        .interactive-chart__ticks { display: flex; justify-content: space-between; font-size: var(--size-micro); color: var(--color-slate-500); padding-inline: ${PADDING / WIDTH * 100}%; }
        .interactive-chart__legend { display: flex; flex-wrap: wrap; gap: var(--space-3); margin-top: var(--space-2); }
        .interactive-chart__legend-item { display: inline-flex; align-items: center; gap: var(--space-1); font-size: var(--size-small); }
        .interactive-chart__swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
        .interactive-chart__table { margin-top: var(--space-3); }
        @media (prefers-reduced-motion: no-preference) {
          .interactive-chart__line { transition: opacity var(--duration-standard) var(--easing-standard); }
        }
      `
        }}
      />
    </div>
  );
}
