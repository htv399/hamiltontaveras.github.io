#!/usr/bin/env node
// TECH-PERF-001 / QA-PERF-002. Checks the static, measurable budgets against
// the built artifact: total font weight and per-route JS (gzip). Lighthouse
// budgets (QA-PERF-001) need a running server and browser and are not
// re-implemented here; they are covered by the Playwright e2e/perf pass
// documented in the final report.
import path from "node:path";
import { existsSync, readdirSync, statSync, readFileSync } from "node:fs";
import { gzipSync } from "node:zlib";

const DIST_DIR = path.resolve(process.cwd(), "dist");
const PUBLIC_FONTS_DIR = path.resolve(process.cwd(), "public/fonts");

const BUDGETS = {
  fontTotalKb: 300,
  routeJsDefaultKbGzip: 35,
  initialJsKbGzip: 90
};

if (!existsSync(DIST_DIR)) {
  console.error("[perf:budget] FAIL — dist/ does not exist. Run `npm run build` first.");
  process.exit(1);
}

let errors = [];

// Fonts
let fontBytes = 0;
if (existsSync(PUBLIC_FONTS_DIR)) {
  for (const f of readdirSync(PUBLIC_FONTS_DIR)) {
    if (f.endsWith(".woff2")) fontBytes += statSync(path.join(PUBLIC_FONTS_DIR, f)).size;
  }
}
const fontKb = fontBytes / 1024;
if (fontKb > BUDGETS.fontTotalKb) errors.push(`Fonts total ${fontKb.toFixed(1)}KB exceeds ${BUDGETS.fontTotalKb}KB budget`);
else console.log(`[perf:budget] Fonts: ${fontKb.toFixed(1)}KB / ${BUDGETS.fontTotalKb}KB`);

// Per-route JS: the shared Astro client runtime is the "initial" bundle;
// each island chunk is a "route" bundle since islands hydrate per-page.
const astroDir = path.join(DIST_DIR, "_astro");
if (existsSync(astroDir)) {
  for (const f of readdirSync(astroDir)) {
    if (!f.endsWith(".js")) continue;
    const full = path.join(astroDir, f);
    const gzipKb = gzipSync(readFileSync(full)).length / 1024;
    const isInitialRuntime = f.startsWith("client.");
    const budget = isInitialRuntime ? BUDGETS.initialJsKbGzip : BUDGETS.routeJsDefaultKbGzip;
    const label = isInitialRuntime ? "initial runtime" : "route chunk";
    if (gzipKb > budget) errors.push(`${f} (${label}) is ${gzipKb.toFixed(1)}KB gzip, exceeds ${budget}KB budget`);
    else console.log(`[perf:budget] ${f} (${label}): ${gzipKb.toFixed(1)}KB gzip / ${budget}KB`);
  }
}

if (errors.length > 0) {
  console.error(`[perf:budget] FAIL — ${errors.length} issue(s):`);
  for (const e of errors) console.error(`  - ${e}`);
  process.exit(1);
}
console.log("[perf:budget] OK.");
