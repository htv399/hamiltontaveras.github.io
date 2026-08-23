#!/usr/bin/env node
// REP-SCRIPT-007 / TECH-SEO-001. Renders the default Open Graph image from
// an SVG template using `sharp`, which is already a transitive dependency
// of astro's image pipeline (no new top-level dependency added — see
// scripts/generate-og.mjs's use in package.json "og:generate").
import path from "node:path";
import { mkdirSync, writeFileSync } from "node:fs";
import sharp from "sharp";

const OUT_DIR = path.resolve(process.cwd(), "public/og");
mkdirSync(OUT_DIR, { recursive: true });

function escapeXml(s) {
  return s.replace(/[<>&'"]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", "'": "&apos;", '"': "&quot;" }[c]));
}

function buildSvg({ name, descriptor }) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <rect width="1200" height="630" fill="#0B1F33"/>
  <rect x="80" y="500" width="120" height="6" fill="#C58A2A"/>
  <text x="80" y="300" font-family="Georgia, 'Source Serif 4', serif" font-size="64" font-weight="600" fill="#FAFBFC">${escapeXml(name)}</text>
  <text x="80" y="360" font-family="Arial, 'IBM Plex Sans', sans-serif" font-size="30" fill="#E8EEF3">${escapeXml(descriptor)}</text>
</svg>`;
}

async function main() {
  const svg = buildSvg({
    name: "Hamilton Taveras",
    descriptor: "Economist working across data systems, quantitative analysis, and valuation."
  });
  const pngPath = path.join(OUT_DIR, "default.png");
  await sharp(Buffer.from(svg)).png().toFile(pngPath);
  writeFileSync(path.join(OUT_DIR, "default.svg"), svg);
  console.log(`[og:generate] Wrote ${pngPath}`);
}

main().catch((err) => {
  console.error("[og:generate] FAIL —", err.message);
  process.exit(1);
});
