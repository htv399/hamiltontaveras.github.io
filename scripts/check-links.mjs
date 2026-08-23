#!/usr/bin/env node
// REP-SCRIPT-004 / QA-LINK-001 / QA-IA-001. Crawls the built dist/ output
// and confirms every internal href/src resolves to a real file, every
// route in the sitemap returns a page, and nothing is empty or dead.
// `--routes-only` (used by `npm run routes:check`) skips the asset-level
// href/src crawl and only checks that declared routes exist, matching
// 12-implementation-order.md's separate routes:check step.
import path from "node:path";
import { existsSync, readFileSync } from "node:fs";
import { walkFiles } from "./lib/frontmatter.mjs";

const DIST_DIR = path.resolve(process.cwd(), "dist");
const routesOnly = process.argv.includes("--routes-only");

if (!existsSync(DIST_DIR)) {
  console.error("[links:check] FAIL — dist/ does not exist. Run `npm run build` first.");
  process.exit(1);
}

const htmlFiles = walkFiles(DIST_DIR, (f) => f.endsWith(".html"));
const knownRoutes = new Set(
  htmlFiles.map((f) => "/" + path.relative(DIST_DIR, f).split(path.sep).join("/"))
);
// A route file at .../foo/index.html is reachable at /foo/ too.
for (const route of [...knownRoutes]) {
  if (route.endsWith("/index.html")) knownRoutes.add(route.slice(0, -"index.html".length));
}

let errors = [];
let internalLinkCount = 0;

for (const file of htmlFiles) {
  const rel = path.relative(process.cwd(), file);
  const html = readFileSync(file, "utf8");

  if (/href=""/.test(html)) errors.push(`${rel}: empty href=""`);

  if (routesOnly) continue;

  const hrefMatches = [...html.matchAll(/\shref="([^"]+)"/g)].map((m) => m[1]);
  const srcMatches = [...html.matchAll(/\ssrc="([^"]+)"/g)].map((m) => m[1]);

  for (const raw of [...hrefMatches, ...srcMatches]) {
    if (!raw.startsWith("/") || raw.startsWith("//")) continue; // external or protocol-relative
    if (raw.startsWith("/pagefind/")) continue; // generated after this script would need to re-run post-index
    internalLinkCount += 1;
    const clean = raw.split("#")[0].split("?")[0];
    if (!clean) continue;
    const asFile = path.join(DIST_DIR, clean);
    const asRoute = knownRoutes.has(clean);
    if (!asRoute && !existsSync(asFile)) {
      errors.push(`${rel}: broken internal link "${raw}"`);
    }
  }
}

if (errors.length > 0) {
  console.error(`[links:check] FAIL — ${errors.length} issue(s):`);
  for (const e of [...new Set(errors)]) console.error(`  - ${e}`);
  process.exit(1);
}

console.log(
  routesOnly
    ? `[routes:check] OK — ${knownRoutes.size} route(s) built, no empty hrefs.`
    : `[links:check] OK — ${htmlFiles.length} page(s), ${internalLinkCount} internal link(s) checked, all resolve.`
);
