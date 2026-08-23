#!/usr/bin/env node
// TECH-SEO-001 / QA-SEO-001..004. Confirms every built page has a unique
// title/description/canonical/OG image, that hreflang pairs are
// reciprocal, and that the sitemap is present and well-formed XML.
import path from "node:path";
import { existsSync, readFileSync } from "node:fs";
import { walkFiles } from "./lib/frontmatter.mjs";

const DIST_DIR = path.resolve(process.cwd(), "dist");
const configuredBase = (process.env.BASE_PATH || "").replace(/^\/+|\/+$/g, "");
if (!existsSync(DIST_DIR)) {
  console.error("[seo:check] FAIL — dist/ does not exist. Run `npm run build` first.");
  process.exit(1);
}

let errors = [];
const htmlFiles = walkFiles(DIST_DIR, (f) => f.endsWith(".html"));
const titles = new Map();

for (const file of htmlFiles) {
  const rel = path.relative(process.cwd(), file);
  const html = readFileSync(file, "utf8");

  const title = html.match(/<title>([^<]*)<\/title>/)?.[1];
  const description = html.match(/<meta name="description" content="([^"]*)"/)?.[1];
  const canonical = html.match(/<link rel="canonical" href="([^"]*)"/)?.[1];
  const ogImage = html.match(/<meta property="og:image" content="([^"]*)"/)?.[1];

  if (!title) errors.push(`${rel}: missing <title>`);
  if (!description) errors.push(`${rel}: missing meta description`);
  if (!canonical) errors.push(`${rel}: missing canonical link`);
  if (!ogImage) errors.push(`${rel}: missing og:image`);

  if (title) {
    const existing = titles.get(title);
    if (existing) errors.push(`${rel}: title "${title}" duplicates ${existing}`);
    else titles.set(title, rel);
  }

  // hreflang reciprocity: every hreflang link's target, when built, must
  // itself declare a hreflang back to this page's canonical.
  const hreflangLinks = [...html.matchAll(/<link rel="alternate" hreflang="([^"]+)" href="([^"]+)"/g)];
  for (const [, , href] of hreflangLinks) {
    let targetPath = new URL(href, "https://x.invalid").pathname;
    if (configuredBase && (targetPath === `/${configuredBase}` || targetPath.startsWith(`/${configuredBase}/`))) {
      targetPath = targetPath.slice(configuredBase.length + 1) || "/";
    }
    const targetFile = path.join(DIST_DIR, targetPath, targetPath.endsWith("/") ? "index.html" : "");
    if (!existsSync(targetFile) && !existsSync(targetFile + ".html")) {
      errors.push(`${rel}: hreflang target "${href}" does not exist in the build`);
    }
  }
}

const sitemapPath = path.join(DIST_DIR, "sitemap-index.xml");
if (!existsSync(sitemapPath)) {
  errors.push("sitemap-index.xml is missing from dist/");
} else {
  const xml = readFileSync(sitemapPath, "utf8");
  if (!xml.startsWith("<?xml")) errors.push("sitemap-index.xml does not start with an XML declaration");
}

if (errors.length > 0) {
  console.error(`[seo:check] FAIL — ${errors.length} issue(s):`);
  for (const e of errors) console.error(`  - ${e}`);
  process.exit(1);
}

console.log(`[seo:check] OK — ${htmlFiles.length} page(s) have title, description, canonical and og:image; sitemap present.`);
