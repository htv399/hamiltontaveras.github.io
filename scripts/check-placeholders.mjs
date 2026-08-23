#!/usr/bin/env node
// REP-SCRIPT-005 / CM-VAL-001 / QA-PRD-005 / QA-SEED-003. Final placeholder
// gate. Checks two layers: (1) content frontmatter, same rule as
// content:validate, kept here too so this single command is a complete
// gate on its own; (2) the built dist/ output, scanning rendered HTML for
// literal "coming soon" style text and empty hrefs. The dist scan only
// runs if dist/ exists (it does not block a preview-only invocation before
// the first build).
import path from "node:path";
import { existsSync, readFileSync } from "node:fs";
import { walkFiles, readFrontmatter } from "./lib/frontmatter.mjs";

const SITE_ENV = process.env.SITE_ENV || "preview";
// Case-sensitive \bTODO\b, not /i: a case-insensitive match would flag the
// ordinary Spanish word "método" (contains "todo" as a substring) and
// similar false positives. "próximamente" / "coming soon" stay
// case-insensitive since they are whole-phrase, not substring-prone.
const forbiddenTextPattern = /próximamente|coming soon|placeholder text/i;
const forbiddenTodoPattern = /\bTODO\b/;
let errors = [];

// 1. Content frontmatter.
const CONTENT_DIR = path.resolve(process.cwd(), "src/content");
const collections = ["analysis", "research", "work", "notes", "courses", "weeks", "materials", "profiles"];
let activePlaceholders = 0;
for (const collection of collections) {
  const dir = path.join(CONTENT_DIR, collection);
  for (const file of walkFiles(dir, (f) => f.endsWith(".mdx") || f.endsWith(".yml"))) {
    const fm = readFrontmatter(file);
    if (fm.placeholder === true) {
      activePlaceholders += 1;
      if (SITE_ENV === "production") {
        errors.push(`${path.relative(process.cwd(), file)}: placeholder:true content cannot ship to production.`);
      }
    }
  }
}

// 2. Rendered output, when present.
const DIST_DIR = path.resolve(process.cwd(), "dist");
if (existsSync(DIST_DIR)) {
  const htmlFiles = walkFiles(DIST_DIR, (f) => f.endsWith(".html"));
  for (const file of htmlFiles) {
    const rel = path.relative(process.cwd(), file);
    const html = readFileSync(file, "utf8");
    if (forbiddenTextPattern.test(html) || forbiddenTodoPattern.test(html)) {
      errors.push(`${rel}: contains forbidden placeholder language ("coming soon" / "próximamente" / stray TODO).`);
    }
    if (/href=""/.test(html) || /href="#"[^>]*>/.test(html)) {
      errors.push(`${rel}: contains an empty or dead href="#".`);
    }
  }
  console.log(`[placeholders:check] Scanned ${htmlFiles.length} built page(s) in dist/.`);
} else {
  console.log("[placeholders:check] dist/ not built yet — skipping rendered-output scan.");
}

console.log(`[placeholders:check] ${activePlaceholders} SEED-001 fixture(s) currently carry placeholder:true (expected in preview).`);

if (errors.length > 0) {
  console.error(`[placeholders:check] FAIL — ${errors.length} issue(s):`);
  for (const e of errors) console.error(`  - ${e}`);
  process.exit(1);
}

console.log(`[placeholders:check] OK for SITE_ENV=${SITE_ENV}.`);
