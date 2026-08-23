#!/usr/bin/env node
// REP-SCRIPT-002 / CM-001 verification. Structural schema validation
// (required fields, enums) is already enforced by src/content.config.ts at
// astro build / astro check time. This script enforces the CM-VAL-* global
// rules zod cannot express: production placeholder gate, translation_key
// integrity and resource completeness. SITE_ENV=production makes
// CM-VAL-001 fatal; preview only reports it.
import path from "node:path";
import { walkFiles, readFrontmatter } from "./lib/frontmatter.mjs";

const SITE_ENV = process.env.SITE_ENV || "preview";
const CONTENT_DIR = path.resolve(process.cwd(), "src/content");

const collections = ["analysis", "research", "work", "notes", "courses", "weeks", "materials", "profiles"];
let errors = [];
let placeholderWarnings = [];

for (const collection of collections) {
  const dir = path.join(CONTENT_DIR, collection);
  const files = walkFiles(dir, (f) => f.endsWith(".mdx") || f.endsWith(".yml"));
  for (const file of files) {
    const rel = path.relative(process.cwd(), file);
    let fm;
    try {
      fm = readFrontmatter(file);
    } catch (err) {
      errors.push(`${rel}: could not parse frontmatter (${err.message})`);
      continue;
    }

    // CM-VAL-001
    if (fm.placeholder === true) {
      placeholderWarnings.push(rel);
      if (SITE_ENV === "production") {
        errors.push(`${rel}: CM-VAL-001 — placeholder:true is forbidden in production.`);
      }
    }

    // CM-VAL-008: no placeholder label leaking into title/summary in production.
    const placeholderLabelPattern = /pendiente|coming soon|próximamente|lorem ipsum|TODO/i;
    if (SITE_ENV === "production") {
      if (fm.title && placeholderLabelPattern.test(fm.title)) {
        errors.push(`${rel}: CM-VAL-008 — title contains a placeholder label.`);
      }
      if (fm.summary && placeholderLabelPattern.test(fm.summary)) {
        errors.push(`${rel}: CM-VAL-008 — summary contains a placeholder label.`);
      }
    }

    // CM-VAL-005: any declared source/resource needs access and license.
    for (const key of ["resources", "sources"]) {
      const list = Array.isArray(fm[key]) ? fm[key] : [];
      for (const resource of list) {
        if (typeof resource !== "object") continue;
        if (!resource.access) errors.push(`${rel}: CM-VAL-005 — ${key} entry "${resource.id ?? resource.label}" is missing access.`);
        if (!resource.license) errors.push(`${rel}: CM-VAL-005 — ${key} entry "${resource.id ?? resource.label}" is missing license.`);
      }
    }

    // CM-VAL-007: published valuation/market analysis requires disclosure.
    if (collection === "work" && fm.kind === "valuation" && !fm.disclosure) {
      errors.push(`${rel}: CM-VAL-007 — a valuation requires disclosure.`);
    }
  }
}

// CM-VAL-006: translation_key shared by two pieces requires different
// languages and no duplicate (same language) pairing.
const byTranslationKey = new Map();
for (const collection of collections) {
  const dir = path.join(CONTENT_DIR, collection);
  for (const file of walkFiles(dir, (f) => f.endsWith(".mdx") || f.endsWith(".yml"))) {
    const fm = readFrontmatter(file);
    if (!fm.translation_key) continue;
    const rel = path.relative(process.cwd(), file);
    const bucket = byTranslationKey.get(fm.translation_key) ?? [];
    bucket.push({ rel, language: fm.language });
    byTranslationKey.set(fm.translation_key, bucket);
  }
}
for (const [key, entries] of byTranslationKey) {
  const languagesUsed = entries.map((e) => e.language);
  const duplicateLanguage = new Set(languagesUsed).size !== languagesUsed.length;
  if (duplicateLanguage) {
    errors.push(`translation_key "${key}" is shared by two pieces in the same language: ${entries.map((e) => e.rel).join(", ")}`);
  }
}

if (placeholderWarnings.length > 0) {
  console.log(`[content:validate] ${placeholderWarnings.length} placeholder:true fixture(s) found (expected in preview, SEED-001).`);
}

if (errors.length > 0) {
  console.error(`[content:validate] FAIL — ${errors.length} issue(s):`);
  for (const e of errors) console.error(`  - ${e}`);
  process.exit(1);
}

console.log(`[content:validate] OK — content passes CM-VAL rules for SITE_ENV=${SITE_ENV}.`);
