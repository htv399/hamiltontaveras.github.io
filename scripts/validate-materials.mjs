#!/usr/bin/env node
// REP-SCRIPT-003 / TCH-001 verification (QA-TCH-002, QA-TCH-003). Validates
// every Material fixture: required_for_local metadata, that the referenced
// local file actually exists under public/, and that its declared checksum
// and size match the real file (catches stale metadata after a file edit).
import path from "node:path";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { walkFiles, readFrontmatter } from "./lib/frontmatter.mjs";

const MATERIALS_DIR = path.resolve(process.cwd(), "src/content/materials");
const PUBLIC_DIR = path.resolve(process.cwd(), "public");

const mimeByType = {
  pdf: ["application/pdf"],
  html: ["text/html"],
  qmd: ["text/plain"],
  r: ["text/plain", "text/x-r-source"],
  rmd: ["text/plain", "text/x-r-markdown"],
  csv: ["text/csv"],
  xlsx: ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
  zip: ["application/zip"],
  notebook: ["application/x-ipynb+json"],
  external: []
};

let errors = [];
const files = walkFiles(MATERIALS_DIR, (f) => f.endsWith(".yml"));

for (const file of files) {
  const rel = path.relative(process.cwd(), file);
  const m = readFrontmatter(file);
  const isLocal = m.material_type !== "external";

  for (const field of ["id", "title", "slug", "material_type", "access", "status", "version", "license", "href"]) {
    if (!m[field]) errors.push(`${rel}: missing required field "${field}"`);
  }

  // Private fixtures validate their metadata but must not require a file in
  // public/, where it would become directly downloadable in production.
  if (isLocal && m.access !== "private") {
    for (const field of ["file_size_bytes", "mime_type", "checksum_sha256"]) {
      if (!m[field]) errors.push(`${rel}: missing required_for_local field "${field}"`);
    }
    if (m.mime_type && !mimeByType[m.material_type]?.includes(m.mime_type)) {
      errors.push(`${rel}: mime_type "${m.mime_type}" is not valid for material_type "${m.material_type}"`);
    }
    if (m.href?.startsWith("/")) {
      const filePath = path.join(PUBLIC_DIR, m.href);
      if (!existsSync(filePath)) {
        errors.push(`${rel}: href "${m.href}" does not resolve to a file under public/`);
      } else {
        const stat = statSync(filePath);
        if (m.file_size_bytes && stat.size !== m.file_size_bytes) {
          errors.push(`${rel}: file_size_bytes (${m.file_size_bytes}) does not match actual file size (${stat.size})`);
        }
        if (m.checksum_sha256) {
          const actual = createHash("sha256").update(readFileSync(filePath)).digest("hex");
          if (actual !== m.checksum_sha256) {
            errors.push(`${rel}: checksum_sha256 does not match the file's actual SHA-256 (${actual})`);
          }
        }
      }
    }
  }
}

if (errors.length > 0) {
  console.error(`[materials:check] FAIL — ${errors.length} issue(s):`);
  for (const e of errors) console.error(`  - ${e}`);
  process.exit(1);
}

console.log(`[materials:check] OK — ${files.length} material record(s) validated.`);
