// Shared helpers for reading and indexing the 01-contratos-site package.
// Used by validate-contracts.mjs and by other scripts that need to check a
// build artifact against declared requirement ids.
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

// Resolved from this file's own location, not process.cwd(), so it works
// regardless of where `npm run` is invoked from. The contracts package
// lives at the repository root so a standalone clone includes the complete
// authoritative specification alongside the implementation.
const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const CONTRACTS_DIR = path.resolve(__dirname, "../../01-contratos-site");
const ID_LIKE = /^[A-Z]{2,}(-[A-Z0-9]+)+$/;

export function listContractFiles() {
  return readdirSync(CONTRACTS_DIR)
    .filter((f) => f.endsWith(".yml") || f.endsWith(".md"))
    .sort();
}

export function loadYamlContracts() {
  const files = listContractFiles().filter((f) => f.endsWith(".yml"));
  const docs = {};
  for (const file of files) {
    const raw = readFileSync(path.join(CONTRACTS_DIR, file), "utf8");
    docs[file] = yaml.load(raw);
  }
  return docs;
}

/** 00-README.md and 01-product-contract.md declare PRD/APP ids in prose,
 * not YAML. Scan every id-like token in the markdown files as "declared" so
 * cross-references from the YAML contracts do not falsely dangle. */
export function collectIdsDeclaredInMarkdown() {
  const files = listContractFiles().filter((f) => f.endsWith(".md"));
  const ids = new Set();
  for (const file of files) {
    const raw = readFileSync(path.join(CONTRACTS_DIR, file), "utf8");
    for (const token of raw.split(/[\s,.:;()]+/)) {
      if (ID_LIKE.test(token)) ids.add(token);
    }
  }
  return ids;
}

/** Collect every string that looks like a stable requirement id (e.g. CMP-HEADER, QA-VIS-001). */
export function collectDeclaredIds(docs) {
  const ids = new Set();
  const idLike = /^[A-Z]{2,}(-[A-Z0-9]+)+$/;

  function walk(node) {
    if (Array.isArray(node)) {
      node.forEach(walk);
    } else if (node && typeof node === "object") {
      for (const [key, value] of Object.entries(node)) {
        if ((key === "id" || key === "decision_id") && typeof value === "string" && idLike.test(value)) {
          ids.add(value);
        }
        walk(value);
      }
    }
  }
  Object.values(docs).forEach(walk);
  return ids;
}

/** Collect every string reference that looks like an id, wherever it appears (maps_to, origin, component:, model:, etc.). */
export function collectReferencedIds(docs) {
  const refs = new Set();
  const idLike = /^[A-Z]{2,}(-[A-Z0-9]+)+$/;

  function walk(node) {
    if (Array.isArray(node)) {
      node.forEach(walk);
    } else if (node && typeof node === "object") {
      for (const value of Object.values(node)) walk(value);
    } else if (typeof node === "string" && idLike.test(node)) {
      refs.add(node);
    }
  }
  Object.values(docs).forEach(walk);
  return refs;
}
